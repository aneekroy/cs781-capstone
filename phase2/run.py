"""CS781 Phase 2 (L-NER) command line.

  stats     annotation conventions, label statistics, offset-alignment ceiling
  cv        2-fold CV on the public folds (fold1<->fold2) -> best epoch count,
            per-label scores, error breakdown, post-processing choice; with
            --final it then retrains on both folds, predicts fold3 and writes
            submission.zip
  predict   predict any text file with a saved model (optionally save token
            log-probs for ensembling)
  ensemble  average saved token log-probs of several models (same tokenizer) and
            decode once with BIOES-constrained Viterbi

Knowledge distillation: pass --teacher <saved model.pt> (optionally with
--kd-unlabeled <official Phase-1 train json> for teacher-labelled extra text) and a
smaller student via --keep-layers on any Phase-1 checkpoint that shares the
teacher's tokenizer.
"""
import argparse
import copy
import json
import math
import random
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import (SEGMENTS, alignment_ceiling, dataset_stats, encode_document, load_gold, load_labels, load_texts,
                   make_windows, segment_tokenizer, spans_to_tags, tag_names, tags_to_spans, write_json)
from .model import CRF, Encoder, Tagger, load_phase1
from .peft import apply_method, count_parameters
from .post import (BoundaryRules, build_gazetteer, coarse_gold, coarse_labels, error_breakdown, postprocess,
                   remove_overlaps, resolve_roles, strict_scores)

PERSON_ROLES = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT"]


def device_and_amp():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return device, torch.bfloat16
    if device.type == "cuda":
        return device, torch.float16
    return device, None


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


# --------------------------------------------------------------------------- data
def load_split(data_dir, fold, labels):
    data_dir = Path(data_dir)
    texts = load_texts(data_dir / "Train" / f"fold{fold}-text.json")
    gold, report = load_gold(data_dir / "Train" / f"fold{fold}-gold.json", texts, labels)
    return texts, gold, report


def examples_for(tokenizer, texts, gold, labels):
    label_index = {l: i + 1 for i, l in enumerate(labels)}
    out = {}
    for doc, text in texts.items():
        ids, offsets = encode_document(tokenizer, text)
        row = {"ids": ids, "offsets": offsets}
        if gold is not None:
            spans = gold.get(doc, [])
            row["tags"], row["conflicts"] = spans_to_tags(offsets, spans, labels)
            pairs = []
            for s, e, label in spans:
                members = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
                if members:
                    pairs.append((members[0], members[-1], label_index[label]))
            row["pairs"] = pairs
        out[doc] = row
    return out


def body_window(args):
    """(window, stride, header) over document tokens. With --header-tokens H every window after the
    first is fed as [first H tokens of the document] + [window - H document tokens], so the total
    length (and the positions the model sees) stays at --window; outputs for the header are dropped."""
    header = getattr(args, "header_tokens", 0) or 0
    window = args.window - header
    return window, min(args.stride, window), header


def window_input(ids, start, length, header):
    """Encoder input for the window ids[start:start+length] and the number of leading header tokens."""
    if header and start > 0:
        return ids[:header] + ids[start:start + length], header
    return ids[start:start + length], 0


ROLE_LABELS = ("APP", "RESP", "WIT")


def entry_bonus(tag_names_, bonus):
    """[K] vector for CRF.viterbi(entry_bonus=...) or None.

    bonus: a number (every label) or {label: value}; applied to B-/S- tags only (starting an
    entity), so it adds entities rather than lengthening them. Old models store 0.0."""
    if not bonus or not isinstance(bonus, dict):
        return None            # 0.0, or a float from the withdrawn 'O' penalty (models before the entry bonus)
    vec = torch.zeros(len(tag_names_))
    for i, name in enumerate(tag_names_):
        if name[:2] in ("B-", "S-"):
            label = name[2:]
            vec[i] = bonus.get(label, 0.0)
    return vec if vec.any() else None


def decode(viterbi, tag_names_, scores, bonus):
    return viterbi(scores, entry_bonus(tag_names_, bonus)) if scores.size(0) else []


def training_windows(examples, window, stride, empty_keep, rare_tags, rare_repeat, rng, header=0):
    items = []
    for doc, row in examples.items():
        n = len(row["ids"])
        if n == 0:
            continue
        for start in make_windows(n, window, stride):
            tags = row["tags"][start:start + window]
            if not any(tags) and rng.random() > empty_keep:
                continue
            repeat = rare_repeat if rare_tags and any(t in rare_tags for t in tags) else 1
            items += [(doc, start, min(window, n), header if header and start > 0 else 0)] * repeat
    rng.shuffle(items)
    return items


def batches_by_length(items, batch_size, rng):
    groups = {}
    for item in items:
        groups.setdefault(tuple(item[2:]), []).append(item)
    out = []
    for group in groups.values():
        out += [group[i:i + batch_size] for i in range(0, len(group), batch_size)]
    rng.shuffle(out)
    return out


def _stream_records(path):
    """Stream the (1+ GB) Phase-1 json with phase1.data.records when available."""
    try:
        from phase1.data import records
        yield from records(path)
    except ImportError:
        from .data import read_json
        data = read_json(path)
        yield from (data if isinstance(data, list) else data.values())


def unlabeled_examples(path, tokenizer, n_docs, seed, ner_dir=None, min_chars=2000, take_prob=0.05):
    """Teacher-labelled extra text for KD: a random sample of official Phase-1 documents.

    Documents that contain an NER document (any fold, including test) are skipped: the teacher
    was trained on both training folds, so labelling a validation-fold document for the student
    would leak gold labels into cross-validation, and test text is never used."""
    from .overlap import NerFingerprints
    fingerprints = NerFingerprints.from_data(ner_dir) if ner_dir else None
    rng = random.Random(seed)
    out, skipped = {}, 0
    for record in _stream_records(path):
        if rng.random() > take_prob:
            continue
        text = record["text"] if isinstance(record, dict) else record
        if len(text) < min_chars:
            continue
        if fingerprints is not None and fingerprints.overlaps(text):
            skipped += 1
            continue
        ids, offsets = encode_document(tokenizer, text)
        out[f"u{len(out)}"] = {"ids": ids, "offsets": offsets, "tags": [0] * len(ids)}
        if len(out) >= n_docs:
            break
    print(f"KD unlabeled: {len(out)} documents, {skipped} skipped because they contain an NER document",
          flush=True)
    return out


# --------------------------------------------------------------------------- model
def build_tagger(args, labels):
    lm, tokenizer, _ = load_phase1(args.checkpoint, args.project_root)
    tokenizer = segment_tokenizer(tokenizer, getattr(args, "segment", "bpe"))   # models saved before --segment: bpe
    encoder = Encoder(lm, mode=args.mode, keep_layers=args.keep_layers,
                      prompt_tokens=args.prompt_tokens if args.method == "prompt" else 0,
                      gradient_checkpointing=args.gradient_checkpointing)
    del lm
    peft_info = apply_method(encoder, args.method, rank=args.rank, alpha=args.alpha, top_k=args.top_k,
                             bottleneck=args.bottleneck)
    tagger = Tagger(encoder, labels, head=args.head, crf=args.crf, span=args.span, max_width=args.max_width,
                    dropout=args.dropout)
    info = {"peft": peft_info, "parameters": count_parameters(tagger),
            "encoder_parameters": count_parameters(encoder), "architecture": encoder.architecture,
            "mode": encoder.mode, "layers": len(encoder.blocks), "segment": getattr(args, "segment", "bpe")}
    return tagger, tokenizer, info


def save_model(path, tagger, args, labels, extra):
    payload = {"args": vars(args), "labels": labels, "state": tagger.state_dict(), **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    tagger, tokenizer, _ = build_tagger(args, payload["labels"])
    tagger.load_state_dict(payload["state"])
    return tagger.to(device).eval(), tokenizer, payload


# --------------------------------------------------------------------------- inference
def window_weights(length, n_total, start):
    """Down-weight window edges, where a token has little context on one side."""
    w = torch.ones(length)
    ramp = max(1, length // 4)
    pos = torch.arange(length, dtype=torch.float32)
    if start > 0:
        w = torch.minimum(w, (pos + 1) / ramp)
    if start + length < n_total:
        w = torch.minimum(w, (length - pos) / ramp)
    return w.clamp(min=0.05)


@torch.no_grad()
def document_logprobs(tagger, ids, window, stride, device, dtype, batch_size=16, header=0):
    """Window-averaged per-token scores for a whole document: [T, K] float32 on CPU."""
    n = len(ids)
    if n == 0:
        return torch.zeros(0, len(tagger.tag_names))
    starts = make_windows(n, window, stride)
    total = torch.zeros(n, len(tagger.tag_names))
    weight = torch.zeros(n, 1)
    length = min(window, n)
    chunks = []
    for group in ([s for s in starts if s == 0 or not header], [s for s in starts if s > 0 and header]):
        chunks += [group[i:i + batch_size] for i in range(0, len(group), batch_size)]
    for chunk in chunks:
        inputs = [window_input(ids, s, length, header) for s in chunk]
        prefix = inputs[0][1]
        batch = torch.tensor([x for x, _ in inputs], device=device)
        with autocast(device, dtype):
            emissions = tagger.emissions(batch)[:, prefix:].float()
        if not tagger.use_crf:
            emissions = emissions.log_softmax(-1)
        for row, s in enumerate(chunk):
            w = window_weights(length, n, s).unsqueeze(1)
            total[s:s + length] += emissions[row].cpu() * w
            weight[s:s + length] += w
    return total / weight


@torch.no_grad()
def document_span_predictions(tagger, ids, offsets, window, stride, device, dtype, batch_size=8):
    n = len(ids)
    if n == 0:
        return []
    starts = make_windows(n, window, stride)
    margin = max(0, (window - stride) // 2)
    labels = tagger.labels
    length = min(window, n)
    scored = {}
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        batch = torch.tensor([ids[s:s + length] for s in chunk], device=device)
        with autocast(device, dtype):
            scores, valid = tagger.head(tagger.encoder(batch))
        probs = scores.float().softmax(-1)
        best_p, best_c = probs.max(-1)
        hit = (best_c > 0) & valid.unsqueeze(0)
        for row, s in enumerate(chunk):
            own_lo = s + (margin if s > 0 else 0)
            own_hi = s + length - (margin if s + length < n else 0)
            for a, b in hit[row].nonzero().tolist():
                ga, gb = s + a, s + b
                if not own_lo <= ga < own_hi:
                    continue
                cs, ce = offsets[ga][0], offsets[gb][1]
                if ce <= cs:
                    continue
                span = (cs, ce, labels[int(best_c[row, a, b]) - 1])
                scored[span] = max(scored.get(span, 0.0), float(best_p[row, a, b]))
    return remove_overlaps(list(scored), scored)


def predict_documents(tagger, examples, args, device, dtype, keep_logprobs=False, o_bias=0.0):
    tagger.eval()
    window, stride, header = body_window(args)
    preds, logprobs = {}, {}
    for doc, row in examples.items():
        if tagger.span:
            preds[doc] = document_span_predictions(tagger, row["ids"], row["offsets"], args.window, args.stride,
                                                   device, dtype)
            continue
        scores = document_logprobs(tagger, row["ids"], window, stride, device, dtype, header=header)
        tags = decode(tagger.crf.viterbi, tagger.tag_names, scores.to(device), o_bias)
        preds[doc] = sorted(set(tags_to_spans(row["offsets"], tags, tagger.labels)))
        if keep_logprobs:
            logprobs[doc] = (scores.log_softmax(-1) if tagger.use_crf else scores).half()
    return preds, logprobs


# --------------------------------------------------------------------------- training
def class_weights(examples, num_tags, device):
    counts = Counter(t for row in examples.values() for t in row["tags"])
    total = sum(counts.values())
    w = torch.tensor([(counts.get(k, 0) / total + 1e-6) ** -0.5 for k in range(num_tags)])
    w = (w / w[0]).clamp(1.0, 10.0)
    return w.to(device)


def train(args, labels, train_texts, train_gold, val_texts=None, val_gold=None, epochs=None, log=None):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device, dtype = device_and_amp()
    tagger, tokenizer, info = build_tagger(args, labels)
    tagger.to(device)
    teacher = None
    if args.teacher:
        teacher, teacher_tok, teacher_payload = load_model(args.teacher, device)
        if teacher_tok.get_vocab() != tokenizer.get_vocab() or teacher.tag_names != tagger.tag_names:
            raise ValueError("Teacher and student must share tokenizer and tag set for token-level KD")
        teacher_segment = teacher_payload["args"].get("segment", "bpe")
        if teacher_segment != args.segment:
            raise ValueError(f"Teacher used --segment {teacher_segment}, student --segment {args.segment}: "
                             "token positions would not line up")
        if tagger.span or teacher.span:
            raise ValueError("Token-level KD needs token heads on both teacher and student")
        teacher_total = sum(p.numel() for p in teacher.parameters())
        info["teacher_parameters"] = teacher_total
        info["student_to_teacher"] = round(info["parameters"]["total"] / teacher_total, 3)
        print(f"KD: student {info['parameters']['total']:,} params vs teacher {teacher_total:,} "
              f"({info['student_to_teacher']:.0%} of teacher)", flush=True)
        if info["parameters"]["total"] >= teacher_total:
            print("WARNING: the student is not smaller than the teacher; lower --keep-layers", flush=True)
        for p in teacher.parameters():
            p.requires_grad = False
    if getattr(args, "augment", 0):
        from .data import augment_documents
        n_before = len(train_texts)
        train_texts, train_gold = augment_documents(train_texts, train_gold, args.augment, random.Random(args.seed + 1))
        print(f"augment: {n_before} training documents -> {len(train_texts)} (entity replacement)", flush=True)
    train_ex = examples_for(tokenizer, train_texts, train_gold, labels)
    val_ex = examples_for(tokenizer, val_texts, None, labels) if val_texts else None
    unlabeled = unlabeled_examples(args.kd_unlabeled, tokenizer, args.kd_unlabeled_docs, args.seed, args.data) \
        if teacher is not None and args.kd_unlabeled else {}
    names = tag_names(labels)
    counts = Counter(l for spans in train_gold.values() for *_, l in spans)
    rare_labels = [l for l, _ in sorted(counts.items(), key=lambda kv: kv[1])[:args.rare_labels]]
    rare_tags = {i for i, n in enumerate(names) if n != "O" and n.split("-", 1)[1] in rare_labels}
    weights = class_weights(train_ex, len(names), device) if args.class_weights and not tagger.use_crf else None
    encoder_params = [p for p in tagger.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in tagger.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    lr_encoder = args.lr_full if args.method in {"full", "partial", "normfit"} else args.lr_peft
    groups = [g for g in ({"params": encoder_params, "lr": lr_encoder}, {"params": head_params, "lr": args.lr_head})
              if g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    epochs = epochs or args.epochs
    window, stride, header = body_window(args)
    if header and tagger.span:
        raise ValueError("--header-tokens needs a token head (not --span)")
    first = training_windows(train_ex, window, stride, args.empty_keep, rare_tags, args.rare_repeat, rng, header)
    n_unlabeled = min(sum(len(make_windows(len(r["ids"]), window, window)) for r in unlabeled.values()),
                      args.kd_unlabeled_windows) if unlabeled else 0
    steps_per_epoch = math.ceil(len(first) / args.batch_size) + math.ceil(n_unlabeled / args.batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(args.warmup * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best = {"macro_f1": -1.0, "epoch": 0, "state": None, "preds": None, "scores": None}
    history, stale = [], 0
    print(json.dumps({"train_windows": len(first), "unlabeled_windows": n_unlabeled, "rare_labels": rare_labels,
                      "alignment_conflicts": sum(r["conflicts"] for r in train_ex.values()), **info}), flush=True)
    for epoch in range(1, epochs + 1):
        tagger.train()
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        items = first if epoch == 1 else training_windows(train_ex, window, stride, args.empty_keep,
                                                          rare_tags, args.rare_repeat, rng, header)
        jobs = [("gold", b) for b in batches_by_length(items, args.batch_size, rng)]
        if unlabeled:
            u_items = []
            for doc, row in unlabeled.items():
                u_items += [(doc, s, min(window, len(row["ids"])), header if header and s > 0 else 0)
                            for s in make_windows(len(row["ids"]), window, window)]
            rng.shuffle(u_items)
            jobs += [("unlabeled", b) for b in batches_by_length(u_items[:args.kd_unlabeled_windows],
                                                                  args.batch_size, rng)]
            rng.shuffle(jobs)
        losses = []
        for kind, batch in jobs:
            source = train_ex if kind == "gold" else unlabeled
            length, prefix = batch[0][2], batch[0][3]
            ids = torch.tensor([window_input(source[d]["ids"], s, length, header)[0] for d, s, *_ in batch],
                               device=device)
            with autocast(device, dtype):
                if tagger.span:
                    scores, valid = tagger.head(tagger.encoder(ids))
                    target = torch.zeros(ids.shape + (length,), dtype=torch.long, device=device)
                    for r, (d, s, *_) in enumerate(batch):
                        for a, b, c in source[d]["pairs"]:
                            if s <= a and b < s + length and b - a < args.max_width:
                                target[r, a - s, b - s] = c
                    mask = valid.unsqueeze(0).expand_as(target)
                    loss = F.cross_entropy(scores[mask].float(), target[mask])
                else:
                    emissions = tagger.emissions(ids)[:, prefix:]
                    loss = torch.zeros((), device=device)
                    if kind == "gold":
                        tags = torch.tensor([source[d]["tags"][s:s + length] for d, s, *_ in batch], device=device)
                        loss = tagger.token_loss(emissions, tags, weights)
                    if teacher is not None:
                        # Soft-label KD on per-token tag distributions (Hinton et al., 2015).
                        with torch.no_grad():
                            target = (teacher.emissions(ids[:, prefix:]).float() / args.kd_temperature).softmax(-1)
                        student = (emissions.float() / args.kd_temperature).log_softmax(-1)
                        kd = F.kl_div(student.flatten(0, 1), target.flatten(0, 1), reduction="batchmean") \
                            * args.kd_temperature ** 2
                        alpha = args.kd_alpha if kind == "gold" else 0.0
                        loss = alpha * loss + (1 - alpha) * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in tagger.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - started, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device.type == "cuda" else None}
        if val_ex is not None and (epoch >= args.eval_from or epoch == epochs):
            preds, _ = predict_documents(tagger, val_ex, args, device, dtype)
            score = strict_scores(val_gold, preds, labels)
            row["val_macro_f1"] = round(score["macro_f1"], 4)
            if score["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": score["macro_f1"], "epoch": epoch, "preds": preds, "scores": score,
                        "state": {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()}}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if log:
            with open(log, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        if val_ex is not None and stale >= args.patience:
            break
    if best["state"] is not None:
        tagger.load_state_dict(best["state"])
    return tagger, tokenizer, info, history, best


# --------------------------------------------------------------------------- post-processing selection
def choose_postprocessing(fold_runs, labels):
    """Greedy forward selection on pooled out-of-fold CV predictions."""
    def pooled(config):
        gold, pred = {}, {}
        for run in fold_runs:
            for doc, spans in run["raw"].items():
                pred[doc] = postprocess(run["texts"][doc], spans, config, run["boundary"], run["gazetteer"])
                gold[doc] = run["gold"][doc]
        return strict_scores(gold, pred, labels)["macro_f1"]

    config = {"boundary": [], "vote": [], "counsel": [], "parties": [], "witness": [], "judges": [], "caseno": [],
              "propagate": [],
              "propagate_ci": [], "parts": [], "prec": [], "rules": []}
    base = pooled(config)
    trace = [("raw", round(base, 4))]
    options = [("boundary", l) for l in labels] + [("vote", l) for l in PERSON_ROLES if l in labels] + \
              ([("counsel", "side")] if {"A.COUNSEL", "R.COUNSEL"} <= set(labels) else []) + \
              ([("parties", "on"), ("parties", "first"), ("parties", "only")] if {"APP", "RESP"} <= set(labels) else []) + \
              ([("witness", "on")] if "WIT" in labels else []) + \
              ([("judges", "on")] if "JUDGE" in labels else []) + \
              ([("caseno", "split")] if "CASENO" in labels else []) + \
              [("propagate", l) for l in labels] + \
              [("propagate_ci", l) for l in labels] + \
              [("parts", l) for l in PERSON_ROLES if l in labels] + \
              ([("prec", "extend"), ("prec", "trim")] if "PREC" in labels else []) + \
              [("rules", l) for l in ["DATE", "CASENO", "STAT", "COURT", "AUTH"] if l in labels]
    for key, label in options:
        trial = copy.deepcopy(config)
        if key == "prec" and trial["prec"]:
            continue                                   # extend and trim are alternatives
        if key == "parties" and label != "on" and "on" not in trial["parties"]:
            continue                                   # 'first' / 'only' refine an accepted 'on'
        trial[key].append(label)
        score = pooled(trial)
        if score > base + 1e-4:
            config, base = trial, score
            trace.append((f"{key}:{label}", round(score, 4)))
    return config, trace


ENTRY_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)


def choose_o_bias(fold_runs, labels):
    """Entity-entry bonus picked on pooled out-of-fold scores, in two steps: one value for the
    under-recalled party roles (APP/RESP/WIT), then one for every other label.
    fold_runs[i]["scores"]: {doc: ([T,K] scores, offsets)}; fold_runs[i]["decode"]: CPU CRF.viterbi."""
    viterbi = torch.no_grad()(fold_runs[0]["decode"])
    names = tag_names(labels)
    gold = {d: g for run in fold_runs for d, g in run["gold"].items()}

    def evaluate(bonus):
        pred = {}
        for run in fold_runs:
            for doc, (scores, offsets) in run["scores"].items():
                pred[doc] = sorted(set(tags_to_spans(offsets, decode(viterbi, names, scores, bonus), labels)))
        return strict_scores(gold, pred, labels)["macro_f1"], pred

    roles = [l for l in ROLE_LABELS if l in labels]
    others = [l for l in labels if l not in roles]
    grid, best = [], (None, -1.0, None)
    role_value = 0.0
    for group, members in (("roles", roles), ("others", others)):
        results = []
        for value in ENTRY_GRID:
            bonus = {**{l: role_value for l in roles}, **{l: value for l in members}}
            score, pred = evaluate(bonus)
            grid.append((group, value, round(score, 4)))
            results.append((score, -value, value, pred, bonus))
        score, _, value, pred, bonus = max(results, key=lambda r: (r[0], r[1]))   # ties -> smaller bonus
        if group == "roles":
            role_value = value
        best = (bonus, score, pred)
    bonus, score, pred = best
    bonus = {l: v for l, v in bonus.items() if v}
    return bonus, score, pred, grid


# --------------------------------------------------------------------------- submission
def to_output(spans, end_inclusive):
    return [[s, e - 1 if end_inclusive else e, label] for s, e, label in sorted(spans)]


def write_submission(predictions, texts, out_dir, end_inclusive, labels=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {doc: to_output(predictions.get(doc, []), end_inclusive) for doc in texts}
    for doc, spans in payload.items():
        for s, e, label in spans:
            end = e + 1 if end_inclusive else e
            if not (0 <= s < end <= len(texts[doc])):
                raise ValueError(f"Bad offsets in {doc}: {s},{e}")
            if labels is not None and label not in labels:
                raise ValueError(f"Unknown label {label} in {doc}")
    write_json(out_dir / "prediction.json", payload)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    with zipfile.ZipFile(out_dir / "submission.zip") as z:
        assert z.namelist() == ["prediction.json"]
        assert set(json.loads(z.read("prediction.json"))) == set(texts)
    return out_dir / "submission.zip"


# --------------------------------------------------------------------------- commands
def cmd_stats(args):
    labels = load_labels(Path(args.data) / "Train" / "labels.txt")
    tokenizer = load_phase1(args.checkpoint, args.project_root)[1] if args.checkpoint else None
    report, texts_all, gold_all = {}, {}, {}
    for fold in (1, 2):
        texts, gold, conv = load_split(args.data, fold, labels)
        report[f"fold{fold}_convention"] = conv
        texts_all.update(texts)
        gold_all.update(gold)
    report["stats"] = dataset_stats(texts_all, gold_all, labels, tokenizer)
    if tokenizer is not None:
        boundary = BoundaryRules().fit(texts_all, gold_all)
        report["alignment"] = {}
        for mode in SEGMENTS:
            seg = segment_tokenizer(tokenizer, mode)
            report["alignment"][mode] = {
                "raw": alignment_ceiling(texts_all, gold_all, seg, labels),
                "with_boundary_rules": alignment_ceiling(texts_all, gold_all, seg, labels, boundary)}
    test = load_texts(Path(args.data) / "Test" / "fold3-text.json")
    report["test_documents"] = len(test)
    report["test_doc_chars_median"] = sorted(map(len, test.values()))[len(test) // 2]
    write_json(args.output, report)
    conv = report["fold1_convention"]
    print(f"documents: train {report['stats']['documents']} | test {report['test_documents']} | "
          f"entities {report['stats']['entities']} | overlapping pairs {report['stats']['overlapping_pairs']}")
    print(f"end offsets: {'INCLUSIVE' if conv['end_inclusive'] else 'exclusive'} {conv['convention_votes']}"
          + (f" | unknown labels {conv['unknown_labels']}" if conv["unknown_labels"] else ""))
    if "doc_tokens_median" in report["stats"]:
        print(f"document tokens: median {report['stats']['doc_tokens_median']}, max {report['stats']['doc_tokens_max']}")
    if tokenizer is not None:
        print("alignment ceiling (share of gold spans a perfect tagger can reproduce exactly):")
        for mode in SEGMENTS:
            row = report["alignment"][mode]
            print(f"  --segment {mode:5s}: raw {row['raw']['exact_rate']:.4f} | with boundary rules "
                  f"{row['with_boundary_rules']['exact_rate']:.4f} | missed {row['raw']['missed_by_label']}")
        for example in report["alignment"]["punct"]["raw"]["examples"][:5]:
            print("   still misaligned with punct:", example)
    print(f"\n{'label':10s} {'count':>6s} {'docs':>5s} {'chars med/max':>14s} {'mention cov':>11s}  last chars")
    for label, row in report["stats"]["labels"].items():
        if not row.get("count"):
            print(f"{label:10s} {0:6d}  (no gold examples)")
            continue
        cov = "-" if row["mention_coverage"] is None else f"{row['mention_coverage']:.3f}"
        last = " ".join(repr(c) for c, _ in row["top_last_chars"][:3])
        print(f"{label:10s} {row['count']:6d} {row['docs']:5d} {row['chars_median']:>6d}/{row['chars_max']:<7d} "
              f"{cov:>11s}  {last}")
    print(f"\nfull report: {args.output}")


def cmd_cv(args):
    labels = load_labels(Path(args.data) / "Train" / "labels.txt")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "args.json", vars(args))
    folds = {f: load_split(args.data, f, labels) for f in (1, 2)}
    end_inclusive = folds[1][2]["end_inclusive"]
    if folds[2][2]["end_inclusive"] != end_inclusive:
        raise ValueError("Folds disagree on the end-offset convention")
    o_bias_auto = str(args.o_bias).lower() == "auto"
    coarse = getattr(args, "coarse_roles", False)
    train_labels = coarse_labels(labels) if coarse else labels
    as_train = coarse_gold if coarse else (lambda g: g)
    if coarse and o_bias_auto:
        print("--coarse-roles: entity-entry bonus not tuned (scores are over merged labels)", flush=True)
        o_bias_auto = False
    if getattr(args, "header_tokens", 0) and args.header_tokens * 2 > args.window:
        raise ValueError("--header-tokens must be at most half of --window")
    if not args.final_only:
        fold_runs, summary = [], {"folds": []}
        for train_fold, val_fold in ((1, 2), (2, 1)):
            tr_texts, tr_gold, _ = folds[train_fold]
            va_texts, va_gold, _ = folds[val_fold]
            tagger, fold_tokenizer, info, history, best = train(args, train_labels, tr_texts, as_train(tr_gold), va_texts,
                                                                as_train(va_gold),
                                                                log=out / f"train_{train_fold}to{val_fold}.jsonl")
            if coarse:                              # merged roles -> the 12 labels, scored against the real gold
                best["preds"] = {d: sorted(set(resolve_roles(va_texts[d], sp))) for d, sp in best["preds"].items()}
                best["scores"] = strict_scores(va_gold, best["preds"], labels)
                print(f"  resolved merged roles: fold macro-F1 {best['macro_f1']:.4f} (merged labels) -> "
                      f"{best['scores']['macro_f1']:.4f} (12 labels)", flush=True)
                best["macro_f1"] = best["scores"]["macro_f1"]
            fold_runs.append({"raw": best["preds"], "texts": va_texts, "gold": va_gold,
                              "boundary": BoundaryRules().fit(tr_texts, tr_gold),
                              "gazetteer": build_gazetteer(tr_texts, tr_gold)})
            if o_bias_auto and not tagger.span:     # keep the best epoch's scores to tune the 'O' penalty
                device, dtype = device_and_amp()
                tagger.to(device).eval()
                window, stride, header = body_window(args)
                fold_runs[-1]["scores"] = {}
                for doc, row in examples_for(fold_tokenizer, va_texts, None, labels).items():
                    fold_runs[-1]["scores"][doc] = (document_logprobs(tagger, row["ids"], window, stride, device,
                                                                      dtype, header=header), row["offsets"])
                fold_runs[-1]["decode"] = copy.deepcopy(tagger.crf).float().cpu().viterbi   # no GPU model kept alive
            summary["folds"].append({"train": train_fold, "val": val_fold, "best_epoch": best["epoch"],
                                     "macro_f1": best["macro_f1"], "per_label": best["scores"]["per_label"],
                                     "errors": error_breakdown(va_texts, va_gold, best["preds"], labels),
                                     "info": info, "history": history})
            del tagger
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        o_bias = {} if o_bias_auto else ({l: float(args.o_bias) for l in labels} if float(args.o_bias) else {})
        if o_bias_auto and "scores" in fold_runs[0]:
            o_bias, biased, preds, grid = choose_o_bias(fold_runs, labels)
            for run in fold_runs:
                run["raw"] = {doc: preds[doc] for doc in run["raw"]}
                run.pop("scores"), run.pop("decode")
            summary["o_bias_grid"] = grid
            print(f"entity-entry bonus: {o_bias or 'none'} (pooled out-of-fold raw macro-F1 {grid[0][2]} -> "
                  f"{round(biased, 4)})", flush=True)
        summary["o_bias"] = o_bias
        config, trace = choose_postprocessing(fold_runs, labels)
        summary.update(cv_macro_f1_raw=float(np.mean([f["macro_f1"] for f in summary["folds"]])),
                       cv_macro_f1_post=trace[-1][1], postprocess=config, postprocess_trace=trace,
                       best_epochs=[f["best_epoch"] for f in summary["folds"]])
        write_json(out / "cv_summary.json", summary)
        torch.save({"oof": {k: v for r in fold_runs for k, v in r["raw"].items()}}, out / "oof_predictions.pt")
        print("CV raw macro-F1:", round(summary["cv_macro_f1_raw"], 4), "| post-processing trace:", trace, flush=True)
    else:
        summary = json.loads((out / "cv_summary.json").read_text())
        config = summary["postprocess"]
        o_bias = summary.get("o_bias", 0.0)
    if not (args.final or args.final_only):
        return
    epochs = max(1, round(float(np.mean(summary["best_epochs"])) * args.final_epoch_scale))
    texts = {**folds[1][0], **folds[2][0]}
    gold = {**folds[1][1], **folds[2][1]}
    tagger, tokenizer, info, _, _ = train(args, train_labels, texts, as_train(gold), epochs=epochs,
                                          log=out / "train_final.jsonl")
    boundary = BoundaryRules().fit(texts, gold)
    gazetteer = build_gazetteer(texts, gold)
    save_model(out / "model.pt", tagger, args, train_labels,
               {"postprocess": config, "boundary": boundary.state(), "gazetteer": gazetteer,
                "end_inclusive": end_inclusive, "final_epochs": epochs, "info": info, "o_bias": o_bias,
                "fine_labels": labels})
    predict_and_write(tagger, tokenizer, args, train_labels, config, boundary, gazetteer, end_inclusive,
                      Path(args.data) / "Test" / "fold3-text.json", out / "submission", keep_logprobs=True,
                      o_bias=o_bias, fine_labels=labels)


def predict_and_write(tagger, tokenizer, args, labels, config, boundary, gazetteer, end_inclusive, test_path,
                      out_dir, keep_logprobs=False, o_bias=0.0, fine_labels=None):
    device, dtype = device_and_amp()
    texts = load_texts(test_path)
    examples = examples_for(tokenizer, texts, None, labels)
    started = time.time()
    raw, logprobs = predict_documents(tagger.to(device), examples, args, device, dtype, keep_logprobs, o_bias)
    if getattr(args, "coarse_roles", False):
        raw = {d: sorted(set(resolve_roles(texts[d], raw[d]))) for d in raw}
    out_labels = fine_labels or labels
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(out_dir) / "raw_spans.json", {d: [list(x) for x in sp] for d, sp in raw.items()})
    final = {d: postprocess(texts[d], raw[d], config, boundary, gazetteer) for d in texts}
    archive = write_submission(final, texts, out_dir, end_inclusive, out_labels)
    if keep_logprobs and logprobs:
        torch.save({"logprobs": logprobs, "offsets": {d: e["offsets"] for d, e in examples.items()},
                    "labels": labels, "o_bias": o_bias}, Path(out_dir) / "token_logprobs.pt")
    counts = Counter(l for spans in final.values() for *_, l in spans)
    write_json(Path(out_dir) / "prediction_summary.json",
               {"documents": len(texts), "entities": sum(counts.values()), "by_label": dict(counts),
                "seconds": round(time.time() - started, 1), "postprocess": config, "o_bias": o_bias})
    print("Wrote", archive, dict(counts), flush=True)
    return final


def cmd_predict(args):
    device, _ = device_and_amp()
    tagger, tokenizer, payload = load_model(args.model, device)
    saved = argparse.Namespace(**payload["args"])
    predict_and_write(tagger, tokenizer, saved, payload["labels"], payload["postprocess"],
                      BoundaryRules.from_state(payload["boundary"]), payload["gazetteer"],
                      payload["end_inclusive"], args.test, args.output, keep_logprobs=args.save_logprobs,
                      o_bias=payload.get("o_bias", 0.0), fine_labels=payload.get("fine_labels"))


def cmd_ensemble(args):
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.logprobs]
    labels = parts[0]["labels"]
    for part in parts[1:]:
        if part["labels"] != labels or part["offsets"].keys() != parts[0]["offsets"].keys():
            raise ValueError("Ensemble members must share labels and documents")
        if any(part["offsets"][d] != parts[0]["offsets"][d] for d in part["offsets"]):
            raise ValueError("Ensemble members must share the tokenizer (token offsets differ)")
    weights = args.weights or [1.0] * len(parts)
    o_bias = {l: args.o_bias for l in labels} if args.o_bias is not None else parts[0].get("o_bias", 0.0)
    crf_names = tag_names(labels)
    crf = CRF(crf_names, learn=False)
    payload = torch.load(args.reference_model, map_location="cpu", weights_only=False)
    texts = load_texts(args.test)
    preds = {}
    for doc in texts:
        offsets = parts[0]["offsets"][doc]
        total = sum(w * p["logprobs"][doc].float() for w, p in zip(weights, parts)) / sum(weights)
        tags = decode(crf.viterbi, crf_names, total, o_bias)
        spans = tags_to_spans(offsets, tags, labels)
        preds[doc] = postprocess(texts[doc], spans, payload["postprocess"],
                                 BoundaryRules.from_state(payload["boundary"]), payload["gazetteer"])
    write_submission(preds, texts, args.output, payload["end_inclusive"], labels)
    print("Ensembled", len(parts), f"models (entry bonus {o_bias or 0}) ->", Path(args.output) / "submission.zip")


def add_model_args(p):
    p.add_argument("--data", required=True, help="Phase_2_LNER directory (contains Train/ and Test/)")
    p.add_argument("--checkpoint", required=True, help="Phase-1 best.pt (tokenizer.json alongside)")
    p.add_argument("--project-root", default=".", help="directory containing the phase1 package")
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="lorma",
                   choices=["full", "partial", "lora", "lorma", "adapter", "prompt", "normfit", "frozen"])
    p.add_argument("--mode", default="bidirectional", choices=["causal", "bidirectional", "encoder"])
    p.add_argument("--segment", default="punct", choices=list(SEGMENTS),
                   help="punct: isolate punctuation before BPE so merged tokens like '].' cannot cross an entity "
                        "boundary (raises the strict-F1 ceiling); bpe: the Phase-1 tokenization unchanged")
    p.add_argument("--head", default="bilstm", choices=["linear", "bilstm"])
    p.add_argument("--crf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--span", action="store_true", help="biaffine span head instead of BIOES tagging")
    p.add_argument("--max-width", type=int, default=48)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--prompt-tokens", type=int, default=20)
    p.add_argument("--keep-layers", type=int, help="truncate the Phase-1 model (KD students)")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--coarse-roles", action="store_true",
                   help="train on PARTY (APP+RESP) and COUNSEL (A./R.COUNSEL); sides are resolved afterwards from the "
                        "header parties and cue words, and everything is scored on the 12 labels")
    p.add_argument("--header-tokens", type=int, default=0,
                   help="prepend the document's first N tokens (parties and roles) to every later window; "
                        "the window then holds window-N document tokens")
    p.add_argument("--augment", type=int, default=0,
                   help="N entity-replaced copies of every training document (mention replacement)")
    p.add_argument("--o-bias", default="0",
                   help="bonus for starting an entity at decoding (recall vs precision); 'auto' tunes one value for "
                        "APP/RESP/WIT and one for the other labels on CV")
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-from", type=int, default=3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr-full", type=float, default=5e-5)
    p.add_argument("--lr-peft", type=float, default=1e-3)
    p.add_argument("--lr-head", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--empty-keep", type=float, default=0.3, help="fraction of entity-free windows kept")
    p.add_argument("--rare-labels", type=int, default=4)
    p.add_argument("--rare-repeat", type=int, default=2)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--teacher", help="saved Phase-2 model.pt to distil from")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--kd-unlabeled", help="official Phase-1 train json for teacher-labelled windows")
    p.add_argument("--kd-unlabeled-docs", type=int, default=300)
    p.add_argument("--kd-unlabeled-windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--final", action="store_true", help="after CV, retrain on both folds and predict fold3")
    p.add_argument("--final-only", action="store_true", help="reuse an existing cv_summary.json")
    p.add_argument("--final-epoch-scale", type=float, default=1.15,
                   help="2x data in the final fit -> slightly more epochs than the CV optimum")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    s.add_argument("--checkpoint")
    s.add_argument("--project-root", default=".")
    s.add_argument("--output", default="phase2_stats.json")
    add_model_args(sub.add_parser("cv"))
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--save-logprobs", action="store_true")
    e = sub.add_parser("ensemble")
    e.add_argument("--logprobs", nargs="+", required=True)
    e.add_argument("--weights", nargs="+", type=float)
    e.add_argument("--o-bias", type=float, help="entity-entry bonus at decoding (default: the first member's tuned value)")
    e.add_argument("--reference-model", required=True, help="model.pt whose post-processing/convention to use")
    e.add_argument("--test", required=True)
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    {"stats": cmd_stats, "cv": cmd_cv, "predict": cmd_predict, "ensemble": cmd_ensemble}[args.command](args)


if __name__ == "__main__":
    main()
