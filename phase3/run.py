"""CS781 Phase 3 (rhetorical roles) command line.

  stats    label counts per domain/split, sentence lengths, test id checks
  truth    write cl_dev_truth.json / it_dev_truth.json and an all-"Fact" dev prediction (format smoke test)
  cv       K-fold CV over train+dev documents (or --protocol dev: train -> dev) with out-of-fold sentence
           posteriors; picks the epoch count and the decoder (Viterbi / posterior argmax, per-label offsets);
           with --final retrains on every labelled document and writes submission/ for the test set
  predict  a saved model on any split
  check    validate a prediction.json against the test (or dev) documents

Models: any Phase-1 checkpoint (--checkpoint) or a Phase-2 NER model (--init-phase2 model.pt, its adapters
merged into the weights), adapted with LoRMA / LoRA / full / ... (phase2.peft). Knowledge distillation:
--teacher <phase-3 run dir> distils the teacher's sentence posteriors into a smaller student; in CV every
student fold learns from the teacher model of the SAME fold, so the student's CV stays honest.
"""
import argparse
import json
import math
import random
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase2.run import autocast, device_and_amp

torch.backends.cuda.matmul.allow_tf32 = True          # fp32 parts (LSTM, CRF, heads) on tensor cores
torch.backends.cudnn.allow_tf32 = True

from .data import (DOMAINS, FastScorer, label_counts, load_split, load_vocab, pooled_score, select_similar,
                   unlabeled_documents, validate_prediction, write_prediction, write_truth)
from .decode import apply_decoding, choose_decoding, finalize_decoding, jsonable
from .model import build_tagger, load_tagger, save_tagger


# --------------------------------------------------------------------------- tokens
def tokenize_docs(tokenizer, docs, max_sent_tokens):
    """doc -> {"ids": [token ids per sentence], ...}; long sentences keep their head and tail."""
    fallback = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<unk>") or 1
    tail = max(1, max_sent_tokens // 4)
    out = OrderedDict()
    for doc_id, doc in docs.items():
        ids = []
        for sentence in doc["sentences"]:
            t = tokenizer.encode(" " + " ".join(sentence.split())).ids or [fallback]
            if len(t) > max_sent_tokens:
                t = t[:max_sent_tokens - tail] + t[-tail:]
            ids.append(t)
        out[doc_id] = {"ids": ids, "labels": doc["labels"], "domain": doc["domain"]}
    return out


def pack(sent_ids, window, packing):
    """Concatenate consecutive sentences into windows of <= window tokens.
    Returns (windows, spans) with spans[i] = (window index, start, end) of sentence i."""
    windows, spans = [], []
    for ids in sent_ids:
        if packing == "isolated" or not windows or len(windows[-1]) + len(ids) > window:
            windows.append([])
        start = len(windows[-1])
        windows[-1].extend(ids)
        spans.append((len(windows) - 1, start, start + len(ids)))
    return windows, spans


def encode_sentences(tagger, sent_ids, args, device, dtype, pad_id):
    """[n, D] sentence vectors (gradients flow when the model trains). With --two-pass, a second pass reads
    the sentences in reverse order, so each sentence is also encoded after the ones that FOLLOW it (a causal
    LM otherwise never sees, e.g., the court's "distinguishable on facts" when it encodes the citation)."""
    forward = _encode_once(tagger, sent_ids, args, device, dtype, pad_id)
    if not getattr(tagger, "two_pass", False):
        return forward
    backward = _encode_once(tagger, sent_ids[::-1], args, device, dtype, pad_id).flip(0)
    return torch.cat([forward, backward], -1)


def _encode_once(tagger, sent_ids, args, device, dtype, pad_id):
    windows, spans = pack(sent_ids, args.window, args.packing)
    order = sorted(range(len(windows)), key=lambda i: len(windows[i]))
    hidden = [None] * len(windows)
    group = []

    def flush():
        length = max(len(windows[i]) for i in group)
        batch = torch.full((len(group), length), pad_id, dtype=torch.long)
        for row, i in enumerate(group):
            batch[row, :len(windows[i])] = torch.tensor(windows[i])
        with autocast(device, dtype):
            h = tagger.encoder(batch.to(device))
        for row, i in enumerate(group):
            hidden[i] = h[row, :len(windows[i])]

    for i in order:
        if group and (len(group) + 1) * max(len(windows[i]), max(len(windows[j]) for j in group)) > args.max_batch_tokens:
            flush()
            group = []
        group.append(i)
    if group:
        flush()
    return torch.stack([tagger.pool(hidden[w][s:e]) for w, s, e in spans])


@torch.no_grad()
def encode_many(tagger, seqs, args, device, dtype, pad_id, batch_tokens=None):
    """No-gradient sentence vectors for many documents at once: windows of DIFFERENT documents share GPU
    batches (sorted by length, so little padding), and each window is pooled as soon as it is encoded, so
    token states never pile up. seqs: list of documents (lists of sentence token ids) -> list of [n_i, D].
    Same windows and pooling as encode_sentences (identical vectors for causal models)."""
    budget = batch_tokens or getattr(args, "infer_batch_tokens", 0) or args.max_batch_tokens
    passes = (False, True) if getattr(tagger, "two_pass", False) else (False,)
    jobs, items = [], []                  # jobs: (doc, reversed, spans); items: (len, job, window tokens)
    for di, ids in enumerate(seqs):
        for rev in passes:
            windows, spans = pack(ids[::-1] if rev else ids, args.window, args.packing)
            j = len(jobs)
            jobs.append([di, rev, spans, [None] * len(spans)])
            by_window = {}
            for si, (w, a, b) in enumerate(spans):
                by_window.setdefault(w, []).append((si, a, b))
            for w, toks in enumerate(windows):
                items.append((len(toks), j, toks, by_window.get(w, [])))
    items.sort(key=lambda t: t[0])
    group = []

    def flush():
        length = max(t[0] for t in group)
        batch = torch.full((len(group), length), pad_id, dtype=torch.long)
        for row, t in enumerate(group):
            batch[row, :t[0]] = torch.tensor(t[2])
        with autocast(device, dtype):
            h = tagger.encoder(batch.to(device, non_blocking=True))
        for row, (_, j, _, sents) in enumerate(group):
            for si, a, b in sents:
                jobs[j][3][si] = tagger.pool(h[row, a:b])

    for t in items:
        if group and (len(group) + 1) * max(t[0], group[-1][0]) > budget:
            flush()
            group = []
        group.append(t)
    if group:
        flush()
    out = [[None, None] for _ in seqs]
    for di, rev, _, vecs in jobs:
        v = torch.stack(vecs)
        out[di][1 if rev else 0] = v.flip(0) if rev else v
    return [torch.cat(o, -1) if o[1] is not None else o[0] for o in out]


def positions(start, end, n, device):
    return ((torch.arange(start, end, dtype=torch.float32) + 0.5) / n).to(device)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_docs(tagger, recs, args, device, dtype, pad_id, temperature=1.0):
    """{doc: {"emissions": [n,K], "logp": [n,K], "domain"}} with document-level inference: sentence vectors
    for the whole document, the context model over overlapping segments (as in training), edge-weighted."""
    tagger.eval()
    out = OrderedDict()
    seg = args.segment_sentences
    order = list(recs)
    chunk, vectors_of = [], {}
    for start in range(0, len(order), 32):                 # 32 documents share encoder batches
        chunk = order[start:start + 32]
        vectors_of = dict(zip(chunk, encode_many(tagger, [recs[d]["ids"] for d in chunk], args, device, dtype, pad_id)))
        for doc_id in chunk:
            out[doc_id] = _predict_one(tagger, recs[doc_id], vectors_of.pop(doc_id), args, seg, device, temperature)
    return out


def _predict_one(tagger, rec, vectors, args, seg, device, temperature):
    n = len(rec["ids"])
    if n <= seg or args.infer_context == "full" or getattr(args, "doc_context", False):
        em = tagger.contextualize(vectors, positions(0, n, n, device), rec["domain"]).float()
    else:
        stride = max(1, seg // 2)
        starts = list(range(0, n - seg + 1, stride))
        if starts[-1] + seg < n:
            starts.append(n - seg)
        total = torch.zeros(n, len(tagger.labels), device=device)
        weight = torch.zeros(n, 1, device=device)
        ramp = max(1, seg // 4)
        for s in starts:
            e = s + seg
            em = tagger.contextualize(vectors[s:e], positions(s, e, n, device), rec["domain"]).float()
            idx = torch.arange(s, e, device=device, dtype=torch.float32)
            left = idx - s + 1 if s > 0 else torch.full_like(idx, 1e9)
            right = e - idx if e < n else torch.full_like(idx, 1e9)
            w = (torch.minimum(left, right) / ramp).clamp(0.05, 1.0).unsqueeze(1)
            total[s:e] += w * em
            weight[s:e] += w
        em = total / weight
    logp = tagger.log_probs(em, temperature)
    return {"emissions": em.cpu().numpy(), "logp": logp.float().cpu().numpy(), "domain": rec["domain"]}


def crf_params(tagger):
    return tagger.crf.params() if tagger.use_crf else None


def quick_decode(tagger, pred, labels):
    """Plain decoding for epoch selection: Viterbi with a CRF, argmax otherwise."""
    from .model import viterbi_np
    params = crf_params(tagger)
    out = {}
    for doc, r in pred.items():
        idx = viterbi_np(r["emissions"], params) if params else r["logp"].argmax(1).tolist()
        out[doc] = [labels[i] for i in idx]
    return out


# --------------------------------------------------------------------------- training
LR_DEFAULT = {"lora": 3e-4, "rslora": 3e-4, "qlora": 3e-4, "lorma": 3e-4, "adalora": 1e-3, "ia3": 3e-3,
              "prefix": 1e-3, "prompt": 1e-2, "adapter": 1e-3}       # used when --lr-peft is not given


def class_weights(train_recs, labels, kind, device):
    if kind == "none":
        return None
    counts = Counter(l for r in train_recs.values() for l in (r["labels"] or ()))
    total = sum(counts.values())
    w = []
    for label in labels:
        c = counts.get(label, 0)
        if c == 0:
            w.append(0.0)
            continue
        ratio = total / (len(labels) * c)
        w.append(min(10.0, math.sqrt(ratio) if kind == "sqrt" else ratio))
    w = torch.tensor(w, dtype=torch.float32)
    return (w / w[w > 0].mean()).to(device)


def make_segments(recs, seg, rng, rare_idx, rare_repeat, index, label_weights=None, domain_weights=None):
    """Every sentence once per epoch, segment borders shifted randomly each epoch; segments holding a
    rare label repeated (--rare-repeat, or per label with --repeat-labels), domains weighted (--domain-weights)."""
    from .mixing import repeats
    items = []
    for doc_id, rec in recs.items():
        n = len(rec["ids"])
        offset = rng.randrange(seg) if n > seg else 0
        bounds = ([0] if offset else []) + list(range(offset, n, seg)) + [n]
        for s, e in zip(bounds, bounds[1:]):
            if e <= s:
                continue
            times = 1
            if rare_repeat > 1 and rec["labels"] and any(index[l] in rare_idx for l in rec["labels"][s:e]):
                times = rare_repeat
            if label_weights and rec["labels"]:
                times = max([times] + [label_weights.get(l, 1) for l in set(rec["labels"][s:e])])
            if domain_weights:
                times = times * domain_weights.get(rec["domain"], 1.0)
            items += [(doc_id, s, e)] * max(1 if times >= 1 else 0, repeats(times, rng))
    rng.shuffle(items)
    return items


def segment_loss(tagger, em, tags, args, weights):
    if args.loss == "crf":
        loss = tagger.crf.nll(em, tags)
        if args.aux_ce:
            loss = loss + args.aux_ce * F.cross_entropy(em.float(), tags, weight=weights)
        return loss
    logp = F.log_softmax(em.float(), -1)
    nll = F.nll_loss(logp, tags, weight=weights, reduction="none")
    if args.loss == "focal":
        pt = logp.gather(1, tags[:, None]).squeeze(1).exp()
        nll = (1 - pt) ** args.focal_gamma * nll
    denom = weights[tags].sum() if weights is not None else torch.tensor(float(len(tags)), device=em.device)
    return nll.sum() / denom.clamp_min(1e-6)


def drop_tokens(sent_ids, p, rng):
    """Noisy-student input noise: drop each token with probability p (every sentence keeps >= 1 token)."""
    if p <= 0:
        return sent_ids
    return [[t for t in ids if rng.random() >= p] or ids[:1] for ids in sent_ids]


@torch.no_grad()
def cache_vectors(tagger, recs, doc_ids, args, device, dtype, pad_id):
    """Sentence vectors of whole documents without gradients (--doc-context): the document context model sees
    every sentence of the document, while encoder gradients flow only through the training segment."""
    was = tagger.training
    tagger.eval()
    doc_ids = [d for d in doc_ids if len(recs[d]["ids"]) > args.segment_sentences]   # shorter ones are all live
    out = {}
    for start in range(0, len(doc_ids), 32):
        chunk = doc_ids[start:start + 32]
        for d, v in zip(chunk, encode_many(tagger, [recs[d]["ids"] for d in chunk], args, device, dtype, pad_id)):
            out[d] = v.to(torch.float16)
    tagger.train(was)
    return out


def silver_budget(args, epoch, epochs):
    """Silver (teacher-labelled) sentences this epoch: constant, or decaying to 0 so the last epochs are gold-only."""
    if getattr(args, "kd_silver_schedule", "constant") == "decay":
        return int(args.kd_unlabeled_sentences * max(0.0, 1 - (epoch - 1) / max(1, epochs - 1)))
    return args.kd_unlabeled_sentences


def train(args, labels, train_recs, val_recs=None, val_gold=None, epochs=None, teacher_logp=None,
          unlabeled_recs=None, log=None):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device, dtype = device_and_amp()
    tagger, tokenizer, info = build_tagger(args, labels)
    tagger.to(device)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    index = {l: i for i, l in enumerate(labels)}
    weights = class_weights(train_recs, labels, args.class_weights, device)
    counts = label_counts({d: {"labels": r["labels"]} for d, r in train_recs.items()})
    present = [l for l in labels if counts.get(l, 0) > 0]
    rare = sorted(present, key=lambda l: counts[l])[:args.rare_labels]
    rare_idx = {index[l] for l in rare}
    from .mixing import parse_weights, splice_documents
    label_weights = parse_weights(args.repeat_labels)
    domain_weights = parse_weights(args.domain_weights)
    splice_rare = [l for l in (args.splice_labels or "").split(",") if l]
    encoder_params = [p for p in tagger.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in tagger.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    lr_encoder = args.lr_full if args.method in {"full", "partial", "normfit"} else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
    groups = [g for g in ({"params": encoder_params, "lr": lr_encoder}, {"params": head_params, "lr": args.lr_head})
              if g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    epochs = epochs or args.epochs
    n_unlab = sum(len(r["ids"]) for r in (unlabeled_recs or {}).values())
    boost = (1 + 0.3 * (args.rare_repeat > 1 or bool(label_weights))) * (1 + args.splice)
    steps_per_epoch = math.ceil((sum(len(r["ids"]) for r in train_recs.values()) * boost
                                 + min(n_unlab, args.kd_unlabeled_sentences)) / args.segment_sentences
                                / args.segments_per_step) + 1
    total = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(args.warmup * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.02, (total - s) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    factory = getattr(tagger.encoder, "allocator_factory", None)
    allocator = factory(tagger.encoder, args, total) if factory else None      # AdaLoRA rank budget
    global_step = 0
    print(json.dumps({"train_docs": len(train_recs), "train_sentences": sum(len(r["ids"]) for r in train_recs.values()),
                      "unlabeled_sentences": n_unlab, "rare_labels": rare, **{k: v for k, v in info.items()}},
                     default=str), flush=True)
    best = {"score": -1.0, "epoch": 0, "state": None}
    stale, history = 0, []
    T = args.kd_temperature
    doc_context = getattr(args, "doc_context", False)
    token_dropout = getattr(args, "token_dropout", 0.0)
    use_aux = bool(getattr(args, "aux_shift", 0) or getattr(args, "aux_coarse", 0))
    for epoch in range(1, epochs + 1):
        tagger.train()
        started = time.time()
        epoch_recs = train_recs
        if args.splice > 0:
            synthetic = splice_documents(train_recs, round(args.splice * len(train_recs)), rng, splice_rare)
            epoch_recs = {**train_recs, **synthetic}
        jobs = [("gold", it) for it in make_segments(epoch_recs, args.segment_sentences, rng, rare_idx,
                                                     args.rare_repeat, index, label_weights, domain_weights)]
        if unlabeled_recs:
            u = [("unlabeled", it) for it in make_segments(unlabeled_recs, args.segment_sentences, rng, set(), 1, index)]
            budget, kept = silver_budget(args, epoch, epochs), []
            for job in u:
                if budget <= 0:
                    break
                kept.append(job)
                budget -= job[1][2] - job[1][1]
            jobs += kept
            rng.shuffle(jobs)
        cache = {}
        if doc_context:
            for kind, recs_k in (("gold", epoch_recs), ("unlabeled", unlabeled_recs)):
                needed = sorted({d for k2, (d, _, _) in jobs if k2 == kind})
                if needed:
                    cache.update({(kind, d): v for d, v in
                                  cache_vectors(tagger, recs_k, needed, args, device, dtype, pad_id).items()})
        losses = []
        for b in range(0, len(jobs), args.segments_per_step):
            batch = jobs[b:b + args.segments_per_step]
            # gradient accumulation per segment: backward right after each segment, so only one segment's
            # activations live on the GPU (identical gradients to summing the losses; ~segments_per_step x less memory)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for kind, (doc_id, s, e) in batch:
                rec = (epoch_recs if kind == "gold" else unlabeled_recs)[doc_id]
                n = len(rec["ids"])
                live = encode_sentences(tagger, drop_tokens(rec["ids"][s:e], token_dropout, rng), args, device, dtype, pad_id)
                prev_h = None
                if doc_context and (kind, doc_id) in cache:
                    c = cache[(kind, doc_id)].to(live.dtype)
                    em_all, h_all = tagger.contextualize(torch.cat([c[:s], live, c[e:]]), positions(0, n, n, device),
                                                         rec["domain"], return_hidden=True)
                    em, h = em_all[s:e].float(), h_all[s:e]
                    prev_h = h_all[s - 1] if s > 0 else None
                else:
                    em, h = tagger.contextualize(live, positions(s, e, n, device), rec["domain"], return_hidden=True)
                    em = em.float()
                part = torch.zeros((), device=device)
                aux = torch.zeros((), device=device)
                if kind == "gold":
                    tags = torch.tensor([index[l] for l in rec["labels"][s:e]], device=device)
                    part = segment_loss(tagger, em, tags, args, weights)
                    if use_aux:
                        prev_tag = torch.tensor(index[rec["labels"][s - 1]], device=device) if prev_h is not None else None
                        aux = tagger.aux_loss(h, tags, prev_h, prev_tag, args.aux_shift, args.aux_coarse)
                if teacher_logp is not None and doc_id in teacher_logp:
                    target = torch.as_tensor(teacher_logp[doc_id][s:e], device=device).float()
                    student = tagger.log_probs(em, T)
                    kl = (target.exp() * (target - student)).sum(-1)
                    if getattr(args, "kd_confidence", False):       # trust confident teacher sentences more
                        w = target.exp().max(-1).values
                        kd = (w * kl).sum() / w.sum().clamp_min(1e-6) * T * T
                    else:
                        kd = kl.mean() * T * T
                    alpha = args.kd_alpha if kind == "gold" else 0.0
                    part = alpha * part + (1 - alpha) * kd
                seg_loss = (part + aux) / len(batch)
                if seg_loss.requires_grad:
                    scaler.scale(seg_loss).backward()
                step_loss += float(seg_loss.detach())
                del seg_loss, part, aux, em, h, live
            if allocator is not None and args.adalora_orth:
                orth = args.adalora_orth * allocator.orth_penalty()
                scaler.scale(orth).backward()
                step_loss += float(orth.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in tagger.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if allocator is not None:
                allocator.step(global_step)
            losses.append(step_loss)
        cache = {}                                           # free the document cache before validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "seconds": round(time.time() - started, 1),
               **({"adalora_rank": sum(allocator.ranks())} if allocator is not None else {}),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device.type == "cuda" else None}
        if val_recs is not None and (epoch >= args.eval_from or epoch == epochs):
            pred = predict_docs(tagger, val_recs, args, device, dtype, pad_id)
            score = pooled_score(val_gold, quick_decode(tagger, pred, labels), labels)["macro_f1"]
            row["val_macro_f1"] = round(score, 4)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()}}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if log:
            with open(log, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        if val_recs is not None and stale >= args.patience:
            break
    if best["state"] is not None:
        tagger.load_state_dict(best["state"])
    return tagger, tokenizer, info, history, best


# --------------------------------------------------------------------------- folds
def make_folds(docs, k, seed):
    """Documents into k folds, per domain, balancing sentence counts (largest documents first)."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    sizes = [0] * k
    for domain in DOMAINS:
        ids = [d for d, doc in docs.items() if doc["domain"] == domain]
        rng.shuffle(ids)
        ids.sort(key=lambda d: -len(docs[d]["sentences"]))
        for d in ids:
            j = min(range(k), key=lambda i: (sizes[i], len(folds[i]), rng.random()))
            folds[j].append(d)
            sizes[j] += len(docs[d]["sentences"])
    return [sorted(f) for f in folds]


def fold_plan(args, train_docs, dev_docs):
    if args.protocol == "dev":
        return [{"train": list(train_docs), "val": list(dev_docs)}]
    pool = OrderedDict(list(train_docs.items()) + list(dev_docs.items()))
    folds = make_folds(pool, args.folds, args.fold_seed)
    return [{"train": [d for j, f in enumerate(folds) if j != i for d in f], "val": f} for i, f in enumerate(folds)]


# --------------------------------------------------------------------------- KD helpers
def _doc_key(doc_id, doc):
    import hashlib
    return f"{doc_id}:{hashlib.md5(chr(10).join(doc['sentences']).encode()).hexdigest()[:12]}"


def teacher_posteriors(teacher_path, docs, args, device, dtype):
    """Teacher sentence log-posteriors at the KD temperature, cached on disk next to the teacher model
    (kd_cache_T<T>.pt, keyed by document id + text hash, invalidated when the model file changes). Runs
    that share teachers (R1A, R1B, the ablations, round 2) compute each teacher's posteriors once; a file
    lock lets two runs on two GPUs share the cache safely."""
    if getattr(args, "no_kd_cache", False):
        return _teacher_posteriors(teacher_path, docs, args, device, dtype)
    import fcntl
    path = Path(teacher_path)
    cache_path = path.with_name(f"kd_cache_T{args.kd_temperature:g}.pt")
    stamp = (path.stat().st_size, int(path.stat().st_mtime))
    keys = {d: _doc_key(d, doc) for d, doc in docs.items()}
    with open(str(cache_path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = {"stamp": stamp, "logp": {}, "parameters": None}
        if cache_path.exists():
            try:
                old = torch.load(cache_path, map_location="cpu", weights_only=False)
                if tuple(old.get("stamp", ())) == stamp:
                    cache = old
            except Exception:
                pass
        missing = OrderedDict((d, docs[d]) for d in docs if keys[d] not in cache["logp"])
        if missing:
            logp, n = _teacher_posteriors(teacher_path, missing, args, device, dtype)
            cache["logp"].update({keys[d]: v.astype(np.float16) for d, v in logp.items()})
            cache["parameters"] = n
            tmp = cache_path.with_suffix(".tmp")
            torch.save(cache, tmp)
            tmp.replace(cache_path)
        print(f"KD cache {cache_path.name}: {len(docs) - len(missing)} documents reused, {len(missing)} computed", flush=True)
    return {d: cache["logp"][keys[d]].astype(np.float32) for d in docs}, cache["parameters"] or 0


def _teacher_posteriors(teacher_path, docs, args, device, dtype):
    teacher, tokenizer, payload = load_tagger(teacher_path, device)
    targs = argparse.Namespace(**payload["args"])
    targs.infer_batch_tokens = getattr(args, "infer_batch_tokens", 0)       # big no-grad batches for old runs too
    recs = tokenize_docs(tokenizer, docs, targs.max_sent_tokens)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    pred = predict_docs(teacher, recs, targs, device, dtype, pad_id, temperature=args.kd_temperature)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {d: r["logp"] for d, r in pred.items()}, n_teacher


def teacher_list(args):
    t = args.teacher
    return [] if not t else ([t] if isinstance(t, str) else list(t))


def ensemble_teacher(paths, docs, args, device, dtype):
    """Several teachers (e.g. the best runs of different backbones): their tempered sentence posteriors are
    averaged, so the student distils the ensemble instead of one run."""
    logps, n_total = [], 0
    for path in paths:
        lp, n = teacher_posteriors(path, docs, args, device, dtype)
        logps.append(lp)
        n_total += n
    if len(logps) == 1:
        return logps[0], n_total
    return {d: np.log(np.clip(np.mean([np.exp(lp[d]) for lp in logps], axis=0), 1e-9, None)) for d in logps[0]}, n_total


def check_teacher_folds(teacher_dir, plan):
    path = Path(teacher_dir) / "folds.json"
    if not path.exists():
        raise SystemExit(f"{teacher_dir}: no folds.json - is it a finished phase3 cv run?")
    theirs = json.loads(path.read_text())
    if [sorted(f["val"]) for f in theirs] != [sorted(f["val"]) for f in plan]:
        raise SystemExit("teacher and student folds differ: use the same --protocol/--folds/--fold-seed")


# --------------------------------------------------------------------------- commands
def records_for_decode(pred, crf):
    return {d: {"emissions": r["emissions"], "logp": r["logp"], "domain": r["domain"], "crf": crf} for d, r in pred.items()}


def cmd_cv(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.final_only:                  # keep the CV run's own arguments
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    train_docs = load_split(args.data, "train", labels)
    dev_docs = load_split(args.data, "dev", labels)
    test_docs = load_split(args.data, "test") if args.final else None
    all_docs = OrderedDict(list(train_docs.items()) + list(dev_docs.items()))
    if set(train_docs) & set(dev_docs):
        raise SystemExit("train and dev share document ids")
    gold = {d: doc["labels"] for d, doc in all_docs.items()}
    plan = fold_plan(args, train_docs, dev_docs)
    (out / "folds.json").write_text(json.dumps(plan, indent=1))
    device, dtype = device_and_amp()
    unlabeled = {}
    if args.teacher and args.kd_unlabeled:
        every = [" ".join(d["sentences"]) for docs in (all_docs, test_docs or load_split(args.data, "test"))
                 for d in docs.values()]
        pool = args.kd_unlabeled_docs * (args.kd_similar_pool if getattr(args, "kd_select_similar", False) else 1)
        unlabeled, dropped = unlabeled_documents(args.kd_unlabeled, pool, args.seed, every,
                                                 take_prob=args.kd_take_prob, select=args.kd_select_domain)
        print(f"KD unlabeled: {len(unlabeled)} judgments, {sum(len(d['sentences']) for d in unlabeled.values())} "
              f"sentences ({dropped} dropped: they overlap a Phase-3 train/dev/test document)"
              f"{'; candidates, ranked per fold by similarity to its training documents' if pool > args.kd_unlabeled_docs else ''}",
              flush=True)
    for t in teacher_list(args):
        check_teacher_folds(t, plan)

    oof, crfs, fold_rows = OrderedDict(), {}, []
    summary_path = out / "cv_summary.json"
    if args.final_only:
        summary = json.loads(summary_path.read_text())
    else:
        for i, fold in enumerate(plan):
            started = time.time()
            tr = OrderedDict((d, all_docs[d]) for d in fold["train"])
            va = OrderedDict((d, all_docs[d]) for d in fold["val"])
            teacher_logp, n_teacher = None, None
            fold_unlabeled = (select_similar(unlabeled, tr, args.kd_unlabeled_docs)
                              if unlabeled and getattr(args, "kd_select_similar", False) else unlabeled)
            if teacher_list(args):
                tpaths = [Path(t) / "folds" / f"fold{i}" / "model.pt" for t in teacher_list(args)]
                teacher_logp, n_teacher = ensemble_teacher(tpaths, OrderedDict(list(tr.items()) + list(fold_unlabeled.items())),
                                                           args, device, dtype)
                print(f"KD: {len(tpaths)} teacher(s) {[str(t) for t in tpaths]} ({n_teacher:,} parameters)", flush=True)
            tok = _tokenizer(args)
            tr_recs = tokenize_docs(tok, tr, args.max_sent_tokens)
            va_recs = tokenize_docs(tok, va, args.max_sent_tokens)
            un_recs = tokenize_docs(tok, fold_unlabeled, args.max_sent_tokens) if fold_unlabeled else None
            print(f"== fold {i + 1}/{len(plan)}: train {len(tr)} docs, validate {len(va)} docs", flush=True)
            tagger, tokenizer, info, history, best = train(
                args, labels, tr_recs, va_recs, {d: gold[d] for d in va}, teacher_logp=teacher_logp,
                unlabeled_recs=un_recs, log=out / f"fold{i}_history.jsonl")
            save_tagger(out / "folds" / f"fold{i}" / "model.pt", tagger, args, labels, {"best_epoch": best["epoch"]})
            pad_id = tokenizer.token_to_id("<pad>") or 0
            pred = predict_docs(tagger, va_recs, args, device, dtype, pad_id)
            crfs[i] = crf_params(tagger)
            for d, r in pred.items():
                oof[d] = {**r, "fold": i, "crf": crfs[i]}
            fold_rows.append({"fold": i, "val_docs": len(va), "best_epoch": best["epoch"],
                              "val_macro_f1": round(best["score"], 4), "minutes": round((time.time() - started) / 60, 1),
                              "history": history, "teacher_parameters": n_teacher,
                              "parameters": info["parameters"], "encoder_parameters": info["encoder_parameters"],
                              "peft": info.get("peft")})
            print(f"fold {i}: best epoch {best['epoch']}  val pooled macro-F1 {best['score']:.4f}", flush=True)
            del tagger
            if device.type == "cuda":
                torch.cuda.empty_cache()
        torch.save({"oof": oof, "labels": labels}, out / "oof.pt")
        summary = summarize(args, labels, oof, gold, dev_docs, fold_rows, out, all_docs)
    if args.final:
        if unlabeled and getattr(args, "kd_select_similar", False):
            unlabeled = select_similar(unlabeled, all_docs, args.kd_unlabeled_docs)
        final_fit(args, labels, all_docs, test_docs, summary, unlabeled, out, device, dtype)


def _tokenizer(args):
    from tokenizers import Tokenizer
    if getattr(args, "init_phase2", None):
        payload = torch.load(args.init_phase2, map_location="cpu", weights_only=False)
        checkpoint = payload["args"]["checkpoint"]
    else:
        checkpoint = args.checkpoint
    return Tokenizer.from_file(str(Path(checkpoint).parent / "tokenizer.json"))


def summarize(args, labels, oof, gold, dev_docs, fold_rows, out, all_docs=None):
    docs = [d for d in oof]
    from .model import viterbi_np
    raw = {}
    for d in docs:
        r = oof[d]
        idx = viterbi_np(r["emissions"], r["crf"]) if r["crf"] is not None else r["logp"].argmax(1).tolist()
        raw[d] = [labels[i] for i in idx]
    raw_score = pooled_score(gold, raw, labels, docs)
    print(f"OOF pooled macro-F1 (plain decoding): {raw_score['macro_f1']:.4f}", flush=True)
    config, results = finalize_decoding(oof, gold, labels, all_docs, rules=getattr(args, "rules", "auto"))
    decoded = apply_decoding(oof, config, labels, all_docs)
    final_score = pooled_score(gold, decoded, labels, docs)
    dev_ids = [d for d in docs if d in dev_docs]
    dev_score = pooled_score(gold, decoded, labels, dev_ids) if dev_ids else None
    per_domain = {dom: round(pooled_score(gold, decoded, labels, [d for d in docs if oof[d]["domain"] == dom])["macro_f1"], 4)
                  for dom in DOMAINS}
    print(f"chosen decoding: {config['mode']} + {config['offsets']} offsets + rules {config['rules']}  "
          f"(honest {config['honest']:.4f})", flush=True)
    print(f"OOF pooled macro-F1 with it: {final_score['macro_f1']:.4f}   dev docs only: "
          f"{dev_score['macro_f1'] if dev_score else float('nan'):.4f}   per domain {per_domain}", flush=True)
    print(f"{'label':24s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'pred':>6s}")
    for label, v in final_score["per_label"].items():
        print(f"{label:24s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['support']:6d} {v['pred']:6d}")
    epochs = [r["best_epoch"] for r in fold_rows]
    summary = {"protocol": args.protocol, "method": args.method,
               "backbone": args.init_phase2 or args.checkpoint, "mode": args.mode, "keep_layers": args.keep_layers,
               "teacher": args.teacher, "parameters": fold_rows[0]["parameters"] if fold_rows else None,
               "mixing": {"repeat_labels": getattr(args, "repeat_labels", ""), "domain_weights": getattr(args, "domain_weights", ""),
                          "splice": getattr(args, "splice", 0.0), "kd_select_domain": getattr(args, "kd_select_domain", False)},
               "encoder_parameters": fold_rows[0].get("encoder_parameters") if fold_rows else None,
               "peft": fold_rows[0].get("peft") if fold_rows else None,
               "minutes": round(sum(r["minutes"] for r in fold_rows), 1),
               "folds": fold_rows, "best_epochs": epochs,
               "oof_macro_f1_plain": round(raw_score["macro_f1"], 5),
               "oof_macro_f1": round(final_score["macro_f1"], 5),
               "oof_macro_f1_honest": round(float(config["honest"]), 5),
               "dev_docs_macro_f1": round(dev_score["macro_f1"], 5) if dev_score else None,
               "oof_by_domain": per_domain,
               "decoding": jsonable(config), "decoding_candidates": [jsonable(r) for r in results],
               "oof_per_label": {l: {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                                 for l, s in final_score["per_label"].items()}}
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def final_fit(args, labels, all_docs, test_docs, summary, unlabeled, out, device, dtype):
    epochs = max(1, round(float(np.mean(summary["best_epochs"])) * args.final_epoch_scale))
    print(f"== final fit on all {len(all_docs)} labelled documents for {epochs} epochs", flush=True)
    teacher_logp = None
    if teacher_list(args):
        teacher_logp, _ = ensemble_teacher([Path(t) / "final" / "model.pt" for t in teacher_list(args)],
                                           OrderedDict(list(all_docs.items()) + list(unlabeled.items())),
                                           args, device, dtype)
    tok = _tokenizer(args)
    recs = tokenize_docs(tok, all_docs, args.max_sent_tokens)
    un_recs = tokenize_docs(tok, unlabeled, args.max_sent_tokens) if unlabeled else None
    tagger, tokenizer, info, history, _ = train(args, labels, recs, epochs=epochs, teacher_logp=teacher_logp,
                                                unlabeled_recs=un_recs, log=out / "final_history.jsonl")
    save_tagger(out / "final" / "model.pt", tagger, args, labels, {"epochs": epochs, "decoding": summary["decoding"]})
    pad_id = tokenizer.token_to_id("<pad>") or 0
    test_recs = tokenize_docs(tokenizer, test_docs, args.max_sent_tokens)
    pred = predict_docs(tagger, test_recs, args, device, dtype, pad_id)
    records = records_for_decode(pred, crf_params(tagger))
    torch.save({"test": records, "labels": labels}, out / "test.pt")
    decoded = apply_decoding(records, summary["decoding"], labels, test_docs)
    info_out = write_prediction(decoded, test_docs, labels, out / "submission")
    print(f"submission: {out / 'submission' / 'submission.zip'}  {json.dumps(info_out['by_label'])}", flush=True)


def cmd_predict(args):
    labels_path = Path(args.data) / "Train" / "label_vocab.json"
    labels = load_vocab(labels_path)
    docs = load_split(args.data, args.split)
    device, dtype = device_and_amp()
    tagger, tokenizer, payload = load_tagger(args.model, device)
    margs = argparse.Namespace(**payload["args"])
    recs = tokenize_docs(tokenizer, docs, margs.max_sent_tokens)
    pred = predict_docs(tagger, recs, margs, device, dtype, tokenizer.token_to_id("<pad>") or 0)
    records = records_for_decode(pred, crf_params(tagger))
    config = payload.get("decoding") or {"mode": "viterbi" if tagger.use_crf else "marginal", "offsets": "none"}
    decoded = apply_decoding(records, config, labels, docs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({args.split: records, "labels": labels}, out / f"{args.split}.pt")
    print(json.dumps(write_prediction(decoded, docs, labels, out)))
    if docs and all(d["labels"] is not None for d in docs.values()):
        s = pooled_score({k: d["labels"] for k, d in docs.items()}, decoded, labels)
        print(f"pooled macro-F1 on {args.split}: {s['macro_f1']:.4f}")


def cmd_check(args):
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    docs = load_split(args.data, args.split)
    pred = json.loads(Path(args.prediction).read_text())
    validate_prediction(pred, docs, labels)
    counts = Counter(l for v in pred.values() for l in v)
    print(f"OK: {len(pred)} documents, {sum(len(v) for v in pred.values())} sentences, labels {dict(counts)}")
    if all(d["labels"] is not None for d in docs.values()):
        from .data import normalize_id
        p = {normalize_id(k): v for k, v in pred.items()}
        s = pooled_score({k: d["labels"] for k, d in docs.items()}, p, labels)
        print(f"pooled macro-F1: {s['macro_f1']:.4f}  P {s['macro_p']:.4f}  R {s['macro_r']:.4f}")


def cmd_truth(args):
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from .data import load_docs, split_paths
    merged = OrderedDict()
    for domain, path in split_paths(args.data, "dev").items():
        docs = load_docs(path, domain)
        write_truth(docs, out / f"{domain.lower()}_dev_truth.json")
        merged.update(docs)
    trivial = {d: ["Fact"] * len(doc["sentences"]) for d, doc in merged.items()}
    (out / "dev_all_fact_prediction.json").write_text(json.dumps(trivial))
    (out / "labels.json").write_text(json.dumps(labels))
    s = pooled_score({k: d["labels"] for k, d in merged.items()}, trivial, labels)
    print(f"wrote {out}/cl_dev_truth.json, it_dev_truth.json, dev_all_fact_prediction.json, labels.json")
    print(f"all-Fact dev pooled macro-F1 (our scorer): {s['macro_f1']:.6f}")
    print("cross-check with the official scorer:\n  python evaluate_phase3.py --truth-cl "
          f"{out}/cl_dev_truth.json --truth-it {out}/it_dev_truth.json --prediction "
          f"{out}/dev_all_fact_prediction.json --labels {Path(args.data) / 'Train' / 'label_vocab.json'}")


def cmd_stats(args):
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    report = {}
    for split in ("train", "dev", "test"):
        docs = load_split(args.data, split, labels if split != "test" else None)
        for domain in DOMAINS:
            sub = {k: d for k, d in docs.items() if d["domain"] == domain}
            lens = [len(s.split()) for d in sub.values() for s in d["sentences"]]
            row = {"documents": len(sub), "sentences": len(lens),
                   "sentences_per_doc_max": max((len(d["sentences"]) for d in sub.values()), default=0),
                   "words_per_sentence_median": float(np.median(lens)) if lens else 0, "words_max": max(lens, default=0)}
            if split != "test":
                c = label_counts(sub)
                row["labels"] = {l: c.get(l, 0) for l in labels}
            report[f"{domain}_{split}"] = row
    print(json.dumps(report, indent=1))
    absent = [l for l in labels if all(report[f"{d}_dev"]["labels"][l] == 0 for d in DOMAINS)]
    print(f"labels with no dev sentence (score 0 there whatever you predict): {absent}")


# --------------------------------------------------------------------------- arguments
def add_model_args(p):
    p.add_argument("--data", required=True, help="Phase_3_RR directory (Train/ and Test/)")
    p.add_argument("--checkpoint", help="Phase-1 best.pt (tokenizer.json alongside)")
    p.add_argument("--init-phase2", help="start from a Phase-2 NER model.pt instead (adapters merged)")
    p.add_argument("--project-root", default=".", help="directory containing the phase1 package")
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="lorma", choices=["lorma", "lora", "rslora", "qlora", "adalora", "prefix",
                                                        "prompt", "ia3", "adapter", "normfit", "partial", "full",
                                                        "frozen"])
    p.add_argument("--mode", default="causal", choices=["causal", "bidirectional", "encoder"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--keep-layers", type=int, help="truncate the Phase-1 model (KD students)")
    p.add_argument("--prompt-tokens", type=int, default=20, help="prompt tuning: virtual tokens")
    p.add_argument("--prefix-length", type=int, default=16, help="prefix tuning: key/value slots per layer")
    p.add_argument("--prefix-hidden", type=int, default=512)
    p.add_argument("--adalora-init-r", type=int, default=12)
    p.add_argument("--adalora-target-r", type=int, default=8, help="average rank per module after pruning")
    p.add_argument("--adalora-orth", type=float, default=0.1, help="orthogonality penalty weight")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"],
                   help="auto: FlashAttention-2 if flash-attn is installed, else SDPA with expanded KV (phase3/flash.py)")
    # sentence encoding
    p.add_argument("--packing", default="packed", choices=["packed", "isolated"],
                   help="packed: consecutive sentences share a window (left context for causal models)")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--max-sent-tokens", type=int, default=128)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--infer-batch-tokens", type=int, default=32768,
                   help="tokens per encoder batch without gradients (validation, cache, teachers): bigger = faster")
    p.add_argument("--pooling", default="meanlast", choices=["mean", "last", "max", "meanlast"])
    # document context
    p.add_argument("--context", default="bilstm", choices=["bilstm", "transformer", "lstm_tf", "none"],
                   help="lstm_tf: BiLSTM then self-attention over the whole document")
    p.add_argument("--tf-layers", type=int, default=2, help="lstm_tf: self-attention layers")
    p.add_argument("--doc-context", action="store_true",
                   help="context model over the whole document (other sentences from a per-epoch no-grad cache)")
    p.add_argument("--two-pass", action="store_true", help="also encode sentences in reverse order (right context)")
    p.add_argument("--aux-shift", type=float, default=0.0, help="weight of the label-shift prediction head")
    p.add_argument("--aux-coarse", type=float, default=0.0, help="weight of the coarse-role head")
    p.add_argument("--token-dropout", type=float, default=0.0, help="noisy student: drop input tokens in training")
    p.add_argument("--context-layers", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--position-buckets", type=int, default=20)
    p.add_argument("--no-position", action="store_true")
    p.add_argument("--domain-heads", action="store_true", help="separate CL / IT output layers")
    p.add_argument("--segment-sentences", type=int, default=48)
    p.add_argument("--segments-per-step", type=int, default=4)
    p.add_argument("--infer-context", default="segments", choices=["segments", "full"])
    # loss
    p.add_argument("--loss", default="crf", choices=["crf", "ce", "focal"])
    p.add_argument("--class-weights", default="none", choices=["none", "sqrt", "inv"])
    p.add_argument("--aux-ce", type=float, default=0.0, help="weighted CE added to the CRF loss")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--rare-labels", type=int, default=4)
    p.add_argument("--rare-repeat", type=int, default=1, help="repeat segments holding a rare label")
    # data mixing (phase3/mixing.py)
    p.add_argument("--repeat-labels", default="", help="e.g. Dissent=4,Issue=3,PrecedentNotReliedUpon=2")
    p.add_argument("--domain-weights", default="", help="e.g. CL=1,IT=1.3 (fractional segment repetition)")
    p.add_argument("--splice", type=float, default=0.0, help="synthetic block-spliced documents per training document per epoch")
    p.add_argument("--splice-labels", default="Dissent,Issue,PrecedentNotReliedUpon,RulingByPresentCourt,None",
                   help="rare roles inserted as blocks into spliced documents")
    # optimisation
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--eval-from", type=int, default=2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr-full", type=float, default=3e-5)
    p.add_argument("--lr-peft", type=float, default=None, help="default per method (LR_DEFAULT)")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rules", default="auto", help="auto (keep each rule that raises OOF) | none | dissent")
    # protocol
    p.add_argument("--protocol", default="cv", choices=["cv", "dev"],
                   help="cv: K folds over train+dev documents; dev: train -> official dev only")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=0, help="keep fixed across runs: ensembles and KD need equal folds")
    p.add_argument("--final", action="store_true", help="after CV, fit on all labelled documents and predict test")
    p.add_argument("--final-only", action="store_true", help="reuse cv_summary.json, only run the final fit")
    p.add_argument("--final-epoch-scale", type=float, default=1.1)
    # knowledge distillation
    p.add_argument("--teacher", nargs="+",
                   help="finished phase3 cv run directory/ies (folds/fold*/model.pt, final/model.pt); several = "
                        "their averaged posteriors are the target (ensemble teacher), all on the same folds")
    p.add_argument("--kd-select-similar", action="store_true",
                   help="draw --kd-similar-pool x more candidates, keep the --kd-unlabeled-docs most similar (TF-IDF) "
                        "to each fold's training documents")
    p.add_argument("--kd-similar-pool", type=int, default=4)
    p.add_argument("--no-kd-cache", action="store_true", help="recompute teacher posteriors (no kd_cache_T*.pt)")
    p.add_argument("--kd-confidence", action="store_true", help="weight KD per sentence by the teacher's confidence")
    p.add_argument("--kd-silver-schedule", default="constant", choices=["constant", "decay"],
                   help="decay: silver sentences per epoch fall linearly to 0 (gold-only at the end)")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--kd-unlabeled", nargs="+", help="official Phase-1 train json(s): teacher-labelled extra judgments")
    p.add_argument("--kd-select-domain", action="store_true",
                   help="keep only competition-law / tax-banking judgments (vocabulary), tagged with that domain")
    p.add_argument("--kd-take-prob", type=float, default=0.1, help="fraction of corpus records considered")
    p.add_argument("--kd-unlabeled-docs", type=int, default=200)
    p.add_argument("--kd-unlabeled-sentences", type=int, default=20000, help="per epoch")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    add_model_args(sub.add_parser("cv"))
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--data", required=True)
    t.add_argument("--output", default="phase3_truth")
    pr = sub.add_parser("predict")
    pr.add_argument("--data", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--split", default="test", choices=["train", "dev", "test"])
    pr.add_argument("--output", required=True)
    pr.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--prediction", required=True)
    c.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    if args.command in ("cv", "predict"):
        from .flash import enable
        print(enable(getattr(args, "attention", "auto")), flush=True)
    if args.command == "cv":
        if not args.checkpoint and not args.init_phase2:
            parser.error("--checkpoint or --init-phase2 is required")
        if args.final_only:
            args.final = True
        cmd_cv(args)
    else:
        {"stats": cmd_stats, "truth": cmd_truth, "predict": cmd_predict, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
