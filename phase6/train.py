"""Fine-tuning a Phase-1 model for abstractive summarization (any technique), knowledge distillation, decoding.

Strategies for long documents (--strategy):
  chunk    sentence-aligned chunks of --chunk tokens; each chunk is trained to produce the summary sentences aligned
           to it (data.chunk_targets); inference summarizes every chunk with a word budget and joins the pieces
  head     only the first chunk, trained to produce the whole summary (truncation baseline: what is lost?)
  extract  the extractive scorer (baseline.py, --extractor) keeps the most salient sentences (--extract-keep of the
           words), then chunk on the compressed document (importance-based context selection)
  fid      encoder-decoder only: all chunks encoded separately, one decoder pass writes the whole summary
--overlap N prepends the last N tokens of the previous chunk as context (overlapping windows without duplicate
targets).

Batches are built by token budget (--micro-tokens per forward, --batch-examples per optimizer step); the loss is
the mean over target tokens of the step. --src-lm w adds w x the language-model loss on the source tokens
(decoder-only; in-task domain adaptation). Epoch selection by held-out teacher-forced loss.

Knowledge distillation (--teacher RUN_DIR):
  --kd logit  word-level KD: KL(teacher || student) at temperature T on every target token, mixed with the gold
              cross-entropy by --kd-alpha (needs the same tokenizer: all Phase-1 runs share one)
  --kd seq    sequence-level KD (Kim & Rush 2016): the teacher summarizes every training chunk once (cached in the
              teacher's folder); each epoch an example's target is the teacher's text with probability --kd-seq-mix
"""
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase2.run import autocast, device_and_amp

from .data import LengthModel, chunk_targets, density_profile, load_splits, prepare_doc, write_submission
from .evaluate import evaluate
from .generate import Decoding, generate, summarize
from .model import build_summarizer, load_summarizer, same_tokenizer, save_model, trainable_state

LR_DEFAULT = {"lora": 3e-4, "rslora": 3e-4, "qlora": 3e-4, "lorma": 3e-4, "adalora": 1e-3, "ia3": 3e-3,
              "prefix": 1e-3, "prompt": 3e-3, "houlsby": 1e-3, "pfeiffer": 1e-3, "adapter": 1e-3}


def log(msg):
    print(msg if isinstance(msg, str) else json.dumps(msg, default=str), flush=True)


# --------------------------------------------------------------------------- plan
def resolve_plan(args, model):
    if args.strategy == "fid" and not model.seq2seq:
        raise SystemExit("--strategy fid needs the encoder-decoder checkpoint")
    base = "chunk" if args.strategy == "extract" else args.strategy
    ctx, sep, prompt = model.max_context, len(model.sep), model.n_prompt
    if args.max_target:
        tgt = args.max_target
    elif base == "chunk":                       # decoder-only: source and target share the context
        tgt = min(ctx - 1, 768) if model.seq2seq else min(512, ctx // 3)
    else:
        tgt = ctx - 1 if model.seq2seq else min(768, ctx // 2)
    tgt = min(tgt, ctx - 1)
    chunk = min(args.chunk, ctx - prompt - args.overlap) if model.seq2seq else \
        min(args.chunk, ctx - prompt - sep - tgt - 1 - args.overlap)
    if chunk < 64:
        raise SystemExit(f"no room for a source chunk: context {ctx}, target {tgt}, overlap {args.overlap}")
    return {"strategy": base, "extract": args.strategy == "extract", "chunk": chunk, "overlap": args.overlap,
            "max_target": tgt, "fid_chunks": args.fid_chunks, "budget": args.budget, "len_scale": args.len_scale,
            "min_ratio": args.min_ratio, "max_ratio": args.max_ratio, "dedup": args.dedup, "align": args.align}


def build_examples(tokenizer, docs, plan, keep=None):
    """Training/validation examples {doc, k, src, tgt}, the prepared documents and chunk targets, statistics."""
    exs, prepared, targets, st = [], {}, {}, Counter()
    t_max = plan["max_target"]
    for d, doc in docs.items():
        prep = prepare_doc(tokenizer, doc["document"], plan["chunk"], plan["overlap"], (keep or {}).get(d))
        prepared[d] = prep
        summary = " ".join(doc["summary"].split())
        st["summary_words"] += len(summary.split())
        if plan["strategy"] == "chunk":
            tg = chunk_targets(prep, summary, plan["align"])
            targets[d] = tg
            for k, (c, sents) in enumerate(zip(prep["chunks"], tg)):
                st["chunks"] += 1
                if not sents:
                    st["chunks_without_target"] += 1
                    continue
                ids = tokenizer.encode(" " + " ".join(sents)).ids
                st["summary_tokens"] += len(ids)
                st["targets_truncated"] += len(ids) > t_max
                exs.append({"doc": d, "k": k, "src": c["ctx"] + c["ids"], "tgt": ids[:t_max]})
            continue
        ids = tokenizer.encode(" " + summary).ids
        st["summary_tokens"] += len(ids)
        st["targets_truncated"] += len(ids) > t_max
        chunks = prep["chunks"]
        st["chunks"] += len(chunks)
        if plan["strategy"] == "head":
            src = chunks[0]["ids"]
            st["chunks_dropped"] += len(chunks) - 1
        else:                                                    # fid
            n, k = len(chunks), plan["fid_chunks"]
            pick = range(n) if n <= k else sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})
            src = [chunks[i]["ids"] for i in pick]
            st["chunks_dropped"] += n - len(pick)
        exs.append({"doc": d, "k": 0, "src": src, "tgt": ids[:t_max]})
    return exs, prepared, targets, st


def ex_size(ex, model):
    s = ex["src"]
    n = sum(len(c) for c in s) if s and isinstance(s[0], list) else len(s)
    return n + len(ex["tgt"]) + len(model.sep) + 2 + model.n_prompt


def micro_batches(sizes, micro_tokens, rng):
    order = sorted(range(len(sizes)), key=lambda i: sizes[i] + rng.random() * 48)
    batches, cur, width = [], [], 0
    for i in order:
        w = max(width, sizes[i])
        if cur and w * (len(cur) + 1) > micro_tokens:
            batches.append(cur)
            cur, w = [], sizes[i]
        cur.append(i)
        width = w
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def heldout_loss(model, exs, args, device, dtype):
    model.eval()
    sizes = [ex_size(e, model) for e in exs]
    total, n = 0.0, 0
    for mb in micro_batches(sizes, args.micro_tokens * 2, random.Random(0)):
        with autocast(device, dtype):
            logits, labels, _ = model.target_logits([exs[i] for i in mb], device)
        total += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        n += labels.numel()
    model.train()
    return total / max(1, n)


# --------------------------------------------------------------------------- knowledge distillation
def _ex_key(e):
    flat = [t for c in e["src"] for t in c] if e["src"] and isinstance(e["src"][0], list) else e["src"]
    return f"{e['doc']}:{e['k']}:{hashlib.md5(np.asarray(flat, dtype=np.int32).tobytes()).hexdigest()[:12]}"


def seq_kd_texts(teacher_dir, exs, args, device, dtype):
    """{example key: teacher summary text} for the training examples (cached in the teacher directory)."""
    teacher_dir = Path(teacher_dir)
    cache_path = teacher_dir / "kd_seq_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [e for e in exs if _ex_key(e) not in cache]
    if args.kd_seq_docs:
        allowed = set(sorted({e["doc"] for e in exs})[:args.kd_seq_docs])
        todo = [e for e in todo if e["doc"] in allowed]
    if todo:
        teacher, ttok, _, _ = load_summarizer(teacher_dir, device)
        dec = Decoding(beams=args.kd_seq_beams, no_repeat_ngram=args.no_repeat_ngram, batch=args.gen_batch)
        log(f"seq-KD: teacher {teacher_dir.name} summarizes {len(todo)} training examples "
            f"({len(exs) - len(todo)} cached)")
        for s in range(0, len(todo), 1024):
            part = todo[s:s + 1024]
            prompts = [teacher.fit_source(e["src"], int(len(e["tgt"]) * 1.2) + 8) for e in part]
            his = [max(2, min(teacher.room(p), int(len(e["tgt"]) * 1.2) + 8)) for p, e in zip(prompts, part)]
            los = [min(h - 1, int(len(e["tgt"]) * 0.8)) for h, e in zip(his, part)]
            outs = generate(teacher, prompts, los, his, dec, device, dtype)
            for e, ids in zip(part, outs):
                cache[_ex_key(e)] = ttok.decode(ids, skip_special_tokens=True)
            cache_path.write_text(json.dumps(cache))
            log(f"  seq-KD {min(len(todo), s + 1024)}/{len(todo)}")
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache


# --------------------------------------------------------------------------- generation + scoring
def eval_ids(dev, n):
    ids = sorted(dev, key=lambda d: hashlib.md5(f"eval:{d}".encode()).hexdigest())
    return ids[:n] if n else ids


def keep_sets(plan, args, docs):
    if not plan.get("extract"):
        return None
    from .baseline import Extractor
    if not args.extractor or not Path(args.extractor).exists():
        raise SystemExit("--strategy extract needs --extractor .../B1_extractive/extractor.pkl (run the baseline)")
    return Extractor.load(args.extractor).keep_sets(docs, args.extract_keep)


def score_and_submit(model, tokenizer, plan, dec, dev, test, out, args, device, dtype, keep_dev=None,
                     keep_test=None, dev_only=False):
    res = {}
    ids = eval_ids(dev, args.eval_docs)
    sub = {d: dev[d] for d in ids}
    t0 = time.time()
    pred, diag = summarize(model, tokenizer, sub, plan, dec, device, dtype, keep_dev, log)
    res["dev_generation_seconds"] = round(time.time() - t0, 1)
    res["dev"] = evaluate(pred, {d: r["summary"] for d, r in sub.items()}, not args.no_bertscore, device,
                          {d: r["document"] for d, r in sub.items()})
    res["dev"]["hit_max_fraction"] = float(np.mean([v["hit_max"] / max(1, v["prompts"]) for v in diag.values()]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=0))
    log({"dev": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["dev"].items()}})
    if not dev_only:
        t0 = time.time()
        tp, _ = summarize(model, tokenizer, test, plan, dec, device, dtype, keep_test, log)
        res["test_generation_seconds"] = round(time.time() - t0, 1)
        want = {d: plan["length"].predict(len(r["document"].split()), plan["len_scale"]) for d, r in test.items()}
        res["submission"] = write_submission(tp, test, out / "submission", want)
        log(f"test: {res['submission']} -> {out / 'submission' / 'submission.zip'}")
    return res


def plan_for_json(plan):
    return {**plan, "length": plan["length"].state()}


def plan_from_json(plan):
    return {**plan, "length": LengthModel(**plan["length"])}


# --------------------------------------------------------------------------- fine-tuning
def finetune(args):
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device, dtype = device_and_amp()
    train, dev, test = load_splits(args.data, args.dev_size)
    if args.max_train_docs and len(train) > args.max_train_docs:
        train = {d: train[d] for d in sorted(rng.sample(sorted(train), args.max_train_docs))}
    if args.random_init and args.method != "full":
        raise SystemExit("--random-init trains every weight: use it with --method full")
    model, tokenizer, info = build_summarizer(args)
    model.to(device)
    plan = resolve_plan(args, model)
    if args.teacher:
        targs = json.loads((Path(args.teacher) / "args.json").read_text())
        same = lambda s: "chunk" if s in ("chunk", "extract") else s
        if same(targs.get("strategy", "chunk")) != same(args.strategy):
            raise SystemExit(f"teacher strategy {targs.get('strategy')} != student strategy {args.strategy}: the "
                             "teacher must be trained on the same kind of examples (chunk/extract, head or fid)")
    keep = keep_sets(plan, args, {**train, **dev, **test})
    t0 = time.time()
    tr_exs, prepared, targets, st = build_examples(tokenizer, train, plan, keep)
    dv_exs, _, _, _ = build_examples(tokenizer, dev, plan, keep)
    plan["length"] = LengthModel.fit(train)
    plan["tpw"] = st["summary_tokens"] / max(1, st["summary_words"])
    plan["profile"] = density_profile(prepared, targets) if plan["strategy"] == "chunk" else None
    sizes = [ex_size(e, model) for e in tr_exs]
    log({"train_docs": len(train), "dev_docs": len(dev), "test_docs": len(test), "train_examples": len(tr_exs),
         "dev_examples": len(dv_exs), **{k: int(v) for k, v in st.items()},
         "tokens_per_summary_word": round(plan["tpw"], 3), "example_tokens_median": int(np.median(sizes)),
         "plan": {k: v for k, v in plan.items() if k not in ("length", "profile")}, "profile": plan["profile"],
         "length_model": plan["length"].state(), "prepare_s": round(time.time() - t0), **info})

    teacher, kd_texts = None, None
    if args.teacher:
        if args.kd == "logit":
            teacher, ttok, targs, _ = load_summarizer(args.teacher, device)
            if not same_tokenizer(ttok, tokenizer):
                raise SystemExit("--kd logit needs the teacher and the student to share the tokenizer; use --kd seq")
            for p in teacher.parameters():
                p.requires_grad = False
            info["teacher"] = {"run": args.teacher, "architecture": teacher.architecture,
                               "parameters": sum(p.numel() for p in teacher.parameters())}
        else:
            kd_texts = seq_kd_texts(args.teacher, tr_exs, args, device, dtype)
            hit = 0
            for e in tr_exs:
                text = kd_texts.get(_ex_key(e))
                if text:
                    e["kd_tgt"] = tokenizer.encode(" " + text).ids[:plan["max_target"]] or e["tgt"]
                    hit += 1
            info["teacher"] = {"run": args.teacher, "kd": "seq", "examples_with_teacher_target": hit}
        log({"teacher": info["teacher"]})

    full_like = args.method in {"full", "partial", "normfit"} or args.random_init
    lr = args.lr_full if full_like else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    per_epoch = max(1, math.ceil(len(tr_exs) / args.batch_examples))
    total = per_epoch * args.epochs
    warmup = max(1, int(args.warmup * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.05, (total - s) / max(1, total - warmup))))
    factory = getattr(model, "allocator_factory", None)
    allocator = factory(model, args, total) if factory else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best, history, step = {"loss": float("inf")}, [], 0
    trained_tokens = 0

    def checkpoint_eval(epoch, losses):
        nonlocal best
        loss = heldout_loss(model, dv_exs, args, device, dtype)
        row = {"epoch": epoch, "step": step, "train_loss": round(float(np.mean(losses[-300:] or [0])), 4),
               "dev_loss": round(loss, 4), "minutes": round((time.time() - started) / 60, 1)}
        log(row)
        history.append(row)
        if loss < best["loss"]:
            best = {"loss": loss, "row": row,
                    "state": {k: v.detach().cpu().clone() for k, v in trainable_state(model).items()}}
            save_model(out / "model.pt", model, args, {"plan": plan_for_json(plan), "dev_row": row})

    model.train()
    for epoch in range(1, args.epochs + 1):
        mbs = micro_batches(sizes, args.micro_tokens, rng)
        steps, cur, n = [], [], 0
        for mb in mbs:
            cur.append(mb)
            n += len(mb)
            if n >= args.batch_examples:
                steps.append(cur)
                cur, n = [], 0
        if cur:
            steps.append(cur)
        losses = []
        t_epoch = time.time()
        for step_mbs in steps:
            chosen = {}
            for mb in step_mbs:
                for i in mb:
                    e = tr_exs[i]
                    if "kd_tgt" in e and rng.random() < args.kd_seq_mix:
                        e = {**e, "tgt": e["kd_tgt"]}
                    chosen[i] = e
            n_tgt = sum(len(e["tgt"]) + 1 for e in chosen.values())
            optimizer.zero_grad(set_to_none=True)
            for j, mb in enumerate(step_mbs):
                exs = [chosen[i] for i in mb]
                with autocast(device, dtype):
                    logits, labels, aux = model.target_logits(exs, device, args.src_lm)
                ce = F.cross_entropy(logits.float(), labels, reduction="sum", label_smoothing=args.label_smoothing)
                loss = ce
                if teacher is not None:
                    with torch.no_grad(), autocast(device, dtype):
                        t_logits, t_labels, _ = teacher.target_logits(exs, device)
                    if not torch.equal(t_labels, labels):
                        raise RuntimeError("teacher and student target tokens differ")
                    tau = args.kd_temperature
                    kl = F.kl_div(F.log_softmax(logits.float() / tau, -1), F.log_softmax(t_logits.float() / tau, -1),
                                  log_target=True, reduction="sum") * tau * tau
                    loss = args.kd_alpha * ce + (1 - args.kd_alpha) * kl
                    del t_logits
                part = loss / n_tgt
                if aux is not None:
                    n_src = sum(len(e["src"]) - 1 for e in exs)
                    part = part + args.src_lm * aux / max(1, n_src) * labels.numel() / n_tgt
                if allocator is not None and args.adalora_orth and j == 0:
                    part = part + args.adalora_orth * allocator.orth_penalty()
                scaler.scale(part).backward()
                losses.append(float(ce) / max(1, labels.numel()))
                trained_tokens += sum(ex_size(e, model) for e in exs)
                del logits, loss, part
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if allocator is not None:
                allocator.step(step)
            if step % 50 == 0:
                log({"epoch": epoch, "step": step, "of": total, "loss": round(float(np.mean(losses[-50:])), 4),
                     "lr": round(scheduler.get_last_lr()[0], 7),
                     "tokens_per_s": int(trained_tokens / max(1e-6, time.time() - started)),
                     "minutes": round((time.time() - started) / 60, 1)})
            if args.eval_every and step % args.eval_every == 0:
                checkpoint_eval(epoch, losses)
        log(f"epoch {epoch}: {round((time.time() - t_epoch) / 60, 1)} min")
        checkpoint_eval(epoch, losses)

    train_minutes = round((time.time() - started) / 60, 1)
    peak = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None
    del teacher
    model.load_state_dict(best["state"], strict=False)
    model.eval()
    dec = decoding_from(args)
    keep_dev = {d: keep[d] for d in dev} if keep else None
    keep_test = {d: keep[d] for d in test} if keep else None
    res = score_and_submit(model, tokenizer, plan, dec, dev, test, out, args, device, dtype, keep_dev, keep_test)
    summary = {"stage": "finetune", "method": args.method, "strategy": args.strategy, "architecture": model.architecture,
               "backbone": args.checkpoint, "teacher": info.get("teacher"), "kd": args.kd if args.teacher else None,
               "best": best["row"], "history": history, "plan": plan_for_json(plan), "decoding": vars(dec),
               "parameters": info["parameters"], "peft": info["peft"], "trainable_parameters":
               info["parameters"]["trainable"], "train_minutes": train_minutes, "peak_gpu_gib": peak,
               "minutes": round((time.time() - started) / 60, 1), **res}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    d = res["dev"]
    log(f"dev BERTScore-F1 {d.get('bertscore_f1', float('nan')):.4f} ROUGE-1 {d['rouge1']:.4f} ROUGE-2 "
        f"{d['rouge2']:.4f} ROUGE-L {d['rougeL']:.4f} words {d['words_pred']:.0f}/{d['words_ref']:.0f} "
        f"| {out / 'summary.json'}")


def decoding_from(args, base=None):
    base = dict(base or {})
    for k in ("beams", "temperature", "top_p", "no_repeat_ngram", "repetition_penalty", "length_penalty", "mbr"):
        v = getattr(args, k, None)
        if v is not None:
            base[k] = v
    base["batch"] = args.gen_batch
    return Decoding(**{k: v for k, v in base.items() if k in Decoding.__dataclass_fields__})


# --------------------------------------------------------------------------- re-decoding a finished run
TUNE = {
    "default": {},
    "len0.8": {"len_scale": 0.8}, "len1.2": {"len_scale": 1.2},
    "len0.4": {"len_scale": 0.4}, "len0.5": {"len_scale": 0.5}, "len0.6": {"len_scale": 0.6},
    "len0.7": {"len_scale": 0.7},
    "greedy": {"beams": 1}, "sample": {"beams": 1, "temperature": 0.7, "top_p": 0.9},
    "noblock": {"no_repeat_ngram": 0}, "block3": {"no_repeat_ngram": 3},
    "uniform": {"budget": "uniform"}, "learned": {"budget": "learned"}, "nodedup": {"dedup": 0.0},
    "beam8": {"beams": 8}, "mbr8": {"beams": 8, "mbr": 8}, "mbr8_len1.2": {"beams": 8, "mbr": 8, "len_scale": 1.2},
}
PLAN_KEYS = ("len_scale", "budget", "dedup", "min_ratio", "max_ratio")


def _load_run(args):
    device, dtype = device_and_amp()
    model, tokenizer, margs, payload = load_summarizer(args.run, device)
    plan = plan_from_json(payload["plan"])
    train, dev, test = load_splits(args.data, margs.dev_size, verbose=False)
    keep = keep_sets(plan, margs, {**dev, **test})
    summ = json.loads((Path(args.run) / "summary.json").read_text()) if (Path(args.run) / "summary.json").exists() else {}
    return device, dtype, model, tokenizer, margs, plan, dev, test, keep, summ.get("decoding", {})


def decode(args):
    """Re-generate a finished run with other decoding / length settings -> RUN/decode_TAG/."""
    device, dtype, model, tokenizer, margs, plan, dev, test, keep, base_dec = _load_run(args)
    for k in PLAN_KEYS:
        if getattr(args, k, None) is not None:
            plan[k] = getattr(args, k)
    dec = decoding_from(args, base_dec)
    out = Path(args.run) / f"decode_{args.tag}"
    margs.eval_docs, margs.no_bertscore = args.eval_docs, args.no_bertscore
    res = score_and_submit(model, tokenizer, plan, dec, dev, test, out, margs, device, dtype,
                           {d: keep[d] for d in dev} if keep else None, {d: keep[d] for d in test} if keep else None,
                           dev_only=args.dev_only)
    (out / "summary.json").write_text(json.dumps({"plan": plan_for_json(plan), "decoding": vars(dec), **res},
                                                 indent=1, default=str))


def ensemble(args):
    """Decode with several finished runs at once (mixture of their next-token distributions) -> OUTPUT/."""
    from .model import Ensemble
    device, dtype = device_and_amp()
    models, toks, plan, margs, base_dec = [], [], None, None, {}
    for i, run in enumerate(args.runs):
        m, tok, a, payload = load_summarizer(run, device)
        models.append(m)
        toks.append(tok)
        if i == 0:
            plan, margs = plan_from_json(payload["plan"]), a
            sp = Path(run) / "summary.json"
            base_dec = json.loads(sp.read_text()).get("decoding", {}) if sp.exists() else {}
    if any(not same_tokenizer(toks[0], t) for t in toks[1:]):
        raise SystemExit("ensemble members must share the tokenizer")
    for k in PLAN_KEYS:
        if getattr(args, k, None) is not None:
            plan[k] = getattr(args, k)
    if plan.get("extract"):
        raise SystemExit("the first ensemble member must use the chunk strategy")
    model = Ensemble(models, args.weights)
    train, dev, test = load_splits(args.data, margs.dev_size, verbose=False)
    dec = decoding_from(args, base_dec)
    out = Path(args.output)
    margs.eval_docs, margs.no_bertscore = args.eval_docs, args.no_bertscore
    log({"ensemble": [str(r) for r in args.runs], "weights": args.weights, "architectures":
         [m.architecture for m in models], "plan": {k: v for k, v in plan.items() if k not in ("length", "profile")}})
    res = score_and_submit(model, toks[0], plan, dec, dev, test, out, margs, device, dtype, dev_only=args.dev_only)
    (out / "summary.json").write_text(json.dumps({"stage": "ensemble", "method": "ensemble", "strategy": plan["strategy"],
                                                  "runs": [str(r) for r in args.runs], "weights": args.weights,
                                                  "plan": plan_for_json(plan), "decoding": vars(dec), **res},
                                                 indent=1, default=str))


def tune(args):
    """One-factor-at-a-time decoding/length variants on held-out documents; the best one also decodes the test
    set (RUN/tuned/submission)."""
    device, dtype, model, tokenizer, margs, plan0, dev, test, keep, base_dec = _load_run(args)
    margs.eval_docs, margs.no_bertscore = args.eval_docs, args.no_bertscore
    names = args.variants or list(TUNE)
    rows = {}
    for name in names:
        v = TUNE[name]
        plan = {**plan0, **{k: v[k] for k in PLAN_KEYS if k in v}}
        if plan["strategy"] != "chunk" and name in ("uniform", "learned"):
            continue
        dec = Decoding(**{**{k: x for k, x in base_dec.items() if k in Decoding.__dataclass_fields__},
                          **{k: x for k, x in v.items() if k in Decoding.__dataclass_fields__}, "batch": args.gen_batch})
        res = score_and_submit(model, tokenizer, plan, dec, dev, test, Path(args.run) / f"tune_{name}", margs,
                               device, dtype, {d: keep[d] for d in dev} if keep else None, None, dev_only=True)
        rows[name] = {"variant": v, **{k: res["dev"].get(k) for k in ("selection_score", "bertscore_f1", "rouge1",
                                                                     "rouge2", "rougeL", "words_pred", "length_ratio",
                                                                     "redundancy_3gram", "number_precision")},
                      "seconds": res["dev_generation_seconds"]}
        log({name: rows[name]})
    best = max(rows, key=lambda k: rows[k]["selection_score"] or 0)
    (Path(args.run) / "tune.json").write_text(json.dumps({"best": best, "rows": rows}, indent=1))
    log(f"best variant: {best} {rows[best]['variant']}")
    if not args.no_test and best != "default":
        v = TUNE[best]
        plan = {**plan0, **{k: v[k] for k in PLAN_KEYS if k in v}}
        dec = Decoding(**{**{k: x for k, x in base_dec.items() if k in Decoding.__dataclass_fields__},
                          **{k: x for k, x in v.items() if k in Decoding.__dataclass_fields__}, "batch": args.gen_batch})
        tp, _ = summarize(model, tokenizer, test, plan, dec, device, dtype, {d: keep[d] for d in test} if keep else None, log)
        want = {d: plan["length"].predict(len(r["document"].split()), plan["len_scale"]) for d, r in test.items()}
        info = write_submission(tp, test, Path(args.run) / "tuned" / "submission", want)
        log(f"tuned test submission ({best}): {info} -> {Path(args.run) / 'tuned' / 'submission' / 'submission.zip'}")


def add_args(p):
    p.add_argument("--data", required=True, help="Phase_6_Summ directory (Train/, Test/)")
    p.add_argument("--checkpoint", required=True, help="Phase-1 best.pt (tokenizer.json alongside)")
    p.add_argument("--project-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--dev-size", type=int, default=200, help="held-out training documents (no dev split is given)")
    p.add_argument("--method", default="lorma", choices=["lorma", "lora", "rslora", "qlora", "adalora", "houlsby",
                                                        "pfeiffer", "adapter", "prefix", "prompt", "ia3", "normfit",
                                                        "partial", "full", "frozen"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--keep-layers", type=int, help="truncate the Phase-1 model (KD students)")
    p.add_argument("--layer-pick", default="first", choices=["first", "spread"],
                   help="which layers a truncated student keeps: the first k, or k evenly spaced incl. the last")
    p.add_argument("--random-init", action="store_true", help="ablation: the architecture without Phase-1 weights")
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--prefix-length", type=int, default=24)
    p.add_argument("--prefix-hidden", type=int, default=512)
    p.add_argument("--adalora-init-r", type=int, default=12)
    p.add_argument("--adalora-target-r", type=int, default=8)
    p.add_argument("--adalora-orth", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
    # long documents
    p.add_argument("--strategy", default="chunk", choices=["chunk", "head", "extract", "fid"])
    p.add_argument("--chunk", type=int, default=1024, help="source tokens per chunk (reduced to fit the context)")
    p.add_argument("--overlap", type=int, default=0, help="tokens of the previous chunk prepended as context")
    p.add_argument("--max-target", type=int, default=0, help="target tokens per example (0 = by strategy)")
    p.add_argument("--fid-chunks", type=int, default=8)
    p.add_argument("--align", default="tfidf", choices=["tfidf", "position"])
    p.add_argument("--extractor", help="extractor.pkl from the baseline (strategy extract)")
    p.add_argument("--extract-keep", type=float, default=0.5, help="share of the document's words kept")
    # length / post-processing
    p.add_argument("--budget", default="learned", choices=["learned", "uniform"])
    p.add_argument("--len-scale", type=float, default=1.0, help="x predicted summary length")
    p.add_argument("--min-ratio", type=float, default=0.8)
    p.add_argument("--max-ratio", type=float, default=1.3)
    p.add_argument("--dedup", type=float, default=0.6, help="drop sentences with >= this share of seen trigrams")
    # optimisation
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per forward")
    p.add_argument("--batch-examples", type=int, default=32, help="examples per optimizer step")
    p.add_argument("--max-train-docs", type=int, default=0)
    p.add_argument("--lr-peft", type=float, default=None)
    p.add_argument("--lr-full", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--src-lm", type=float, default=0.0, help="weight of the source language-model loss")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # knowledge distillation
    p.add_argument("--teacher", help="finished phase6 run directory (model.pt)")
    p.add_argument("--kd", default="logit", choices=["logit", "seq"])
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--kd-seq-mix", type=float, default=0.5)
    p.add_argument("--kd-seq-beams", type=int, default=1)
    p.add_argument("--kd-seq-docs", type=int, default=0, help="limit the documents the teacher summarizes")
    add_decode_args(p, defaults=True)


def add_decode_args(p, defaults=False):
    d = Decoding()
    p.add_argument("--beams", type=int, default=d.beams if defaults else None)
    p.add_argument("--temperature", type=float, default=d.temperature if defaults else None)
    p.add_argument("--top-p", type=float, default=d.top_p if defaults else None)
    p.add_argument("--no-repeat-ngram", type=int, default=d.no_repeat_ngram if defaults else None)
    p.add_argument("--repetition-penalty", type=float, default=d.repetition_penalty if defaults else None)
    p.add_argument("--length-penalty", type=float, default=d.length_penalty if defaults else None)
    p.add_argument("--mbr", type=int, default=d.mbr if defaults else None,
                   help="minimum-Bayes-risk choice among the top N beam hypotheses (needs --beams >= N)")
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--eval-docs", type=int, default=100, help="held-out documents summarized for scoring (0 = all)")
    p.add_argument("--no-bertscore", action="store_true")
