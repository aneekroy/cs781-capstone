"""Multi-label statute identification on a Phase-1 backbone (any phase3.peft technique) - training and inference.

Document -> windows. The facts of a case (~2.4k words, ~3.5k tokens) are tokenized once and cut into windows of
--chunk tokens (--stride < chunk gives overlapping windows). Training sees up to --train-chunks windows per
document (a random subset, kept in order, re-drawn every epoch); inference sees up to --eval-chunks (evenly
spaced if there are more). --select head|tail restricts both to the start or the end of the text (ablation
"complete document vs selected part").

Windows -> document. Each window becomes a vector (mean + last-token pooling of the encoder states, projected to
--hidden, plus an embedding of its relative position). --pool chooses how the document is formed:
  label  label-wise attention (each statute attends to the windows that describe its offence; LAAT-style),
  attn   one shared attention over windows, then a linear layer,
  max    a statute score per window, max over windows,
  mean   mean of the window vectors, then a linear layer.

Loss (--loss): bce | wbce (per-statute positive weight sqrt(neg/pos), capped) | asl (asymmetric loss,
Ridnik et al. 2021: gamma_neg=4, clip 0.05 - the many easy negatives of 100 statutes are down-weighted) |
focal. --rare-repeat N repeats documents that carry a rare statute (bottom quartile by frequency) N times per epoch.

Knowledge distillation (--teacher RUN): the teacher's document logits on the training documents (computed once,
cached in the teacher's folder) give soft targets sigmoid(z/T); the student mixes them with the gold labels
(--kd-alpha). Teacher and student may use different tokenizers: the targets are per document and statute.
Memory: documents are processed --docs-per-forward at a time with backward right after, gradients accumulated
over --doc-batch documents, so one optimizer step never holds more than a few documents' activations.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase2.run import autocast, device_and_amp
from phase3.model import build_encoder, trainable_state

from .data import finish, label_space, load_split, multi_hot, scores, apply_thresholds


# --------------------------------------------------------------------------- windows
def doc_windows(tokenizer, text, chunk, stride, max_tokens):
    ids = tokenizer.encode(" " + " ".join(text.split())).ids[:max_tokens] or [tokenizer.token_to_id("<eos>") or 1]
    stride = stride or chunk
    windows = [ids[i:i + chunk] for i in range(0, max(1, len(ids) - chunk + stride), stride)]
    windows = [w for w in windows if w] or [ids[:chunk]]
    if len(windows) > 1 and len(windows[-1]) < chunk // 4 and stride == chunk:
        windows[-2] = windows[-2] + windows[-1]
        windows[-2] = windows[-2][-chunk:]
        windows.pop()
    return windows


def pick(windows, k, select, rng=None):
    """Indices of the windows to use: at most k, in document order."""
    n = len(windows)
    idx = list(range(n))
    if select == "head":
        return idx[:k]
    if select == "tail":
        return idx[-k:]
    if n <= k:
        return idx
    if rng is not None:
        return sorted(rng.sample(idx, k))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


# --------------------------------------------------------------------------- model
class StatuteClassifier(nn.Module):
    def __init__(self, encoder, n_labels, pool="label", hidden=512, dropout=0.1, positions=16):
        super().__init__()
        self.encoder = encoder
        self.pool, self.positions = pool, positions
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * encoder.dim, hidden), nn.GELU())
        self.pos = nn.Embedding(positions, hidden)
        nn.init.zeros_(self.pos.weight)
        self.drop = nn.Dropout(dropout)
        if pool == "label":
            self.att_in = nn.Linear(hidden, hidden)
            self.queries = nn.Parameter(torch.randn(n_labels, hidden) * 0.02)
            self.out_w = nn.Parameter(torch.randn(n_labels, hidden) * 0.02)
            self.out_b = nn.Parameter(torch.zeros(n_labels))
        elif pool == "attn":
            self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.head = nn.Linear(hidden, n_labels)
        elif pool in ("max", "mean"):
            self.head = nn.Linear(hidden, n_labels)
        else:
            raise ValueError(pool)

    def window_vectors(self, ids, lengths):
        h = self.encoder(ids)
        mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (h * mask).sum(1) / lengths[:, None].clamp_min(1)
        last = h[torch.arange(ids.size(0), device=ids.device), (lengths - 1).clamp_min(0)]
        return torch.cat((mean, last), -1).float()

    def doc_logits(self, vecs, rel_pos):
        """vecs [C, 2D] window vectors of ONE document, rel_pos [C] in [0,1) -> logits [L]."""
        v = self.proj(vecs) + self.pos((rel_pos * self.positions).long().clamp(0, self.positions - 1))
        if self.pool == "label":
            a = torch.softmax(torch.tanh(self.att_in(v)) @ self.queries.t(), 0)      # [C, L]
            d = a.t() @ self.drop(v)                                                  # [L, H]
            return (d * self.out_w).sum(-1) + self.out_b
        if self.pool == "attn":
            a = torch.softmax(self.att(v).squeeze(-1), 0)
            return self.head(self.drop((a[:, None] * v).sum(0)))
        if self.pool == "max":
            return self.head(self.drop(v)).max(0).values
        return self.head(self.drop(v.mean(0)))

    def forward(self, docs_windows, pad_id, device, dtype):
        """docs_windows: list (per document) of (windows, rel_positions) -> logits [B, L]."""
        flat = [w for ws, _ in docs_windows for w in ws]
        length = max(len(w) for w in flat)
        ids = torch.full((len(flat), length), pad_id, dtype=torch.long)
        for i, w in enumerate(flat):
            ids[i, :len(w)] = torch.tensor(w)
        lengths = torch.tensor([len(w) for w in flat])
        with autocast(device, dtype):
            vecs = self.window_vectors(ids.to(device), lengths.to(device))
        out, at = [], 0
        for ws, pos in docs_windows:
            out.append(self.doc_logits(vecs[at:at + len(ws)], torch.tensor(pos, device=device, dtype=torch.float32)))
            at += len(ws)
        return torch.stack(out)


def build_model(args, n_labels):
    if not hasattr(args, "window"):
        args.window = args.chunk
    encoder, tokenizer, info = build_encoder(args)
    args.chunk = min(args.chunk, args.window)              # a soft prompt takes positions from the window
    model = StatuteClassifier(encoder, n_labels, args.pool, args.hidden, args.dropout)
    from phase2.peft import count_parameters
    info["parameters"] = count_parameters(model)
    info["encoder_parameters"] = count_parameters(encoder)
    return model, tokenizer, info


def save_model(path, model, args, labels, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "labels": labels, "state": trainable_state(model), **(extra or {})}, path)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    model, tokenizer, _ = build_model(args, len(payload["labels"]))
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if unexpected or [m for m in missing if m in trainable or not m.startswith("encoder.")]:
        raise ValueError(f"{path}: state does not fit: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval(), tokenizer, args, payload["labels"]


# --------------------------------------------------------------------------- inference
def tokenize_docs(tokenizer, docs, args):
    return {d: doc_windows(tokenizer, doc["text"], args.chunk, args.stride, args.max_tokens) for d, doc in docs.items()}


def with_positions(windows, idx):
    n = len(windows)
    return [windows[i] for i in idx], [(i + 0.5) / n for i in idx]


@torch.no_grad()
def predict(model, tokenizer, docs, args, device, dtype, windows=None):
    """[n_docs, L] logits in docs order (no gradients; documents packed up to --eval-batch windows)."""
    model.eval()
    pad_id = tokenizer.token_to_id("<pad>") or 0
    windows = windows or tokenize_docs(tokenizer, docs, args)
    out, group, size = [], [], 0
    order = list(docs)

    def flush():
        out.append(model(group, pad_id, device, dtype).float().cpu())

    for d in order:
        item = with_positions(windows[d], pick(windows[d], args.eval_chunks, args.select))
        if group and size + len(item[0]) > args.eval_batch:
            flush()
            group, size = [], 0
        group.append(item)
        size += len(item[0])
    if group:
        flush()
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------- losses
def statute_loss(logits, y, kind, pos_weight=None, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, focal_gamma=2.0):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, y)
    if kind == "wbce":
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    if kind == "asl":
        p_neg = (p - clip).clamp(min=0) if clip else p
        pos = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gamma_pos
        neg = (1 - y) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** gamma_neg
        return -(pos + neg).sum(-1).mean() / y.size(-1)
    if kind == "focal":
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.where(y > 0, p, 1 - p)
        return ((1 - pt) ** focal_gamma * ce).mean()
    raise ValueError(kind)


# --------------------------------------------------------------------------- KD
def _key(d, doc):
    return f"{d}:{hashlib.md5(' '.join(doc['text'].split()).encode()).hexdigest()[:12]}"


def teacher_logits(teacher_dir, docs, device, dtype, labels):
    """Teacher logits [n, L] for docs (cached next to the teacher's model.pt, keyed by id + text hash)."""
    path = Path(teacher_dir) / "model.pt"
    cache_path = Path(teacher_dir) / "kd_train_logits.npz"
    keys = [_key(d, doc) for d, doc in docs.items()]
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if int(z["stamp"]) == int(path.stat().st_mtime):
            cache = dict(zip(z["keys"], z["logits"]))
    missing = {d: doc for (d, doc), k in zip(docs.items(), keys) if k not in cache}
    n_params = None
    if missing:
        teacher, tok, targs, t_labels = load_model(path, device)
        if list(t_labels) != list(labels):
            raise SystemExit("teacher and student label spaces differ")
        n_params = sum(p.numel() for p in teacher.parameters())
        z = predict(teacher, tok, missing, targs, device, dtype)
        cache.update({_key(d, doc): row for (d, doc), row in zip(missing.items(), z)})
        np.savez_compressed(cache_path, keys=np.array(list(cache), dtype=object), logits=np.stack(list(cache.values())),
                            stamp=int(path.stat().st_mtime))
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"KD: teacher {teacher_dir}: {len(docs) - len(missing)} documents from cache, {len(missing)} computed"
          + (f" ({n_params:,} parameters)" if n_params else ""), flush=True)
    return np.stack([cache[k] for k in keys])


# --------------------------------------------------------------------------- training
def finetune(args):
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device, dtype = device_and_amp()
    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    labels = label_space(train, dev)
    if args.max_train_docs and len(train) > args.max_train_docs:
        keep = sorted(rng.sample(sorted(train), args.max_train_docs))
        train = {d: train[d] for d in keep}
    y_train, y_dev = multi_hot(train, labels).astype(np.float32), multi_hot(dev, labels)
    soft = None
    if args.teacher:
        soft = sigmoid(teacher_logits(args.teacher, train, device, dtype, labels) / args.kd_temperature).astype(np.float32)
    model, tokenizer, info = build_model(args, len(labels))
    model.to(device)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    t0 = time.time()
    tr_windows = tokenize_docs(tokenizer, train, args)
    dev_windows = tokenize_docs(tokenizer, dev, args)
    n_win = [len(w) for w in tr_windows.values()]
    print(json.dumps({"train_docs": len(train), "dev_docs": len(dev), "test_docs": len(test), "statutes": len(labels),
                      "windows_per_doc_median": float(np.median(n_win)), "windows_per_doc_p90": float(np.percentile(n_win, 90)),
                      "labels_per_doc": round(float(y_train.sum(1).mean()), 2), "tokenize_s": round(time.time() - t0),
                      **info}, default=str), flush=True)
    freq = y_train.sum(0)
    pos_weight = torch.tensor(np.clip(np.sqrt((len(y_train) - freq) / np.maximum(freq, 1)), 1, args.max_pos_weight),
                              device=device, dtype=torch.float32)
    rare = set(np.flatnonzero((freq > 0) & (freq <= np.percentile(freq[freq > 0], 25))))
    order = list(train)
    row_of = {d: i for i, d in enumerate(order)}
    dev_eval = list(dev)
    if args.eval_docs and len(dev_eval) > args.eval_docs:              # epoch selection on a fixed dev subsample
        dev_eval = sorted(random.Random(1).sample(dev_eval, args.eval_docs))
    dev_eval_docs = {d: dev[d] for d in dev_eval}
    y_dev_eval = multi_hot(dev_eval_docs, labels)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    from phase3.run import LR_DEFAULT
    full_like = args.method in {"full", "partial", "normfit"}
    lr_enc = args.lr_full if full_like else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
    optimizer = torch.optim.AdamW([g for g in ({"params": enc_params, "lr": lr_enc},
                                               {"params": head_params, "lr": args.lr_head}) if g["params"]],
                                  weight_decay=args.weight_decay)
    per_epoch = len(order) + (args.rare_repeat - 1) * sum(1 for d in order if set(np.flatnonzero(y_train[row_of[d]])) & rare)
    total = max(1, math.ceil(per_epoch / args.doc_batch) * args.epochs)
    warmup = max(1, int(0.06 * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.02, (total - s) / max(1, total - warmup))))
    factory = getattr(model.encoder, "allocator_factory", None)
    allocator = factory(model.encoder, args, total) if factory else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best, history, step = {"score": -1.0}, [], 0

    def evaluate(epoch, losses):
        nonlocal best
        z = predict(model, tokenizer, dev_eval_docs, args, device, dtype, dev_windows)
        p = sigmoid(z)
        s05 = scores(y_dev_eval, apply_thresholds(p, 0.5))
        from .data import fit_thresholds
        t = fit_thresholds(y_dev_eval, p, "global")
        sg = scores(y_dev_eval, apply_thresholds(p, t))
        row = {"epoch": epoch, "step": step, "loss": round(float(np.mean(losses[-500:] or [0])), 5),
               "dev_macro_f1@0.5": round(s05["macro_f1"], 4), "dev_macro_f1_global_t": round(sg["macro_f1"], 4),
               "global_t": round(float(t), 3), "dev_micro_f1@0.5": round(s05["micro_f1"], 4),
               "minutes": round((time.time() - started) / 60, 1)}
        print(json.dumps(row), flush=True)
        history.append(row)
        if sg["macro_f1"] > best["score"]:
            best = {"score": sg["macro_f1"], "row": row,
                    "state": {k: v.detach().cpu().clone() for k, v in trainable_state(model).items()}}
            save_model(out / "model.pt", model, args, labels, {"dev_row": row})
        model.train()

    for epoch in range(1, args.epochs + 1):
        model.train()
        jobs = []
        for d in order:
            reps = args.rare_repeat if set(np.flatnonzero(y_train[row_of[d]])) & rare else 1
            jobs += [d] * reps
        rng.shuffle(jobs)
        losses = []
        for b in range(0, len(jobs), args.doc_batch):
            batch = jobs[b:b + args.doc_batch]
            optimizer.zero_grad(set_to_none=True)
            for g in range(0, len(batch), args.docs_per_forward):
                group = batch[g:g + args.docs_per_forward]
                items = [with_positions(tr_windows[d], pick(tr_windows[d], args.train_chunks, args.select, rng))
                         for d in group]
                logits = model(items, pad_id, device, dtype).float()
                y = torch.tensor(y_train[[row_of[d] for d in group]], device=device)
                loss = statute_loss(logits, y, args.loss, pos_weight, args.asl_gamma_neg, 0.0, args.asl_clip,
                                    args.focal_gamma)
                if soft is not None:
                    target = torch.tensor(soft[[row_of[d] for d in group]], device=device)
                    kd = F.binary_cross_entropy_with_logits(logits / args.kd_temperature, target) * args.kd_temperature ** 2
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
                part = loss * len(group) / len(batch)
                if allocator is not None and args.adalora_orth and g == 0:
                    part = part + args.adalora_orth * allocator.orth_penalty()
                scaler.scale(part).backward()
                losses.append(float(loss.detach()))
                del logits, loss, part
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if allocator is not None:
                allocator.step(step)
            if args.eval_every and step % args.eval_every == 0:
                evaluate(epoch, losses)
            if step % 200 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": round(float(np.mean(losses[-200:])), 5),
                                  "minutes": round((time.time() - started) / 60, 1)}), flush=True)
        evaluate(epoch, losses)
    model.load_state_dict(best["state"], strict=False)
    p_dev = sigmoid(predict(model, tokenizer, dev, args, device, dtype, dev_windows))
    p_test = sigmoid(predict(model, tokenizer, test, args, device, dtype))
    finish(y_dev, p_dev, p_test, dev, test, labels, out,
           {"stage": "finetune", "method": args.method, "pool": args.pool, "loss": args.loss,
            "backbone": args.init_phase2 or args.checkpoint, "teacher": args.teacher, "history": history,
            "best": best["row"], "parameters": info.get("parameters"), "encoder_parameters": info.get("encoder_parameters"),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peft": info.get("peft"), "minutes": round((time.time() - started) / 60, 1)})


def add_args(p):
    p.add_argument("--data", required=True, help="Phase_5_LSI directory (Train/, Test/)")
    p.add_argument("--checkpoint", help="Phase-1 best.pt (tokenizer.json alongside)")
    p.add_argument("--init-phase2", help="start from a Phase-2 NER model.pt (adapters merged)")
    p.add_argument("--project-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="lorma", choices=["lorma", "lora", "rslora", "qlora", "adalora", "prefix", "prompt",
                                                        "ia3", "adapter", "normfit", "partial", "full", "frozen"])
    p.add_argument("--mode", default="causal", choices=["causal", "bidirectional", "encoder"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--bottleneck", type=int, default=64, help="adapters: bottleneck size")
    p.add_argument("--top-k", type=int, default=4, help="partial fine-tuning: number of top blocks trained")
    p.add_argument("--keep-layers", type=int, help="truncate the Phase-1 model (KD students)")
    p.add_argument("--prompt-tokens", type=int, default=20)
    p.add_argument("--prefix-length", type=int, default=16)
    p.add_argument("--prefix-hidden", type=int, default=512)
    p.add_argument("--adalora-init-r", type=int, default=12)
    p.add_argument("--adalora-target-r", type=int, default=8)
    p.add_argument("--adalora-orth", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
    # document representation
    p.add_argument("--chunk", type=int, default=512, help="tokens per window")
    p.add_argument("--stride", type=int, default=0, help="window step (0 = chunk, no overlap; e.g. 384 for 25%% overlap)")
    p.add_argument("--max-tokens", type=int, default=16384, help="tokens kept per document")
    p.add_argument("--train-chunks", type=int, default=8, help="windows per document in training")
    p.add_argument("--eval-chunks", type=int, default=16, help="windows per document at inference")
    p.add_argument("--select", default="all", choices=["all", "head", "tail"])
    p.add_argument("--pool", default="label", choices=["label", "attn", "max", "mean"])
    p.add_argument("--hidden", type=int, default=512)
    # loss / imbalance
    p.add_argument("--loss", default="asl", choices=["bce", "wbce", "asl", "focal"])
    p.add_argument("--max-pos-weight", type=float, default=30.0)
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--asl-clip", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--rare-repeat", type=int, default=1, help="repeat documents with a bottom-quartile statute")
    # optimisation
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--doc-batch", type=int, default=16, help="documents per optimizer step")
    p.add_argument("--docs-per-forward", type=int, default=2, help="documents encoded together (memory)")
    p.add_argument("--eval-batch", type=int, default=64, help="windows per forward at inference")
    p.add_argument("--eval-docs", type=int, default=3000, help="dev documents used for epoch selection (0 = all)")
    p.add_argument("--eval-every", type=int, default=0, help="optimizer steps between dev evaluations (0 = per epoch)")
    p.add_argument("--max-train-docs", type=int, default=0)
    p.add_argument("--lr-peft", type=float, default=None)
    p.add_argument("--lr-full", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # knowledge distillation
    p.add_argument("--teacher", help="finished phase5 run directory (model.pt)")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
