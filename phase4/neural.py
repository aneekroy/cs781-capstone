"""Stages 1-2: chunk-level fine-tuning of a Phase-1/2 backbone, then chunk embeddings for the aggregator.

Stage 1 (finetune). A judgment is ~3-5k words, far beyond one 512-token window, and the outcome is argued
at the end (ILDC finding: the last 512 tokens carry most of the signal). Each document contributes its last
--tail-chunks windows of --chunk tokens (plus, with --head-chunk, its first window: appeal type and
parties). Every window is a training example labelled with the document's outcome; the document's
probability is the sigmoid of the mean window logit. The backbone is adapted with any phase3.peft
technique (LoRMA by default). A learned chunk-position embedding tells the head how far from the end a
window is. Knowledge distillation (--teacher <stage-1 run>): the teacher's document probability (temperature
T) is a soft target for every window of that document, mixed with the gold label (--kd-alpha); document-
level targets need no shared tokenizer.

Stage 2 (embed). With the stage-1 model, every document is cut into its last --embed-chunks windows (+ head);
each window becomes its mean-pooled vector plus its stage-1 logit, saved per split for stage 3.
"""
import argparse
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

from .data import best_threshold, honest_threshold, load_split, macro_f1, write_prediction


# --------------------------------------------------------------------------- chunks
def doc_chunks(tokenizer, text, chunk, tail_chunks, head_chunk):
    """Token-id windows: the last tail_chunks windows (oldest first) and optionally the first window.
    Only the needed end of the text is tokenized (about 8 characters per token, generous)."""
    need = chunk * tail_chunks
    tail_text = text if len(text) <= need * 8 + 2000 else text[-(need * 8 + 2000):]
    ids = tokenizer.encode(" " + tail_text).ids
    ids = ids[-need:] if len(ids) > need else ids
    windows = [ids[i:i + chunk] for i in range(0, len(ids), chunk)] or [[tokenizer.token_to_id("<eos>") or 1]]
    if len(windows) > 1 and len(windows[0]) < chunk // 4:     # a short leftover at the front: merge it away
        windows = windows[1:]
    out = [("tail", len(windows) - 1 - i, w) for i, w in enumerate(windows)]   # position counted from the end
    if head_chunk:
        head = tokenizer.encode(" " + text[: chunk * 8]).ids[:chunk]
        out = [("head", 0, head)] + out
    return out


# --------------------------------------------------------------------------- model
class ChunkClassifier(nn.Module):
    def __init__(self, encoder, max_positions=64, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        dim = encoder.dim * 2                                  # mean + last-token pooling
        self.pos = nn.Embedding(max_positions + 1, dim)         # index max_positions = head chunk
        nn.init.zeros_(self.pos.weight)
        self.max_positions = max_positions
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)

    def pooled(self, ids, lengths):
        h = self.encoder(ids)
        mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (h * mask).sum(1) / lengths[:, None].clamp_min(1)
        last = h[torch.arange(ids.size(0), device=ids.device), (lengths - 1).clamp_min(0)]
        return mean.float(), torch.cat((mean, last), -1).float()

    def forward(self, ids, lengths, positions):
        mean, vec = self.pooled(ids, lengths)
        vec = vec + self.pos(positions.clamp(0, self.max_positions))
        return self.head(self.drop(vec)).squeeze(-1), mean


def batch_tensors(windows, pad_id, device):
    length = max(len(w) for w in windows)
    ids = torch.full((len(windows), length), pad_id, dtype=torch.long)
    for i, w in enumerate(windows):
        ids[i, :len(w)] = torch.tensor(w)
    lengths = torch.tensor([len(w) for w in windows])
    return ids.to(device), lengths.to(device)


def position_index(kind, pos, max_positions):
    return max_positions if kind == "head" else min(pos, max_positions - 1)


def build_classifier(args):
    if not hasattr(args, "window"):
        args.window = args.chunk
    encoder, tokenizer, info = build_encoder(args)
    args.chunk = min(args.chunk, args.window)              # a soft prompt takes positions from the window
    model = ChunkClassifier(encoder, max_positions=max(args.tail_chunks, args.embed_chunks) + 1,
                            dropout=args.dropout)
    return model, tokenizer, info


def save_classifier(path, model, args, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "state": trainable_state(model), **(extra or {})}, path)


def load_classifier(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    model, tokenizer, _ = build_classifier(args)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if unexpected or [m for m in missing if m in trainable or not m.startswith("encoder.")]:
        raise ValueError(f"{path}: state does not fit: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval(), tokenizer, args


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def doc_logits(model, tokenizer, docs, args, device, dtype, tail_chunks, head_chunk, return_vectors=False):
    """{doc: mean window logit} (and {doc: [n_windows, d+1] fp16 vectors}: pooled mean + logit)."""
    model.eval()
    pad_id = tokenizer.token_to_id("<pad>") or 0
    logits, vectors = {}, {}
    items = []
    for d, doc in docs.items():
        for kind, pos, w in doc_chunks(tokenizer, doc["text"], args.chunk, tail_chunks, head_chunk):
            items.append((d, kind, pos, w))
    per_doc = {}
    for b in range(0, len(items), args.eval_batch):
        batch = items[b:b + args.eval_batch]
        ids, lengths = batch_tensors([w for *_, w in batch], pad_id, device)
        positions = torch.tensor([position_index(k, p, model.max_positions) for _, k, p, _ in batch], device=device)
        with autocast(device, dtype):
            logit, mean = model(ids, lengths, positions)
        for (d, kind, pos, _), lg, m in zip(batch, logit.float().cpu(), mean.float().cpu()):
            per_doc.setdefault(d, []).append((kind, pos, float(lg), m))
    for d, rows in per_doc.items():
        tail = [r for r in rows if r[0] == "tail"] or rows
        logits[d] = float(np.mean([r[2] for r in tail]))
        if return_vectors:
            ordered = [r for r in rows if r[0] == "head"] + sorted(tail, key=lambda r: -r[1])   # head, then oldest -> last
            vectors[d] = torch.stack([torch.cat((r[3], torch.tensor([r[2]]))) for r in ordered]).half()
    return (logits, vectors) if return_vectors else logits


def probs_from_logits(logits):
    return {d: 1 / (1 + math.exp(-v)) for d, v in logits.items()}


# --------------------------------------------------------------------------- stage 1
def finetune(args):
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device, dtype = device_and_amp()
    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    if args.max_train_docs and len(train) > args.max_train_docs:
        keep = rng.sample(sorted(train), args.max_train_docs)
        train = {d: train[d] for d in sorted(keep)}
    teacher_prob = None
    if args.teacher:
        teacher, t_tok, t_args = load_classifier(Path(args.teacher) / "model.pt", device)
        n_teacher = sum(p.numel() for p in teacher.parameters())
        t_logits = doc_logits(teacher, t_tok, train, t_args, device, dtype, t_args.tail_chunks, t_args.head_chunk)
        teacher_prob = {d: 1 / (1 + math.exp(-v / args.kd_temperature)) for d, v in t_logits.items()}
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"KD: teacher {args.teacher} ({n_teacher:,} parameters), soft targets for {len(teacher_prob)} documents",
              flush=True)
    model, tokenizer, info = build_classifier(args)
    model.to(device)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    examples = []
    for d, doc in train.items():
        for kind, pos, w in doc_chunks(tokenizer, doc["text"], args.chunk, args.tail_chunks, args.head_chunk):
            examples.append((d, position_index(kind, pos, model.max_positions), w, doc["label"]))
    y = np.array([e[3] for e in examples])
    pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), device=device) if args.balance else None
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    from phase3.run import LR_DEFAULT
    lr_enc = args.lr_full if args.method in {"full", "partial", "normfit"} else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
    optimizer = torch.optim.AdamW([g for g in ({"params": enc_params, "lr": lr_enc},
                                               {"params": head_params, "lr": args.lr_head}) if g["params"]],
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(0.06 * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.02, (total - s) / max(1, total - warmup))))
    factory = getattr(model.encoder, "allocator_factory", None)
    allocator = factory(model.encoder, args, total) if factory else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    print(json.dumps({"train_docs": len(train), "train_windows": len(examples), "dev_docs": len(dev),
                      "positive_rate": round(float(y.mean()), 3), **info}, default=str), flush=True)
    y_dev = np.array([dev[d]["label"] for d in dev])
    best, history, step = {"score": -1}, [], 0

    def evaluate():
        logits = doc_logits(model, tokenizer, dev, args, device, dtype, args.tail_chunks, args.head_chunk)
        p = np.array([1 / (1 + math.exp(-logits[d])) for d in dev])
        return macro_f1(y_dev, (p >= 0.5).astype(int)), honest_threshold(y_dev, p), logits

    def checkpoint_if_best(epoch, losses):
        nonlocal best
        f05, honest, logits = evaluate()
        row = {"epoch": epoch, "step": step, "loss": round(float(np.mean(losses[-200:] or [0])), 4),
               "dev_macro_f1": round(f05, 4), "dev_macro_f1_honest_threshold": round(honest, 4),
               "minutes": round((time.time() - started) / 60, 1)}
        print(json.dumps(row), flush=True)
        history.append(row)
        if max(f05, honest) > best["score"]:
            best = {"score": max(f05, honest), "row": row, "logits": logits,
                    "state": {k: v.detach().cpu().clone() for k, v in trainable_state(model).items()}}
            save_classifier(out / "model.pt", model, args, {"dev_row": row})
        model.train()

    def optimizer_step():
        nonlocal step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        if allocator is not None:
            allocator.step(step)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(examples)
        losses, pending = [], False
        for b in range(0, len(examples), args.batch_size):
            batch = examples[b:b + args.batch_size]
            ids, lengths = batch_tensors([e[2] for e in batch], pad_id, device)
            positions = torch.tensor([e[1] for e in batch], device=device)
            target = torch.tensor([float(e[3]) for e in batch], device=device)
            with autocast(device, dtype):
                logit, _ = model(ids, lengths, positions)
            logit = logit.float()
            loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
            if teacher_prob is not None:
                soft = torch.tensor([teacher_prob.get(e[0], float(e[3])) for e in batch], device=device)
                kd = F.binary_cross_entropy_with_logits(logit / args.kd_temperature, soft) * args.kd_temperature ** 2
                loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            if allocator is not None and args.adalora_orth:
                loss = loss + args.adalora_orth * allocator.orth_penalty()
            scaler.scale(loss / args.grad_accum).backward()
            losses.append(loss.item())
            pending = True
            if (b // args.batch_size + 1) % args.grad_accum == 0:
                optimizer_step()
                pending = False
                if args.eval_every and step % args.eval_every == 0:
                    checkpoint_if_best(epoch, losses)
        if pending:
            optimizer_step()
        checkpoint_if_best(epoch, losses)                      # every epoch ends with a dev evaluation
    model.load_state_dict(best["state"], strict=False)
    finish(args, model, tokenizer, dev, test, best["logits"], out, device, dtype,
           {"stage": "finetune", "history": history, "best": best["row"], "parameters": info.get("parameters"),
            "encoder_parameters": info.get("encoder_parameters"), "peft": info.get("peft"),
            "teacher": args.teacher, "minutes": round((time.time() - started) / 60, 1)})


def finish(args, model, tokenizer, dev, test, dev_logits, out, device, dtype, summary):
    """Dev threshold (honest check), test probabilities, submission, summary, probs.json."""
    y_dev = np.array([dev[d]["label"] for d in dev])
    p_dev = probs_from_logits(dev_logits)
    pd = np.array([p_dev[d] for d in dev])
    f05 = macro_f1(y_dev, (pd >= 0.5).astype(int))
    threshold, tuned = best_threshold(y_dev, pd)
    honest = honest_threshold(y_dev, pd)
    use = honest > f05 + 0.003
    t = threshold if use else 0.5
    test_logits = doc_logits(model, tokenizer, test, args, device, dtype, args.tail_chunks, args.head_chunk)
    p_test = probs_from_logits(test_logits)
    (out / "probs.json").write_text(json.dumps({"dev": p_dev, "test": p_test}))
    info = write_prediction({d: int(p_test[d] >= t) for d in test}, test, out / "submission", p_test)
    summary.update({"dev_macro_f1_at_0.5": round(f05, 5), "dev_macro_f1_tuned": round(tuned, 5),
                    "dev_macro_f1_honest": round(honest if use else f05, 5), "threshold": round(t, 3), "test": info})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"dev macro-F1 {f05:.4f} @0.5 | honest-threshold {honest:.4f} -> threshold {t:.2f}  "
          f"submission: {out / 'submission' / 'submission.zip'} {info}", flush=True)


# --------------------------------------------------------------------------- stage 2
def embed(args):
    device, dtype = device_and_amp()
    run = Path(args.run)
    model, tokenizer, margs = load_classifier(run / "model.pt", device)
    margs.eval_batch = args.eval_batch
    for split in ("train", "dev", "test"):
        docs = load_split(args.data, split)
        if split == "train" and args.max_train_docs and len(docs) > args.max_train_docs:
            keep = random.Random(0).sample(sorted(docs), args.max_train_docs)
            docs = {d: docs[d] for d in sorted(keep)}
        started = time.time()
        _, vectors = doc_logits(model, tokenizer, docs, margs, device, dtype, args.embed_chunks,
                                margs.head_chunk, return_vectors=True)
        labels = {d: docs[d]["label"] for d in docs}
        torch.save({"vectors": vectors, "labels": labels, "embed_chunks": args.embed_chunks,
                    "head_chunk": margs.head_chunk}, run / f"emb_{split}.pt")
        print(f"embedded {split}: {len(vectors)} documents in {(time.time() - started) / 60:.1f} min", flush=True)
