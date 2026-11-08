"""Continued pretraining of a Phase-1 checkpoint before NER fine-tuning (DAPT + TAPT).

The NER labels are fixed (70 documents), but unlabeled legal text is plentiful: the official
Phase-1 corpus has >4.5B tokens and the Phase-1 models saw 0.5-4 tokens per parameter. This
stage starts FROM a Phase-1 checkpoint and keeps training it on more official text:

  clm  causal language modelling (dense_gpt, gqa_local, recurrent_hybrid) - same objective as
       Phase 1, more tokens, plus the punctuation segmentation the NER runs use
  mlm  masked-token prediction on the ENCODER of the encoder-decoder (15% of tokens; 80% -> <unk>
       as the mask, 10% random, 10% kept), predicted through the tied embedding. Every token is
       supervised with both-side context, which is what a tagger needs. The decoder is unchanged.

Data mixes official Phase-1 documents with the NER *training* documents' text (never the test
text). Output is a normal Phase-1-format checkpoint (best.pt / latest.pt + tokenizer.json), so
`phase2.run cv --checkpoint <adapt dir>/best.pt` uses it directly.

    python -m phase2.adapt prepare --raw SC.json HC.json --weights 0.6 0.4 --ner-data Phase_2_LNER \\
        --tokenizer <phase1 run>/tokenizer.json --tokens 1000000000 --output adapt_data
    python -m phase2.adapt train --checkpoint <phase1 run>/best.pt --data adapt_data --objective mlm \\
        --hours 10 --output adapt/encdec_mlm
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .data import load_texts, segment_tokenizer, write_json

MASK_FRACTION = 0.15


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- data
def _records(path):
    from phase1.data import records  # streams 1+ GB JSON with ijson
    yield from records(path)


def prepare(raw_paths, weights, ner_data, tokenizer_path, output, tokens, segment="punct", val_tokens=2_000_000,
            val_every=200, min_chars=500, batch_docs=256, exclude_ner=True):
    """Tokenize official documents into train.bin/val.bin (+ ner.bin from NER training texts).

    With exclude_ner, corpus documents containing any NER document (training folds or test) are
    skipped, so the NER test text never enters continued pretraining."""
    from tokenizers import Tokenizer
    from .overlap import NerFingerprints
    fingerprints = NerFingerprints.from_data(ner_data) if (exclude_ner and ner_data) else None
    excluded = 0
    output = Path(output)
    if (output / "metadata.json").exists():
        print("Reusing", output)
        return json.loads((output / "metadata.json").read_text())
    stage = output.with_name(output.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    base = Tokenizer.from_file(str(tokenizer_path))
    tokenizer = segment_tokenizer(base, segment)
    eos = tokenizer.token_to_id("<eos>")
    if eos is None or tokenizer.get_vocab_size() > 65535:
        raise ValueError("Expected an <eos> token and a vocabulary that fits uint16")
    weights = [w / sum(weights) for w in weights]
    quotas = [int(tokens * w) for w in weights]
    sources = []
    started = time.time()
    with (stage / "train.bin").open("wb") as train_handle, (stage / "val.bin").open("wb") as val_handle:
        val_used = 0
        for path, quota in zip(raw_paths, quotas):
            used, docs, batch = 0, 0, []

            def flush(batch):
                nonlocal used, val_used
                for i, encoding in enumerate(tokenizer.encode_batch(batch)):
                    ids = np.asarray(encoding.ids + [eos], dtype=np.uint16)
                    if val_used < val_tokens and (docs - len(batch) + i) % val_every == 0:
                        ids.tofile(val_handle)
                        val_used += len(ids)
                    elif used < quota:
                        ids[:quota - used].tofile(train_handle)
                        used += min(len(ids), quota - used)

            for record in _records(path):
                text = record.get("text", "") if isinstance(record, dict) else str(record)
                if len(text) < min_chars:
                    continue
                if fingerprints is not None and fingerprints.overlaps(text):
                    excluded += 1
                    continue
                batch.append(text)
                docs += 1
                if len(batch) == batch_docs:
                    flush(batch)
                    batch = []
                    if used >= quota and val_used >= val_tokens:
                        break
                    if docs % (batch_docs * 40) == 0:
                        rate = used / max(1e-9, time.time() - started)
                        print(f"{Path(path).name}: {used / 1e6:,.0f}M / {quota / 1e6:,.0f}M tokens "
                              f"({rate / 1e3:,.0f}k tok/s)", flush=True)
            if batch and (used < quota or val_used < val_tokens):
                flush(batch)
            sources.append({"path": str(path), "quota": quota, "tokens": used, "documents": docs})
            if used < quota:
                print(f"NOTE: {Path(path).name} ran out at {used / 1e6:,.0f}M of {quota / 1e6:,.0f}M tokens")
    ner_tokens = 0
    if ner_data:
        texts = {}
        for fold in (1, 2):   # training folds only: the test documents' text is never used
            texts.update(load_texts(Path(ner_data) / "Train" / f"fold{fold}-text.json"))
        with (stage / "ner.bin").open("wb") as handle:
            for encoding in tokenizer.encode_batch(list(texts.values())):
                ids = np.asarray(encoding.ids + [eos], dtype=np.uint16)
                ids.tofile(handle)
                ner_tokens += len(ids)
    shutil.copyfile(tokenizer_path, stage / "tokenizer.json")
    metadata = {"segment": segment, "dtype": "uint16", "tokenizer_sha256": file_sha256(tokenizer_path),
                "train_tokens": sum(s["tokens"] for s in sources), "val_tokens": val_used, "ner_tokens": ner_tokens,
                "sources": sources, "excluded_ner_overlap_documents": excluded,
                "minutes": round((time.time() - started) / 60, 1)}
    write_json(stage / "metadata.json", metadata)
    shutil.rmtree(output, ignore_errors=True)
    stage.rename(output)
    print(json.dumps(metadata, indent=1))
    return metadata


class Stream:
    """Random fixed-length blocks from train.bin, with a share drawn from ner.bin (TAPT)."""
    def __init__(self, directory, split="train", ner_fraction=0.0):
        directory = Path(directory)
        self.main = np.memmap(directory / f"{split}.bin", dtype=np.uint16, mode="r")
        ner = directory / "ner.bin"
        self.ner = np.memmap(ner, dtype=np.uint16, mode="r") if ner_fraction > 0 and ner.exists() else None
        self.ner_fraction = ner_fraction if self.ner is not None else 0.0

    def batch(self, rng, size, length):
        rows = []
        for _ in range(size):
            source = self.ner if self.ner_fraction and rng.random() < self.ner_fraction else self.main
            if len(source) <= length:
                raise ValueError(f"Token file shorter than one block ({len(source)} <= {length})")
            start = int(rng.integers(0, len(source) - length))
            rows.append(source[start:start + length].astype(np.int64))
        return np.stack(rows)


# --------------------------------------------------------------------------- objectives
def mask_tokens(ids, mask_id, vocab_size, special_ids, generator, fraction=MASK_FRACTION):
    """BERT masking. Returns (corrupted ids, labels with -100 where not predicted)."""
    import torch
    probability = torch.full(ids.shape, fraction, device=ids.device)
    special = torch.zeros_like(ids, dtype=torch.bool)
    for token in special_ids:
        special |= ids == token
    chosen = torch.bernoulli(probability, generator=generator).bool() & ~special
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    corrupted = ids.clone()
    roll = torch.rand(ids.shape, device=ids.device, generator=generator)
    corrupted[chosen & (roll < 0.8)] = mask_id
    swap = chosen & (roll >= 0.8) & (roll < 0.9)
    corrupted[swap] = torch.randint(len(special_ids), vocab_size, (int(swap.sum()),), device=ids.device,
                                    generator=generator)
    return corrupted, labels


def objective_loss(model, objective, batch, mask_id, special_ids, generator):
    import torch
    from torch.nn import functional as F
    if objective == "clm":
        logits, _ = model(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1))
    corrupted, labels = mask_tokens(batch, mask_id, model.embedding.num_embeddings, special_ids, generator)
    hidden = model.encode(corrupted)
    logits = F.linear(hidden, model.embedding.weight)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)


def wsd(step, total, warmup, decay_fraction=0.2, floor=0.1):
    """Warmup, constant, then linear decay over the last `decay_fraction` of steps."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_fraction))
    if step < decay_start:
        return 1.0
    return max(floor, 1 - (1 - floor) * (step - decay_start) / max(1, total - decay_start))


# --------------------------------------------------------------------------- training
def train(checkpoint, data, output, objective, hours, project_root=None, seq_len=512, micro_batch=16,
          accumulation=4, lr=1e-4, warmup=200, eval_every=500, eval_batches=20, ner_fraction=0.02, seed=42,
          max_steps=None, resume=False):
    import torch
    from contextlib import nullcontext
    from .model import load_phase1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = Path(data)
    meta = json.loads((data / "metadata.json").read_text())
    tokenizer_path = Path(checkpoint).parent / "tokenizer.json"
    if file_sha256(tokenizer_path) != meta["tokenizer_sha256"]:
        raise ValueError("The adaptation data was tokenized with a different tokenizer than this checkpoint")
    model, tokenizer, resolved = load_phase1(checkpoint, project_root)
    architecture = model.config.architecture
    if objective == "mlm" and architecture != "encoder_decoder":
        raise ValueError("mlm adapts the encoder of the encoder_decoder checkpoint; use clm for decoder-only models")
    if objective == "clm" and architecture == "encoder_decoder":
        raise ValueError("Use --objective mlm for the encoder_decoder checkpoint (NER uses its encoder)")
    model.config.gradient_checkpointing = False
    if seq_len + (objective == "clm") > model.config.max_context:
        raise ValueError(f"--seq-len {seq_len} exceeds the model context {model.config.max_context}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device.type == "cuda" else None)
    autocast = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype is not None else nullcontext
    model.to(device).train()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                                  lr=lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    mask_id = tokenizer.token_to_id("<unk>")   # byte-level BPE never emits <unk>: a free mask token
    special_ids = [tokenizer.token_to_id(t) for t in ["<pad>", "<bos>", "<eos>", "<unk>"]]
    generator = torch.Generator(device=device).manual_seed(seed)
    block = seq_len + (objective == "clm")
    train_stream = Stream(data, "train", ner_fraction)
    val_stream = Stream(data, "val")
    tokens_per_step = micro_batch * accumulation * seq_len
    step, total, best, elapsed = 0, max_steps or 10**9, math.inf, 0.0
    if resume and (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, total, best, elapsed = state["step"], state["total_steps"], state["best_loss"], state["train_seconds"]
        rng.bit_generator.state = state["numpy_rng"]
        print(f"resumed at step {step}/{total}", flush=True)

    @torch.no_grad()
    def validate():
        model.eval()
        val_rng = np.random.default_rng(seed + 1)
        val_generator = torch.Generator(device=device).manual_seed(seed + 1)
        losses = []
        for _ in range(eval_batches):
            batch = torch.from_numpy(val_stream.batch(val_rng, micro_batch, block)).to(device)
            with autocast():
                losses.append(float(objective_loss(model, objective, batch, mask_id, special_ids, val_generator)))
        model.train()
        return float(np.mean(losses))

    adapted = dict(resolved)
    adapted["run_id"] = f"{resolved.get('run_id', architecture)}+{objective}"
    adapted["adaptation"] = {"parent": str(Path(checkpoint).resolve()), "parent_step": resolved.get("step"),
                             "objective": objective, "data": str(data.resolve()), "segment": meta["segment"],
                             "seq_len": seq_len, "tokens_per_step": tokens_per_step, "lr": lr,
                             "ner_fraction": ner_fraction, "hours": hours}
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")

    def save(name, loss):
        torch.save({"model": model.state_dict(), "resolved": adapted, "step": step, "validation_loss": loss},
                   output / name)

    start_loss = validate()
    print(json.dumps({"architecture": architecture, "objective": objective, "start_val_loss": round(start_loss, 4),
                      "tokens_per_step": tokens_per_step, "precision": str(dtype)}), flush=True)
    log = (output / "adapt_metrics.jsonl").open("a")
    session = time.time()
    while step < total:
        for group in optimizer.param_groups:
            group["lr"] = lr * wsd(step, total, warmup)
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accumulation):
            batch = torch.from_numpy(train_stream.batch(rng, micro_batch, block)).to(device, non_blocking=True)
            with autocast():
                loss = objective_loss(model, objective, batch, mask_id, special_ids, generator)
            scaler.scale(loss / accumulation).backward()
            running += float(loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        elapsed += time.time() - tick
        if max_steps is None and step == 30 and not resume:
            per_step = elapsed / step   # size the schedule to the wall-clock budget
            total = max(60, int(hours * 3600 * 0.93 / per_step))
            print(json.dumps({"seconds_per_step": round(per_step, 3), "planned_steps": total,
                              "planned_tokens": total * tokens_per_step,
                              "tokens_per_second": round(tokens_per_step / per_step)}), flush=True)
        if step % 50 == 0 or step == total:
            row = {"step": step, "loss": round(running, 4), "lr": optimizer.param_groups[0]["lr"],
                   "tokens": step * tokens_per_step, "hours": round(elapsed / 3600, 3)}
            if step % eval_every == 0 or step == total:
                row["val_loss"] = round(validate(), 4)
                save("latest.pt", row["val_loss"])
                if row["val_loss"] < best:
                    best = row["val_loss"]
                    save("best.pt", best)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "total_steps": total, "best_loss": best, "train_seconds": elapsed,
                            "numpy_rng": rng.bit_generator.state}, output / "last.pt")
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    final = validate()
    save("latest.pt", final)
    if final < best:
        best = final
        save("best.pt", best)
    summary = {"start_val_loss": start_loss, "final_val_loss": final, "best_val_loss": best, "steps": step,
               "tokens": step * tokens_per_step, "train_hours": round(elapsed / 3600, 2),
               "session_hours": round((time.time() - session) / 3600, 2)}
    write_json(output / "adapt_summary.json", summary)
    (output / "last.pt").unlink(missing_ok=True)   # optimizer state no longer needed
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", nargs="+", required=True, help="official Phase-1 train_*.json files")
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--ner-data", help="Phase_2_LNER directory (its training folds' text is added; never the test)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--tokens", type=int, default=1_000_000_000)
    p.add_argument("--segment", default="punct", choices=["punct", "bpe"])
    p.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--objective", required=True, choices=["clm", "mlm"])
    t.add_argument("--hours", type=float, default=10)
    t.add_argument("--project-root")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--micro-batch", type=int, default=16)
    t.add_argument("--accumulation", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--ner-fraction", type=float, default=0.02)
    t.add_argument("--max-steps", type=int, help="fixed step count instead of the --hours budget (tests)")
    t.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        weights = args.weights or [1.0] * len(args.raw)
        if len(weights) != len(args.raw):
            parser.error("one weight per --raw file")
        prepare(args.raw, weights, args.ner_data, args.tokenizer, args.output, args.tokens, args.segment)
    else:
        train(args.checkpoint, args.data, args.output, args.objective, args.hours, args.project_root, args.seq_len,
              args.micro_batch, args.accumulation, args.lr, args.warmup, args.eval_every, ner_fraction=args.ner_fraction,
              max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
