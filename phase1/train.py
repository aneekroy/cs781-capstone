"""Bounded single-GPU training with mixed precision and exact-state resume.

Training samples blocks with replacement. `token_budget` counts tokens
processed, not a promise to cover every unique corpus token once.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import file_hash, fingerprint, read_json, seed_everything, write_json, quarantine_directory
from .data import TokenCorpus
from .models import ModelConfig, build_model, parameter_count


def precision(device, requested="auto"):
    if device.type != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError("CPU smoke tests require fp32")
        return "fp32", None
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    choice = ("bf16" if native_bf16 else "fp16") if requested == "auto" else requested
    if choice == "bf16" and not native_bf16:
        raise ValueError("bf16 is emulated on this GPU; use fp16")
    return choice, {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[choice]


def amp_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def loss_for_block(model, tokens, bos_id, source_fraction=0.75, cut=None):
    if model.config.architecture == "encoder_decoder":
        cut = int(tokens.size(1) * source_fraction) if cut is None else cut
        if not 1 <= cut < tokens.size(1):
            raise ValueError("source_fraction must leave nonempty encoder and decoder sequences")
        source, labels = tokens[:, :cut], tokens[:, cut:]
        start = torch.full_like(labels[:, :1], bos_id)
        inputs = torch.cat((start, labels[:, :-1]), dim=1)
        logits, _ = model(inputs, source_ids=source)
    else:
        logits, _ = model(tokens[:, :-1])
        labels = tokens[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
    return loss, labels.numel()


def source_cut(settings, length, rng):
    span = settings.get("source_length_range")
    if span is None:
        fractions = settings.get("source_fraction_range")
        if fractions is None:
            return int(length * settings["source_fraction"])
        span = [int(length * fractions[0]), int(length * fractions[1])]
    low, high = max(1, int(span[0])), min(length-1, int(span[1]))
    if high < low:
        raise ValueError("Encoder source length range leaves no decoder targets")
    return int(rng.integers(low, high+1))


@torch.no_grad()
def evaluate(model, corpus, settings, bos_id, device, dtype):
    model.eval()
    rng = np.random.default_rng(settings["seed"] + 1000)
    weighted_loss, targets = 0.0, 0
    length = settings["sequence_length"] + (model.config.architecture != "encoder_decoder")
    for _ in range(settings["eval_batches"]):
        tokens = torch.from_numpy(corpus.sample(rng, settings["micro_batch_size"], length)).to(device)
        with amp_context(device, dtype):
            cut = source_cut(settings, length, rng) if model.config.architecture == "encoder_decoder" else None
            loss, count = loss_for_block(model, tokens, bos_id, settings["source_fraction"], cut)
        weighted_loss += loss.item() * count
        targets += count
    model.train()
    mean = weighted_loss / targets
    if not math.isfinite(mean):
        raise FloatingPointError("Nonfinite validation loss")
    return {"validation_loss": mean, "perplexity": math.exp(min(mean, 700)),
            "validation_target_tokens": targets}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def environment():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"python": sys.version, "platform": platform.platform(), "git_revision": revision,
            "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "tokenizers", "ijson", "huggingface_hub"]},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "command": sys.argv}


def source_digest(training_only=False):
    root = Path(__file__).parent
    names = {"train.py", "models.py", "data.py", "common.py"}
    return fingerprint({path.name: file_hash(path) for path in sorted(root.glob("*.py"))
                        if not training_only or path.name in names})


def check_resume(previous, current, allow_code_change=False):
    signature = previous.get("resume_signature")
    if signature is None:
        # Pre-patch checkpoints can be reviewed explicitly without pretending
        # they used the new source cuts/context settings.
        legacy = {key: previous[key] for key in
                  ["run_id", "model", "training", "data_fingerprint", "tokenizer_sha256"]}
        legacy["training"] = {"source_fraction_range": None, "source_length_range": None,
                              "long_sequence_length": None, "long_sequence_probability": 0.0,
                              "require_corpus_budget": False, **legacy["training"]}
        signature = fingerprint(legacy)
    if signature != current["resume_signature"]:
        raise ValueError("Resume model, training settings or data changed; use a new run_id/output path")
    changed = previous.get("training_code_fingerprint") != current["training_code_fingerprint"]
    if changed and not allow_code_change:
        raise ValueError("Training code changed; use a new run or explicitly --allow-code-change after review")
    return changed


def resolve(config_file, data_dir):
    config = read_json(config_file)
    metadata = read_json(Path(data_dir) / "metadata.json")
    model_config = dict(config["model"])
    model_config["vocab_size"] = metadata["vocab_size"]
    model_config = asdict(ModelConfig(**model_config))
    settings = dict(config["training"])
    defaults = {"seed": 42, "sequence_length": 1024, "micro_batch_size": 1,
                "gradient_accumulation": 16, "token_budget": 20_000_000,
                "learning_rate": 3e-4, "min_lr_ratio": 0.1, "warmup_fraction": 0.03,
                "weight_decay": 0.1, "grad_clip": 1.0, "eval_every": 250,
                "eval_batches": 16, "source_fraction": 0.75, "precision": "auto",
                "source_fraction_range": None, "source_length_range": None,
                "long_sequence_length": None, "long_sequence_probability": 0.0,
                "require_corpus_budget": False}
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    settings = defaults | settings
    if min(settings[key] for key in ["sequence_length", "micro_batch_size", "gradient_accumulation", "token_budget", "eval_every", "eval_batches"]) <= 0:
        raise ValueError("Training budgets/batches/intervals must be positive")
    if not 0 < settings["source_fraction"] < 1 or not 0 <= settings["warmup_fraction"] < 1:
        raise ValueError("Invalid source/warmup fraction")
    if settings["sequence_length"] > model_config["max_context"]:
        raise ValueError("Training sequence exceeds model max_context")
    tokens_per_step = settings["sequence_length"] * settings["micro_batch_size"] * settings["gradient_accumulation"]
    if not 0 <= settings["long_sequence_probability"] <= 1:
        raise ValueError("Invalid long context probability")
    long_length = settings["long_sequence_length"]
    if long_length is not None and (not settings["sequence_length"] <= long_length <= model_config["max_context"]
                                    or tokens_per_step % (long_length * settings["micro_batch_size"])):
        raise ValueError("Long context must fit model and divide the effective tokens per update")
    for key in ["source_fraction_range", "source_length_range"]:
        span = settings[key]
        if span is not None and (len(span) != 2 or not 0 < span[0] <= span[1] or
                                 (key == "source_fraction_range" and span[1] >= 1)):
            raise ValueError(f"Invalid {key}")
    stored_train_tokens = metadata["counts"]["train"]
    if settings["require_corpus_budget"] and stored_train_tokens < settings["token_budget"]:
        raise ValueError("Prepared corpus is smaller than token budget; prepare more data or explicitly permit resampling")
    total_steps = math.ceil(settings["token_budget"] / tokens_per_step)
    resolved = {"run_id": config["run_id"], "model": model_config, "training": settings,
                "data_directory": str(Path(data_dir).resolve()), "data_fingerprint": fingerprint(metadata),
                "tokenizer_sha256": metadata["tokenizer_sha256"],
                "code_fingerprint": source_digest(), "total_steps": total_steps,
                "training_code_fingerprint": source_digest(True),
                "stored_train_tokens": stored_train_tokens,
                "processed_to_stored_token_ratio": total_steps*tokens_per_step / stored_train_tokens,
                "tokens_per_step": tokens_per_step, "parameters": parameter_count(model_config)}
    resolved["resume_signature"] = fingerprint({key: resolved[key] for key in
        ["run_id", "model", "training", "data_fingerprint", "tokenizer_sha256"]})
    resolved["run_fingerprint"] = fingerprint(resolved)
    return resolved


def train(config_file, data_dir, output_dir, resume=False, stop_after_step=None,
          max_hours=None, allow_small_model=False, dry_run=False, allow_code_change=False):
    resolved = resolve(config_file, data_dir)
    size = resolved["parameters"]
    if not allow_small_model and not 50_000_000 <= size <= 300_000_000:
        raise ValueError(f"Competition model must have 50M–300M parameters, got {size:,}")
    if size > 300_000_000:
        raise ValueError("Models larger than 300M are never allowed")
    if dry_run:
        print(json.dumps(resolved, indent=2))
        return resolved
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not (output / "last.pt").exists():
        if (output / "status.json").exists() and read_json(output / "status.json").get("status") == "completed":
            raise FileExistsError('Completed model-only run has no optimizer checkpoint; retain it and choose a new output path')
        quarantine_directory(output, "Run has no resumable checkpoint; restarting from initialization")
        resume = False
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Run exists: {output}; use --resume or a new output path")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = resolved["training"]
    seed_everything(settings["seed"])
    dtype_name, dtype = precision(device, settings["precision"])
    train_data, val_data = TokenCorpus(data_dir), TokenCorpus(data_dir, "validation")
    # Verify actual token bytes before committing a long run or resuming one.
    for split in ["train", "validation"]:
        expected = train_data.metadata[f"{split}_sha256"]
        if file_hash(Path(data_dir) / f"{split}.bin") != expected:
            raise ValueError(f"{split} token fingerprint changed")
    from tokenizers import Tokenizer
    tokenizer_path = Path(data_dir) / "tokenizer.json"
    if file_hash(tokenizer_path) != train_data.metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint changed")
    bos_id = Tokenizer.from_file(str(tokenizer_path)).token_to_id("<bos>")
    model = build_model(resolved["model"]).to(device)
    assert sum(p.numel() for p in model.parameters()) == size
    decayed = [p for p in model.parameters() if p.ndim >= 2]
    nondecayed = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decayed, "weight_decay": settings["weight_decay"]},
                                  {"params": nondecayed, "weight_decay": 0.0}],
                                 lr=settings["learning_rate"], betas=(0.9, 0.95), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype_name == "fp16", init_scale=1024.0)
    rng = np.random.default_rng(settings["seed"])
    first_step, best_loss, cumulative_train_seconds, targets_seen = 0, math.inf, 0.0, 0
    previous_wall_seconds = 0.0
    if resume:
        path = output / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint absent: {path}")
        state = torch.load(path, map_location=device, weights_only=False)
        code_changed = check_resume(state["resolved"], resolved, allow_code_change)
        if code_changed:
            write_json(output / "resume_code_change.json", {"previous": state["resolved"]["training_code_fingerprint"],
                       "current": resolved["training_code_fingerprint"], "step": state["step"],
                       "authorized_by": "--allow-code-change; controlled equivalence is not claimed"})
        if state.get("precision") != dtype_name or state.get("device_type") != device.type:
            raise ValueError("Resume precision/backend changed; resume on the same backend and precision")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_step, best_loss = state["step"], state["best_loss"]
        targets_seen, cumulative_train_seconds = state["targets_seen"], state["train_seconds"]
        previous_wall_seconds = state.get("wall_seconds", cumulative_train_seconds)
        rng.bit_generator.state = state["numpy_generator"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        if first_step >= resolved["total_steps"]:
            raise ValueError("Run already completed; use a new output path for another experiment")
    write_json(output / "resolved_config.json", resolved)
    if not resume:
        shutil.copyfile(tokenizer_path, output / "tokenizer.json")
        write_json(output / "environment.json", environment())
    write_json(output / "status.json", {"status": "running", "step": first_step,
                                       "started_utc": datetime.now(timezone.utc).isoformat()})
    session_start = time.perf_counter()
    start_train_seconds = cumulative_train_seconds
    model.train()
    step = first_step
    try:
        for step in range(first_step + 1, resolved["total_steps"] + 1):
            warmup = max(1, int(resolved["total_steps"] * settings["warmup_fraction"]))
            if step <= warmup:
                multiplier = step / warmup
            else:
                progress = (step-warmup) / max(1, resolved["total_steps"]-warmup)
                multiplier = settings["min_lr_ratio"] + (1-settings["min_lr_ratio"]) * (1+math.cos(math.pi*progress))/2
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rate"] * multiplier
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss, target_count = torch.zeros((), device=device), 0
            sequence_length = settings["sequence_length"]
            if settings["long_sequence_length"] and rng.random() < settings["long_sequence_probability"]:
                sequence_length = settings["long_sequence_length"]
            accumulation = resolved["tokens_per_step"] // (sequence_length * settings["micro_batch_size"])
            length = sequence_length + (model.config.architecture != "encoder_decoder")
            for _ in range(accumulation):
                tokens = torch.from_numpy(train_data.sample(rng, settings["micro_batch_size"], length)).to(device)
                with amp_context(device, dtype):
                    cut = source_cut(settings, length, rng) if model.config.architecture == "encoder_decoder" else None
                    loss, count = loss_for_block(model, tokens, bos_id, settings["source_fraction"], cut)
                total_loss += loss.detach()
                target_count += count
                scaler.scale(loss / accumulation).backward()
            mean_loss = float(total_loss / accumulation)
            if not math.isfinite(mean_loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            finite_gradients = math.isfinite(float(grad_norm))
            if not finite_gradients and dtype_name != "fp16":
                raise FloatingPointError("Nonfinite gradient norm; retry from checkpoint with bf16/fp32 or lower LR")
            # fp16 overflow is handled by GradScaler's skipped update and scale
            # reduction. Tokens consumed by a skipped update remain logged.
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            cumulative_train_seconds += time.perf_counter() - tick
            targets_seen += target_count
            stop = ((stop_after_step is not None and step >= stop_after_step) or
                    (max_hours is not None and time.perf_counter()-session_start >= max_hours*3600))
            row = {"step": step, "input_tokens_seen": step * resolved["tokens_per_step"],
                   "target_tokens_seen": targets_seen, "train_loss": mean_loss,
                   "sequence_length": sequence_length, "gradient_accumulation": accumulation,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_updated": finite_gradients,
                   "train_tokens_per_second": (step-first_step)*resolved["tokens_per_step"] /
                   max(cumulative_train_seconds-start_train_seconds, 1e-9),
                   "train_seconds": cumulative_train_seconds,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            save = step == first_step+1 or step % settings["eval_every"] == 0 or step == resolved["total_steps"] or stop
            if save:
                row.update(evaluate(model, val_data, settings, bos_id, device, dtype))
                atomic_torch_save({"model": model.state_dict(), "resolved": resolved,
                                   "step": step, "validation_loss": row["validation_loss"]}, output / "latest.pt")
                if row["validation_loss"] < best_loss:
                    best_loss = row["validation_loss"]
                    atomic_torch_save({"model": model.state_dict(), "resolved": resolved,
                                       "step": step, "validation_loss": best_loss}, output / "best.pt")
                atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scaler": scaler.state_dict(), "resolved": resolved, "step": step,
                                   "precision": dtype_name, "device_type": device.type,
                                   "best_loss": best_loss, "targets_seen": targets_seen,
                                   "train_seconds": cumulative_train_seconds,
                                   "wall_seconds": previous_wall_seconds+time.perf_counter()-session_start,
                                   "numpy_generator": rng.bit_generator.state, "python_rng": random.getstate(),
                                   "torch_rng": torch.get_rng_state(),
                                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, output / "last.pt")
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if save or step % 10 == 0:
                print(row, flush=True)
            if stop:
                break
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "step": step, "error": str(error),
                                           "resume_checkpoint": str(output / "last.pt")})
        raise
    result = {"status": "completed" if step == resolved["total_steps"] else "paused",
              "run_id": resolved["run_id"], "step": step, "best_validation_loss": best_loss,
              "best_checkpoint": str((output / "best.pt").resolve()),
              "input_tokens_seen": step*resolved["tokens_per_step"], "target_tokens_seen": targets_seen,
              "train_seconds": cumulative_train_seconds, "session_seconds": time.perf_counter()-session_start,
              "wall_seconds": previous_wall_seconds+time.perf_counter()-session_start,
              "precision": dtype_name}
    write_json(output / "status.json", result)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-code-change", action="store_true", help="Audit opt-in; settings/data/model must still match")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--allow-small-model", action="store_true", help="Only for non-submittable smoke tests")
    args = vars(parser.parse_args())
    args["config_file"], args["data_dir"], args["output_dir"] = args.pop("config"), args.pop("data"), args.pop("output")
    train(**args)


if __name__ == "__main__":
    main()
