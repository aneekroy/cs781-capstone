"""CS781 Phase 4 (CJPE: binary judgment prediction) command line - one subcommand per pipeline stage.

  stats      data checks: sizes, label balance, lengths, train/dev vs test near-duplicates
  baseline   stage 0: TF-IDF + logistic regression on head + tail of each judgment
  finetune   stage 1: Phase-1/2 backbone + PEFT (LoRMA default) on the last windows of each judgment (+KD)
  embed      stage 2: window embeddings of every document with a stage-1 model
  aggregate  stage 3: BiGRU + attention over the window embeddings, several seeds
  check      validate a prediction.json against the test file
(stage 4, the ensemble: python -m phase4.ensemble)
"""
import argparse
import json
from pathlib import Path

import numpy as np


def cmd_stats(args):
    from .data import load_split, near_duplicates, test_overlap
    raw = {s: load_split(args.data, s, exclude_test=False) for s in ("train", "dev")}
    test_ids = set(load_split(args.data, "test", verbose=False))
    for s, docs in raw.items():
        ov = test_overlap(args.data, docs)
        n_exact = sum(1 for k, _ in ov.values() if k == "exact")
        covered = {t for _, t in ov.values()}
        print(f"{s}: {len(ov)} of {len(docs)} documents duplicate a test judgment ({n_exact} exact, {len(ov) - n_exact} near); "
              f"they cover {len(covered)} of {len(test_ids)} test judgments; ids shared with test: {len(set(docs) & test_ids)}",
              flush=True)
    print("training / selection below uses the splits WITHOUT those copies:", flush=True)
    splits = {s: load_split(args.data, s) for s in ("train", "dev", "test")}
    report = {}
    for s, docs in splits.items():
        words = [len(d["text"].split()) for d in docs.values()]
        row = {"documents": len(docs), "words_median": int(np.median(words)) if words else 0,
               "words_p90": int(np.percentile(words, 90)) if words else 0, "words_max": max(words, default=0)}
        if s != "test":
            row["accepted_rate"] = round(float(np.mean([d["label"] for d in docs.values()])), 3)
        report[s] = row
    print(json.dumps(report, indent=1))
    sample = dict(list(splits["train"].items())[: args.dup_sample])
    dups = near_duplicates(splits["test"], {**sample, **splits["dev"]})
    print(f"test documents sharing >=50% of sampled 8-grams with a train/dev document: {len(dups)} "
          f"(train sample {len(sample)}) {dups[:5]}")
    print(f"train/dev overlap of ids: {len(set(splits['train']) & set(splits['dev']))}, "
          f"train/test: {len(set(splits['train']) & set(splits['test']))}, dev/test: {len(set(splits['dev']) & set(splits['test']))}")


def cmd_check(args):
    from .data import load_split, validate_prediction
    test = load_split(args.data, "test")
    pred = json.loads(Path(args.prediction).read_text())
    pred = [{"id": str(p["id"]), "pred": p["pred"]} for p in pred] if isinstance(pred, list) else pred
    validate_prediction(pred, test)
    ones = sum(p["pred"] for p in pred) if isinstance(pred, list) else sum(pred.values())
    print(f"OK: {len(test)} documents, {ones} predicted accepted")


def add_backbone_args(p):
    p.add_argument("--data", required=True, help="Phase_4_CJPE directory (Train/, Test/)")
    p.add_argument("--checkpoint", help="Phase-1 best.pt (tokenizer.json alongside)")
    p.add_argument("--init-phase2", help="start from a Phase-2 NER model.pt (adapters merged)")
    p.add_argument("--project-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="lorma", choices=["lorma", "lora", "rslora", "qlora", "adalora", "prefix",
                                                        "prompt", "ia3", "adapter", "normfit", "partial", "full"])
    p.add_argument("--mode", default="causal", choices=["causal", "bidirectional", "encoder"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--keep-layers", type=int)
    p.add_argument("--prompt-tokens", type=int, default=20)
    p.add_argument("--prefix-length", type=int, default=16)
    p.add_argument("--prefix-hidden", type=int, default=512)
    p.add_argument("--adalora-init-r", type=int, default=12)
    p.add_argument("--adalora-target-r", type=int, default=8)
    p.add_argument("--adalora-orth", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--chunk", type=int, default=512, help="tokens per window")
    p.add_argument("--tail-chunks", type=int, default=4, help="windows from the end of each judgment (training)")
    p.add_argument("--head-chunk", action="store_true", help="also the first window (appeal type, parties)")
    p.add_argument("--embed-chunks", type=int, default=16, help="windows per document for stage 2/3")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=0, help="optimizer steps between dev evaluations (0: per epoch)")
    p.add_argument("--max-train-docs", type=int, default=0, help="subsample training documents (0: all)")
    p.add_argument("--lr-peft", type=float, default=None)
    p.add_argument("--lr-full", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--balance", action="store_true", help="positive-class weight = negatives / positives")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher", help="stage-1 run directory to distil from (KD)")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    s.add_argument("--dup-sample", type=int, default=5000)
    b = sub.add_parser("baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--head-words", type=int, default=300)
    b.add_argument("--tail-words", type=int, default=2000)
    b.add_argument("--max-features", type=int, default=300000)
    b.add_argument("--c-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    b.add_argument("--max-train-docs", type=int, default=0)
    add_backbone_args(sub.add_parser("finetune"))
    e = sub.add_parser("embed")
    e.add_argument("--data", required=True)
    e.add_argument("--run", required=True, help="stage-1 run directory")
    e.add_argument("--embed-chunks", type=int, default=16)
    e.add_argument("--eval-batch", type=int, default=32)
    e.add_argument("--max-train-docs", type=int, default=0)
    a = sub.add_parser("aggregate")
    a.add_argument("--data", required=True)
    a.add_argument("--run", required=True, help="stage-1 run directory with emb_*.pt")
    a.add_argument("--output", required=True)
    a.add_argument("--hidden", type=int, default=256)
    a.add_argument("--dropout", type=float, default=0.2)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--patience", type=int, default=4)
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--seeds", type=int, default=5)
    a.add_argument("--balance", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--prediction", required=True)
    for sp in (sub.choices["finetune"], e):
        sp.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"],
                        help="FlashAttention-2 if flash-attn is installed, else SDPA with expanded KV (phase3/flash.py)")
    args = parser.parse_args()
    if args.command in ("finetune", "embed"):
        from phase3.flash import enable
        print(enable(args.attention), flush=True)
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "baseline":
        from .baseline import run
        run(args)
    elif args.command == "finetune":
        if not args.checkpoint and not args.init_phase2:
            parser.error("--checkpoint or --init-phase2 is required")
        args.window = args.chunk                       # phase3.model.build_encoder sizes prompts against it
        from .neural import finetune
        finetune(args)
    elif args.command == "embed":
        from .neural import embed
        embed(args)
    elif args.command == "aggregate":
        from .aggregate import run
        run(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
