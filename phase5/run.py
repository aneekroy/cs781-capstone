"""CS781 Phase 5 (Legal Statute Identification, multi-label) command line.

  stats      sizes, statute frequencies, labels per document, document lengths, train/dev copies of test documents
  baseline   TF-IDF + one-vs-rest logistic regression (CPU)
  finetune   Phase-1 backbone + any PEFT technique, all windows of the facts, label-wise attention (+KD)
  check      validate a prediction.json against the test set and the label space
The ensemble is `python -m phase5.ensemble`.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def cmd_stats(args):
    from .data import label_space, load_split, test_overlap
    raw = {s: load_split(args.data, s, exclude_test=False) for s in ("train", "dev")}
    test = load_split(args.data, "test")
    for s, docs in raw.items():
        ov = test_overlap(args.data, docs)
        exact = sum(1 for k, _ in ov.values() if k == "exact")
        print(f"{s}: {len(ov)} of {len(docs)} documents duplicate a test document ({exact} exact); they cover "
              f"{len({t for _, t in ov.values()})} of {len(test)} test documents; ids shared with test: "
              f"{len(set(docs) & set(test))}", flush=True)
    print("training / selection uses the splits without those copies:", flush=True)
    train, dev = load_split(args.data, "train"), load_split(args.data, "dev")
    labels = label_space(train, dev)
    report = {"statutes": len(labels), "statute_ids": labels[:10] + (["..."] if len(labels) > 10 else [])}
    for s, docs in (("train", train), ("dev", dev), ("test", test)):
        words = [len(d["text"].split()) for d in docs.values()]
        row = {"documents": len(docs), "words_median": int(np.median(words)) if words else 0,
               "words_p90": int(np.percentile(words, 90)) if words else 0}
        if s != "test":
            n = [len(d["labels"]) for d in docs.values()]
            row.update({"labels_per_doc": round(float(np.mean(n)), 2), "labels_per_doc_max": max(n)})
        report[s] = row
    freq = Counter(l for d in train.values() for l in d["labels"])
    dev_freq = Counter(l for d in dev.values() for l in d["labels"])
    counts = sorted(freq.values())
    report["train_statute_frequency"] = {"min": counts[0], "p25": int(np.percentile(counts, 25)),
                                         "median": int(np.median(counts)), "max": counts[-1]}
    report["statutes_with_<20_dev_positives"] = sum(1 for l in labels if dev_freq.get(l, 0) < 20)
    report["statutes_missing_from_dev"] = [l for l in labels if dev_freq.get(l, 0) == 0]
    report["statutes_missing_from_train"] = [l for l in labels if freq.get(l, 0) == 0]
    print(json.dumps(report, indent=1))


def cmd_check(args):
    from .data import label_space, load_split, validate_prediction
    test = load_split(args.data, "test")
    labels = label_space(load_split(args.data, "train", verbose=False), load_split(args.data, "dev", verbose=False))
    pred = json.loads(Path(args.prediction).read_text())
    validate_prediction(pred, test, labels)
    n = [len(p["pred"]) for p in pred]
    print(f"OK: {len(pred)} documents, {np.mean(n):.2f} statutes per document, {sum(1 for x in n if x == 0)} empty")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--max-features", type=int, default=300000)
    b.add_argument("--max-words", type=int, default=6000)
    b.add_argument("--c-grid", type=float, nargs="+", default=[1.0, 4.0])
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--max-train-docs", type=int, default=0)
    from .neural import add_args
    add_args(sub.add_parser("finetune"))
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--prediction", required=True)
    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "baseline":
        from .baseline import run
        run(args)
    elif args.command == "finetune":
        if not args.checkpoint and not args.init_phase2:
            parser.error("--checkpoint or --init-phase2 is required")
        args.window = args.chunk
        from phase3.flash import enable
        print(enable(args.attention), flush=True)
        from .neural import finetune
        finetune(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
