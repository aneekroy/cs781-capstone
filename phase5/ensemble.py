"""Ensemble of finished Phase-5 runs (TF-IDF baseline, fine-tuned models): average statute probabilities, members
chosen greedily on dev by the honest split-half Macro-F1 (per-statute thresholds with shrinkage), then the
usual threshold choice (phase5.data.choose_thresholds) on the averaged probabilities.

  python -m phase5.ensemble --data Phase_5_LSI --runs runsP5/B0_tfidf runsP5/L1 ... --output runsP5/E1
"""
import argparse
from pathlib import Path

import numpy as np

from .data import (apply_thresholds, finish, fit_thresholds, halves, label_space, load_probs, load_split, multi_hot,
                   scores)


def honest(y, p, shrink=10.0):
    """Split-half Macro-F1 with per-statute thresholds (the selection score; cheap)."""
    a, b = halves(len(y))
    pred = np.zeros_like(y, dtype=np.int8)
    for fit, ev in ((a, b), (b, a)):
        pred[ev] = apply_thresholds(p[ev], fit_thresholds(y[fit], p[fit], "per_label", shrink))
    return scores(y, pred)["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixed", action="store_true", help="use every listed run")
    ap.add_argument("--max-members", type=int, default=6)
    args = ap.parse_args()
    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    labels = label_space(train, dev)
    dev_ids, test_ids = list(dev), list(test)
    y = multi_hot(dev, labels).astype(bool)
    members = []
    for r in args.runs:
        r = Path(r)
        if not (r / "probs.npz").exists():
            print(f"[skip] {r.name}: no probs.npz")
            continue
        pr = load_probs(r / "probs.npz")
        if pr["labels"] != labels or pr["dev_ids"] != dev_ids or pr["test_ids"] != test_ids:
            print(f"[skip] {r.name}: different documents or label space (rerun it with the current loader)")
            continue
        members.append({"name": r.name, "dev": pr["dev"], "test": pr["test"]})
        print(f"{r.name:36s} dev honest macro-F1 {honest(y, pr['dev']):.4f}", flush=True)
    if not members:
        raise SystemExit("no usable runs")
    avg = lambda idx, split: np.mean([members[i][split] for i in idx], axis=0)
    if args.fixed:
        chosen = list(range(len(members)))
    else:
        single = [(honest(y, m["dev"]), i) for i, m in enumerate(members)]
        best, first = max(single)
        chosen = [first]
        while len(chosen) < args.max_members:
            trials = [(honest(y, avg(chosen + [j], "dev")), j) for j in range(len(members)) if j not in chosen]
            if not trials:
                break
            s, j = max(trials)
            if s <= best + 1e-4:
                break
            chosen.append(j)
            best = s
            print(f"+ {members[j]['name']} -> {s:.4f}", flush=True)
    names = [members[i]["name"] for i in chosen]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    finish(y, avg(chosen, "dev"), avg(chosen, "test"), dev, test, labels, out,
           {"stage": "ensemble", "members": names})
    print(f"ensemble of {len(names)}: {names}")


if __name__ == "__main__":
    main()
