"""Print real out-of-fold errors with context, for writing targeted rules.

    python -m phase2.errors --data Phase_2_LNER --run runsP2/X23_denseorig_entry --labels APP RESP WIT --n 12

For each label: MISSED (gold, not predicted), SPURIOUS (predicted, not gold) and CONFUSED
(right span, wrong label), each shown as  ...left [[span]] right...  after the run's own
post-processing. Text is training data (folds 1-2), never test text.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from .data import load_labels
from .post import BoundaryRules, build_gazetteer, postprocess


def show(text, s, e, width=70):
    left = text[max(0, s - width):s].replace("\n", " ")
    right = text[e:e + width].replace("\n", " ")
    return f"...{left}[[{text[s:e]}]]{right}..."


def main():
    from .run import load_split
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--labels", nargs="+", default=["APP", "RESP", "WIT"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run = Path(args.run)
    labels = load_labels(Path(args.data) / "Train" / "labels.txt")
    config = json.loads((run / "cv_summary.json").read_text())["postprocess"]
    oof = torch.load(run / "oof_predictions.pt", map_location="cpu", weights_only=False)["oof"]
    folds = {f: load_split(args.data, f, labels) for f in (1, 2)}
    rng = random.Random(args.seed)
    found = {l: {"MISSED": [], "SPURIOUS": [], "CONFUSED": []} for l in args.labels}
    for train_fold, val_fold in ((1, 2), (2, 1)):
        texts, gold, _ = folds[val_fold]
        boundary = BoundaryRules().fit(*folds[train_fold][:2])
        gazetteer = build_gazetteer(*folds[train_fold][:2])
        for doc, text in texts.items():
            pred = set(postprocess(text, [tuple(x) for x in oof[doc]], config, boundary, gazetteer))
            g = set(map(tuple, gold.get(doc, [])))
            p_by_span = {(s, e): l for s, e, l in pred}
            g_by_span = {(s, e): l for s, e, l in g}
            for s, e, l in g - pred:
                if l in found:
                    other = p_by_span.get((s, e))
                    kind = "CONFUSED" if other else "MISSED"
                    found[l][kind].append(f"[{doc}] gold {l}" + (f", predicted {other}" if other else "")
                                          + ": " + show(text, s, e))
            for s, e, l in pred - g:
                if l in found and (s, e) not in g_by_span:
                    found[l]["SPURIOUS"].append(f"[{doc}] predicted {l}: " + show(text, s, e))
    for l, kinds in found.items():
        for kind, rows in kinds.items():
            print(f"\n=== {l} {kind}: {len(rows)} in total, {min(args.n, len(rows))} shown ===")
            for row in rng.sample(rows, min(args.n, len(rows))):
                print(row)


if __name__ == "__main__":
    main()
