"""Ensemble of finished phase3 runs: average sentence posteriors, decode with tuned per-label offsets.

  python -m phase3.ensemble --data Phase_3_RR --runs runsP3/R1 runsP3/R2 ... --output runsP3/ENS1

Members are chosen greedily on the out-of-fold pooled macro-F1 (the honest half-swap score of the
decoder, so offset tuning cannot pick members by overfitting). Out-of-fold posteriors of runs with
different fold splits can still be averaged: every document's posterior comes from a model that never
saw it. --fixed keeps every listed run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_split, load_vocab, pooled_score, write_prediction
from .decode import (apply_decoding, build_sequence_records, choose_decoding, choose_sequence_decoding, crf_average,
                     jsonable, transition_prior)


def load_run(run):
    run = Path(run)
    oof = torch.load(run / "oof.pt", map_location="cpu", weights_only=False)["oof"]
    test_path = run / "test.pt"
    test = torch.load(test_path, map_location="cpu", weights_only=False)["test"] if test_path.exists() else None
    summary = json.loads((run / "cv_summary.json").read_text())
    return {"name": run.name, "oof": oof, "test": test, "summary": summary}


def average(parts, weights=None):
    """parts: list of {doc: record}; probability average -> log."""
    weights = weights or [1.0] * len(parts)
    total = sum(weights)
    out = {}
    for doc in parts[0]:
        p = sum(w * np.exp(part[doc]["logp"]) for w, part in zip(weights, parts)) / total
        logp = np.log(np.clip(p, 1e-12, None))
        out[doc] = {"logp": logp, "emissions": logp, "domain": parts[0][doc]["domain"], "crf": None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixed", action="store_true", help="use every listed run (no greedy selection)")
    ap.add_argument("--max-members", type=int, default=7)
    ap.add_argument("--rules", default="auto")
    ap.add_argument("--sequence", default="marginal,smooth,crfavg",
                    help="decoders to compare for the averaged posteriors (marginal = the old per-sentence argmax)")
    args = ap.parse_args()
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    labelled = {d: doc for split in ("train", "dev") for d, doc in load_split(args.data, split, labels).items()}
    gold = {d: doc["labels"] for d, doc in labelled.items()}
    runs = []
    for r in args.runs:
        if not (Path(r) / "oof.pt").exists():
            print(f"[skip] {r}: no oof.pt (CV not finished)")
            continue
        run = load_run(r)
        if set(run["oof"]) != set(gold):
            print(f"[skip] {r}: out-of-fold documents differ from train+dev (protocol dev?)")
            continue
        runs.append(run)
        print(f"{run['name']:40s} OOF {run['summary']['oof_macro_f1']:.4f}  honest {run['summary']['oof_macro_f1_honest']:.4f}"
              f"  test.pt {'yes' if run['test'] is not None else 'NO'}")
    if not runs:
        raise SystemExit("no usable runs")
    quiet = lambda *a, **k: None

    from .decode import with_rules
    from .rules import RULES
    # member selection is scored with the text rules only (fast); the final decoder tries every rule variant
    rule_set = ["dissent"] if args.rules == "auto" else [r for r in args.rules.split(",") if r and r != "none" and r in RULES]

    def score(members):        # members judged with the rules in place, as they will be decoded
        recs = average([m["oof"] for m in members])
        if rule_set:
            recs = with_rules(recs, labelled, rule_set, labels)
        best, _ = choose_decoding(recs, gold, labels, log=quiet, modes=["marginal"])
        return best["honest"]

    if args.fixed:
        chosen = runs
        trace = [(m["name"], None) for m in chosen]
    else:
        singles = sorted(((score([r]), i) for i, r in enumerate(runs)), reverse=True)
        chosen, best = [runs[singles[0][1]]], singles[0][0]
        trace = [(chosen[0]["name"], round(best, 5))]
        print(f"start {chosen[0]['name']} {best:.4f}")
        while len(chosen) < args.max_members:
            trials = [(score(chosen + [r]), r) for r in runs if r not in chosen]
            if not trials:
                break
            s, r = max(trials, key=lambda t: t[0])
            if s <= best + 0.001:
                break
            chosen.append(r)
            best = s
            trace.append((r["name"], round(s, 5)))
            print(f"+ {r['name']} -> {s:.4f}")
    prior = transition_prior(gold.values(), labels)
    oof = average([m["oof"] for m in chosen])
    crf_oof = crf_average([m["oof"] for m in chosen])
    config, oof_recs = choose_sequence_decoding(oof, crf_oof, gold, labels, labelled, prior, rules=args.rules,
                                                kinds=args.sequence.split(","))
    decoded = apply_decoding(oof_recs, config, labels, labelled)
    final = pooled_score(gold, decoded, labels, list(oof))
    print(f"ensemble of {len(chosen)}: OOF {final['macro_f1']:.4f} (honest {config['honest']:.4f}) "
          f"decoding {config['sequence']['kind']} {config['mode']} + {config['offsets']} offsets + rules {config['rules']}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"members": [m["name"] for m in chosen], "trace": trace, "oof_macro_f1": round(final["macro_f1"], 5),
               "oof_macro_f1_honest": round(float(config["honest"]), 5), "decoding": jsonable(config),
               "oof_per_label": {l: round(v["f1"], 4) for l, v in final["per_label"].items()}}
    missing = [m["name"] for m in chosen if m["test"] is None]
    if missing:
        print(f"no test.pt for {missing} (run them with --final); OOF only")
    else:
        test_docs = load_split(args.data, "test")
        test = average([m["test"] for m in chosen])
        test = build_sequence_records(config["sequence"], test, crf_average([m["test"] for m in chosen]), prior)
        pred = apply_decoding(test, config, labels, test_docs)
        info = write_prediction(pred, test_docs, labels, out / "submission")
        summary["test_by_label"] = info["by_label"]
        print(f"submission: {out / 'submission' / 'submission.zip'}  {json.dumps(info['by_label'])}")
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
