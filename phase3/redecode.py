"""Re-choose the decoding of finished runs with the current decoder + rules, without retraining.

    python -m phase3.redecode --data Phase_3_RR --runs runsP3/P1_dense_lorma_ctx runsP3/K2_gqa_lora_kd ...

Uses the saved out-of-fold posteriors (oof.pt) and test posteriors (test.pt). The previous submission is kept
as submission_before_rules/; cv_summary.json gets the new decoding and scores (old ones under *_before_rules).
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from .data import DOMAINS, load_split, load_vocab, pooled_score, write_prediction
from .decode import apply_decoding, finalize_decoding, jsonable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rules", default="auto")
    ap.add_argument("--workers", type=int, default=0, help="parallel runs (default: min(8, CPUs))")
    args = ap.parse_args()
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = args.workers or min(8, os.cpu_count() or 1, len(args.runs))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for line in pool.map(redecode_one, [(args.data, r, args.rules) for r in args.runs]):
                print(line, flush=True)
    else:
        for r in args.runs:
            print(redecode_one((args.data, r, args.rules)), flush=True)


def redecode_one(job):
    data, run, rules = job
    run = Path(run)
    labels = load_vocab(Path(data) / "Train" / "label_vocab.json")
    labelled = {d: doc for split in ("train", "dev") for d, doc in load_split(data, split, labels).items()}
    gold = {d: doc["labels"] for d, doc in labelled.items()}
    test_docs = load_split(data, "test")
    if True:
        if not (run / "oof.pt").exists():
            return f"[skip] {run.name}: no oof.pt"
        summary = json.loads((run / "cv_summary.json").read_text())
        oof = torch.load(run / "oof.pt", map_location="cpu", weights_only=False)["oof"]
        ids = [d for d in oof if d in gold]
        quiet = lambda *a, **k: None
        config, results = finalize_decoding(oof, gold, labels, labelled, log=quiet, rules=rules)
        decoded = apply_decoding(oof, config, labels, labelled)
        score = pooled_score(gold, decoded, labels, ids)
        old = summary.get("oof_macro_f1_honest")
        if "decoding_before_rules" not in summary:
            summary["decoding_before_rules"] = summary["decoding"]
            summary["oof_macro_f1_honest_before_rules"] = old
        summary.update({
            "decoding": jsonable(config), "oof_macro_f1": round(score["macro_f1"], 5),
            "oof_macro_f1_honest": round(float(config["honest"]), 5),
            "oof_by_domain": {dom: round(pooled_score(gold, decoded, labels,
                                                      [d for d in ids if oof[d]["domain"] == dom])["macro_f1"], 4)
                              for dom in DOMAINS},
            "oof_per_label": {l: {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                              for l, s in score["per_label"].items()}})
        (run / "cv_summary.json").write_text(json.dumps(summary, indent=1))
        line = (f"{run.name:32s} honest {old:.4f} -> {config['honest']:.4f}  ({config['mode']} + {config['offsets']} "
                f"offsets + rules {config['rules']})  Dissent F1 {score['per_label']['Dissent']['f1']:.3f}")
        test_path = run / "test.pt"
        if test_path.exists():
            test = torch.load(test_path, map_location="cpu", weights_only=False)["test"]
            pred = apply_decoding(test, config, labels, test_docs)
            if (run / "submission").exists() and not (run / "submission_before_rules").exists():
                shutil.copytree(run / "submission", run / "submission_before_rules")
            info = write_prediction(pred, test_docs, labels, run / "submission")
            line += f"  test Dissent {info['by_label']['Dissent']}"
        return line


if __name__ == "__main__":
    main()
