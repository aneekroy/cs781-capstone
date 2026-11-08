"""Final fit + test submission for a run whose cross-validation is already done (e.g. the S_* screen runs,
which ran without --final): trains once on every labelled document for the CV-chosen epoch count, predicts the
test set with the CV-chosen decoding (offsets + rules) and writes <run>/submission/submission.zip.

    python -m phase3.finalize --run runsP3/S_dense_dapt [--data Phase_3_RR]
"""
import argparse
import json
from pathlib import Path

from .run import cmd_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", help="override the data path stored in args.json")
    a = ap.parse_args()
    run = Path(a.run)
    if not (run / "cv_summary.json").exists() or not (run / "args.json").exists():
        raise SystemExit(f"{run}: needs cv_summary.json and args.json (a finished cv run)")
    stored = json.loads((run / "args.json").read_text())
    args = argparse.Namespace(**stored)
    defaults = {"rules": "auto", "repeat_labels": "", "domain_weights": "", "splice": 0.0,
                "splice_labels": "Dissent,Issue,PrecedentNotReliedUpon,RulingByPresentCourt,None",
                "kd_select_domain": False, "kd_take_prob": 0.1, "prompt_tokens": 20, "prefix_length": 16,
                "prefix_hidden": 512, "adalora_init_r": 12, "adalora_target_r": 8, "adalora_orth": 0.1}
    for key, value in defaults.items():               # runs made by older versions
        if not hasattr(args, key):
            setattr(args, key, value)
    if isinstance(getattr(args, "kd_unlabeled", None), str):
        args.kd_unlabeled = [args.kd_unlabeled]
    if a.data:
        args.data = a.data
    args.output = str(run)
    args.final = True
    args.final_only = True
    cmd_cv(args)


if __name__ == "__main__":
    main()
