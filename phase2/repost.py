"""Re-choose post-processing for a finished run and rewrite its submission (CPU, no retraining).

Every run keeps its raw out-of-fold spans (oof_predictions.pt) and its raw test scores
(submission/token_logprobs.pt). This picks post-processing again on the out-of-fold spans with
the current rule set (counsel side, case-insensitive and first-name/surname propagation,
precedent citations, plus the original rules), decodes the test scores with the run's own CRF
transitions, and writes <run>/repost/submission.zip. --keep-config reuses the original choice,
which must reproduce the original prediction.json exactly (a check that the pipeline is intact).

    python -m phase2.repost --data Phase_2_LNER --run runsP2/X1_denseorig_lorma
"""
import argparse
import json
from pathlib import Path

import torch

from .data import load_labels, load_texts, tag_names, tags_to_spans, write_json
from .model import CRF
from .post import BoundaryRules, build_gazetteer, postprocess


def main():
    from .run import choose_postprocessing, decode, load_split, write_submission
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", help="default: <run>/repost")
    p.add_argument("--keep-config", action="store_true", help="reuse the original post-processing (check)")
    args = p.parse_args()

    run = Path(args.run)
    out = Path(args.output) if args.output else run / "repost"
    labels = load_labels(Path(args.data) / "Train" / "labels.txt")
    folds = {f: load_split(args.data, f, labels) for f in (1, 2)}
    end_inclusive = folds[1][2]["end_inclusive"]
    summary = json.loads((run / "cv_summary.json").read_text())
    if isinstance(summary.get("o_bias"), (int, float)) and summary["o_bias"]:
        print("WARNING: this run was decoded with the withdrawn 'O' penalty; its out-of-fold spans and test "
              "decoding differ, rerun it with --o-bias auto instead")
    old_config, old_score = summary["postprocess"], summary["cv_macro_f1_post"]

    if args.keep_config:
        config, score, trace = old_config, old_score, summary["postprocess_trace"]
    else:
        oof = torch.load(run / "oof_predictions.pt", map_location="cpu", weights_only=False)["oof"]
        fold_runs = []
        for train_fold, val_fold in ((1, 2), (2, 1)):
            texts, gold, _ = folds[val_fold]
            tr_texts, tr_gold, _ = folds[train_fold]
            fold_runs.append({"raw": {d: [tuple(x) for x in oof[d]] for d in texts}, "texts": texts, "gold": gold,
                              "boundary": BoundaryRules().fit(tr_texts, tr_gold),
                              "gazetteer": build_gazetteer(tr_texts, tr_gold)})
        config, trace = choose_postprocessing(fold_runs, labels)
        score = trace[-1][1]
        print(f"{run.name}: CV post {old_score:.4f} (original rules) -> {score:.4f} (current rules)")
        print("  trace:", trace)

    # test: raw spans from the saved scores, decoded with the run's own learned transitions
    part = torch.load(run / "submission" / "token_logprobs.pt", map_location="cpu", weights_only=False)
    names = tag_names(labels)
    crf = CRF(names)
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=False)["state"]
    crf.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("crf.")})
    bonus = part.get("o_bias", 0.0)
    test = load_texts(Path(args.data) / "Test" / "fold3-text.json")
    texts = {**folds[1][0], **folds[2][0]}
    gold = {**folds[1][1], **folds[2][1]}
    boundary, gazetteer = BoundaryRules().fit(texts, gold), build_gazetteer(texts, gold)
    preds = {}
    with torch.no_grad():
        for doc in test:
            lp = part["logprobs"].get(doc)
            raw = [] if lp is None else tags_to_spans(part["offsets"][doc], decode(crf.viterbi, names, lp.float(), bonus),
                                                      labels)
            preds[doc] = postprocess(test[doc], sorted(set(raw)), config, boundary, gazetteer)
    write_submission(preds, test, out, end_inclusive, labels)
    write_json(out / "repost_summary.json", {"run": run.name, "cv_post_original": old_score, "cv_post": score,
                                              "postprocess": config, "postprocess_trace": trace,
                                              "entry_bonus": bonus})
    if args.keep_config:
        original = json.loads((run / "submission" / "prediction.json").read_text())
        mine = json.loads((out / "prediction.json").read_text())
        same = sum(original[d] == mine[d] for d in original)
        print(f"reproduced {same}/{len(original)} documents of the original prediction.json")
    print("Wrote", out / "submission.zip")


if __name__ == "__main__":
    main()
