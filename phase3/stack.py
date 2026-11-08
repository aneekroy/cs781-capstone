"""Stacked ensemble: a second-level classifier over the members' sentence predictions.

  python -m phase3.stack --data Phase_3_RR --runs runsP3/A runsP3/B ... --output runsP3/ST1

Averaging and voting weight every member equally for every label. The stacker learns, per label, which
members to trust (a gqa model may be better at PRE-NR, a dense one at ROD), and it sees the neighbourhood:
features per sentence are
  * every member's posterior (13 values) and its own decoded label (one-hot, its Viterbi + offsets + rules);
  * the members' averaged posterior of the two previous and two next sentences (role continuity);
  * relative position in the document, log sentence length, document length, domain.
Honest training: the members' predictions are already out-of-fold; the stacker itself is evaluated with
document-grouped 5-fold CV on top (it never scores a document it was fitted on). Its out-of-fold posteriors
then go through the same decoder choice as every run (offsets with the honest half-swap check, Dissent rule),
and the final stacker is refitted on all 90 documents to predict the test set.
Models: multinomial logistic regression and gradient-boosted trees; the better out-of-fold one is used.

Changes after the first runs (stack 0.566 / 0.541 vs plain average 0.582 / 0.586 honest):
  * no class balancing: "balanced" weights multiply Issue/None/PRE-NR by ~50x, the stacker then floods them,
    and the decoder's offsets (which already handle the per-label trade-off honestly) cannot undo it;
  * member features are log-posteriors (clipped), not probabilities + one-hot decodes: the one-hots come from
    each member's decoder tuned on all OOF documents (a leak the stacker learns to over-trust) and double the
    width; --onehot brings them back;
  * stronger regularisation (logreg C=0.05, shallower trees with larger leaves);
  * the stacker's posteriors are decoded with the same sequence candidates as the average (marginal or
    Viterbi with a gold transition prior), and the plain average is decoded the same way as the reference;
  * the submission written is the stack only if it beats that reference by --margin (0.005); otherwise the
    reference average is written and the summary says so (--force-stack overrides).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data import DOMAINS, load_split, load_vocab, pooled_score, write_prediction
from .decode import (apply_decoding, build_sequence_records, choose_sequence_decoding, crf_average, jsonable,
                     transition_prior)
from .ensemble import average, load_run


def member_features(run, labelled, test_docs, labels):
    index = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cfg = run["summary"]["decoding"]
    out = {}
    for split, recs, docs in (("oof", run["oof"], labelled), ("test", run["test"], test_docs)):
        if recs is None:
            continue
        dec = apply_decoding(recs, cfg, labels, docs if cfg.get("rules") else None)
        feats = {}
        for d, r in recs.items():
            p = np.exp(r["logp"])
            onehot = np.zeros((len(p), k))
            onehot[np.arange(len(p)), [index[l] for l in dec[d]]] = 1
            feats[d] = (p, onehot)
        out[split] = feats
    return out


def doc_matrix(members, split, doc, sentences, domain, k, onehot=False):
    probs = [m[split][doc][0] for m in members]
    n = len(probs[0])
    mean = np.mean(probs, axis=0)
    shifted = []
    for s in (-2, -1, 1, 2):
        sh = np.zeros_like(mean)
        if s < 0:
            sh[-s:] = mean[:s]
        else:
            sh[:-s] = mean[s:]
        shifted.append(sh)
    pos = (np.arange(n) + 0.5) / n
    words = np.array([math.log1p(len(x.split())) for x in sentences])
    extra = np.stack([pos, pos ** 2, words, np.full(n, math.log1p(n)), np.full(n, float(domain == "CL"))], 1)
    logs = [np.log(np.clip(p, 1e-4, None)) for p in probs]
    parts = logs + ([m[split][doc][1] for m in members] if onehot else []) + [np.log(np.clip(mean, 1e-4, None))]
    return np.concatenate(parts + shifted + [extra], 1)


def build(members, split, docs, k, onehot=False):
    xs, ids = [], []
    for d, doc in docs.items():
        xs.append(doc_matrix(members, split, d, doc["sentences"], doc["domain"], k, onehot))
        ids.append((d, len(doc["sentences"])))
    return np.concatenate(xs), ids


def make_model(kind, seed=0):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
                                          l2_regularization=2.0, random_state=seed)


def full_proba(model, x, k):
    p = np.full((len(x), k), 1e-6)
    p[:, model.classes_] = model.predict_proba(x)
    return p / p.sum(1, keepdims=True)


def split_back(p, ids):
    out, at = {}, 0
    for d, n in ids:
        out[d] = p[at:at + n]
        at += n
    return out


def records_from(p_by_doc, docs):
    return {d: {"logp": np.log(np.clip(p, 1e-9, None)), "emissions": np.log(np.clip(p, 1e-9, None)),
                "domain": docs[d]["domain"], "crf": None} for d, p in p_by_doc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", default="logreg,gbt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rules", default="auto")
    ap.add_argument("--sequence", default="marginal,smooth")
    ap.add_argument("--onehot", action="store_true", help="also feed each member's decoded labels (old behaviour)")
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--force-stack", action="store_true")
    args = ap.parse_args()
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    k = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    labelled = {d: doc for split in ("train", "dev") for d, doc in load_split(args.data, split, labels).items()}
    gold = {d: doc["labels"] for d, doc in labelled.items()}
    test_docs = load_split(args.data, "test")
    prior = transition_prior(gold.values(), labels)
    kinds = args.sequence.split(",")
    members, names, runs = [], [], []
    for r in args.runs:
        if not (Path(r) / "oof.pt").exists():
            print(f"[skip] {r}: no oof.pt")
            continue
        run = load_run(r)
        if run["test"] is None or set(run["oof"]) != set(gold):
            print(f"[skip] {run['name']}: no test.pt or different documents")
            continue
        members.append(member_features(run, labelled, test_docs, labels))
        names.append(run["name"])
        runs.append(run)
    if len(members) < 2:
        raise SystemExit("need at least two runs with oof.pt and test.pt")
    print(f"{len(members)} members: {names}")
    quiet = lambda *a, **kw: None
    # reference: plain average of the same members, decoded with the same candidates (+ CRF pooling)
    avg_oof = average([r["oof"] for r in runs])
    ref_config, ref_recs = choose_sequence_decoding(avg_oof, crf_average([r["oof"] for r in runs]), gold, labels,
                                                    labelled, prior, log=quiet, rules=args.rules,
                                                    kinds=kinds + ["crfavg"])
    print(f"reference - plain average of the same members: honest macro-F1 {ref_config['honest']:.4f} "
          f"({ref_config['sequence']['kind']})", flush=True)
    x, ids = build(members, "oof", labelled, k, args.onehot)
    y = np.concatenate([[index[l] for l in gold[d]] for d, _ in ids])
    groups = np.concatenate([[i] * n for i, (_, n) in enumerate(ids)])
    from sklearn.model_selection import GroupKFold
    best = None
    for kind in args.models.split(","):
        oof_p = np.zeros((len(y), k))
        for tr, va in GroupKFold(n_splits=args.folds).split(x, y, groups):
            model = make_model(kind).fit(x[tr], y[tr])
            oof_p[va] = full_proba(model, x[va], k)
        records = records_from(split_back(oof_p, ids), labelled)
        config, recs = choose_sequence_decoding(records, None, gold, labels, labelled, prior, log=quiet,
                                                rules=args.rules, kinds=kinds)
        print(f"stacker {kind:7s}: out-of-fold honest macro-F1 {config['honest']:.4f} ({config['sequence']['kind']}, "
              f"{config['offsets']} offsets, rules {config['rules']})", flush=True)
        if best is None or config["honest"] > best[0]["honest"]:
            best = (config, kind, records, recs)
    config, kind, records, recs = best
    use_stack = args.force_stack or config["honest"] >= ref_config["honest"] + args.margin
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if use_stack:
        decoded = apply_decoding(recs, config, labels, labelled)
        model = make_model(kind).fit(x, y)
        xt, tids = build(members, "test", test_docs, k, args.onehot)
        test = records_from(split_back(full_proba(model, xt, k), tids), test_docs)
        test = build_sequence_records(config["sequence"], test, None, prior)
        chosen, used = config, f"stack-{kind}"
    else:
        print(f"stack {config['honest']:.4f} does not beat the average {ref_config['honest']:.4f} by {args.margin}: "
              f"writing the average instead", flush=True)
        decoded = apply_decoding(ref_recs, ref_config, labels, labelled)
        test = build_sequence_records(ref_config["sequence"], average([r["test"] for r in runs]),
                                      crf_average([r["test"] for r in runs]), prior)
        chosen, used = ref_config, "average"
    final = pooled_score(gold, decoded, labels, list(gold))
    pred = apply_decoding(test, chosen, labels, test_docs)
    info = write_prediction(pred, test_docs, labels, out / "submission")
    summary = {"combine": used, "stack_model": kind, "members": names,
               "average_honest": round(float(ref_config["honest"]), 5), "stack_honest": round(float(config["honest"]), 5),
               "oof_macro_f1": round(final["macro_f1"], 5), "oof_macro_f1_honest": round(float(chosen["honest"]), 5),
               "decoding": jsonable(chosen),
               "oof_by_domain": {dom: round(pooled_score(gold, decoded, labels,
                                                         [d for d in gold if labelled[d]["domain"] == dom])["macro_f1"], 4)
                                 for dom in DOMAINS},
               "oof_per_label": {l: round(v["f1"], 4) for l, v in final["per_label"].items()},
               "test_by_label": info["by_label"]}
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{used} of {len(names)}: OOF {final['macro_f1']:.4f} honest {chosen['honest']:.4f}")
    print(f"submission: {out / 'submission' / 'submission.zip'}  {json.dumps(info['by_label'])}")


if __name__ == "__main__":
    main()
