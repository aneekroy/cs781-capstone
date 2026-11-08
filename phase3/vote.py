"""Sentence-level voting ensemble (the Phase-3 counterpart of phase2.vote).

  python -m phase3.vote --data Phase_3_RR --runs runsP3/A runsP3/B ... --output runsP3/V1 [--fixed]

Every member votes with its OWN decoded labels: its CV-chosen decoder (Viterbi over its CRF, or posterior
argmax), its offsets and its rules. Unlike posterior averaging, each vote keeps that member's sequence
structure. Per sentence:
  * plurality of the member votes, ties broken by the members' averaged probability;
  * per-label thresholds k_L (like phase 2's per-label k): a label with at least k_L votes takes the sentence
    even without a plurality, so a rare role (Issue, PRE-NR, None, RPC) that a few members see is not
    outvoted by Fact / ROD. k_L = 0 switches the override off for that label;
  * the document rules (Dissent) are applied last.
Members are added greedily while the out-of-fold pooled macro-F1 of the vote improves, with a look-ahead
over pairs when no single run helps (two members can break ties that one cannot). The thresholds are tuned
by coordinate ascent; they are used only if they win on the honest half-swap check (tune on half the
documents, score the other half, swap).
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import DOMAINS, FastScorer, load_split, load_vocab, pooled_score, write_prediction
from .decode import apply_decoding, halves, jsonable, with_rules
from .ensemble import load_run


def member_votes(run, labelled, test_docs, labels):
    """{split: {doc: label-index array}} with the member's own decoding, and its probabilities."""
    index = {l: i for i, l in enumerate(labels)}
    config = run["summary"]["decoding"]
    out = {"oof": {}, "test": {}, "p_oof": {}, "p_test": {}}
    dec = apply_decoding(run["oof"], config, labels, labelled if config.get("rules") else None)
    out["oof"] = {d: np.array([index[l] for l in v]) for d, v in dec.items()}
    out["p_oof"] = {d: np.exp(r["logp"]) for d, r in run["oof"].items()}
    if run["test"] is not None:
        dec = apply_decoding(run["test"], config, labels, test_docs if config.get("rules") else None)
        out["test"] = {d: np.array([index[l] for l in v]) for d, v in dec.items()}
        out["p_test"] = {d: np.exp(r["logp"]) for d, r in run["test"].items()}
    return out


def vote_doc(votes, probs, k, thresholds=None, forced=None):
    """votes: list of [n] index arrays; probs: list of [n,K]; thresholds: [K] ints (0 = off)."""
    n = len(votes[0])
    counts = np.zeros((n, k))
    for v in votes:
        counts[np.arange(n), v] += 1
    score = counts + 1e-3 * np.mean(probs, axis=0)          # plurality, ties -> averaged probability
    if thresholds is not None and np.any(thresholds > 0):
        qualified = (thresholds[None, :] > 0) & (counts >= thresholds[None, :])
        score = score + 100.0 * qualified
    pred = score.argmax(1)
    if forced is not None:
        pred = np.where(forced >= 0, forced, pred)
    return pred


def vote_all(members, split, docs, k, thresholds=None, forced=None):
    return {d: vote_doc([m[split][d] for m in members], [m["p_" + split][d] for m in members], k, thresholds,
                        None if forced is None else forced.get(d)) for d in docs}


def pooled(pred, gold_idx, docs, k):
    return FastScorer(np.concatenate([gold_idx[d] for d in docs]), k)(np.concatenate([pred[d] for d in docs]))


def tune_thresholds(members, docs, gold_idx, k, forced, rounds=2):
    n = len(members)
    th = np.zeros(k, dtype=np.int64)
    best = pooled(vote_all(members, "oof", docs, k, th, forced), gold_idx, docs, k)
    if n < 2:
        return th, best
    for _ in range(rounds):
        improved = False
        for label in range(k):
            keep = th[label]
            for value in range(0, n):                       # k_L = n is the plurality anyway
                if value == keep:
                    continue
                th[label] = value
                s = pooled(vote_all(members, "oof", docs, k, th, forced), gold_idx, docs, k)
                if s > best + 1e-9:
                    best, keep, improved = s, value, True
            th[label] = keep
        if not improved:
            break
    return th, best


def honest_thresholds(members, docs, gold_idx, k, forced, domains):
    """Half-swap score of threshold tuning, plain plurality score, thresholds fitted on all documents."""
    a, b = halves(docs, domains)
    preds = {}
    for fit, ev in ((a, b), (b, a)):
        th, _ = tune_thresholds(members, fit, gold_idx, k, forced)
        preds.update(vote_all(members, "oof", ev, k, th, forced))
    honest = pooled(preds, gold_idx, docs, k)
    plain = pooled(vote_all(members, "oof", docs, k, None, forced), gold_idx, docs, k)
    th_all, fit = tune_thresholds(members, docs, gold_idx, k, forced)
    return honest, plain, th_all, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixed", action="store_true", help="vote with every listed run (no greedy selection)")
    ap.add_argument("--max-members", type=int, default=9)
    ap.add_argument("--rules", default="auto", help="auto/dissent: apply the Dissent rule last; none: off")
    ap.add_argument("--allow-no-test", action="store_true", help="also consider runs without test.pt (OOF study only)")
    args = ap.parse_args()
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    k = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    labelled = {d: doc for split in ("train", "dev") for d, doc in load_split(args.data, split, labels).items()}
    gold = {d: doc["labels"] for d, doc in labelled.items()}
    gold_idx = {d: np.array([index[l] for l in v]) for d, v in gold.items()}
    test_docs = load_split(args.data, "test")
    rule_set = [] if args.rules == "none" else ["dissent"]
    from .rules import forced_indices
    forced_oof = {d: forced_indices(labelled[d]["sentences"], rule_set, index) for d in labelled} if rule_set else None
    forced_test = {d: forced_indices(test_docs[d]["sentences"], rule_set, index) for d in test_docs} if rule_set else None

    members = []
    for r in args.runs:
        if not (Path(r) / "oof.pt").exists():
            print(f"[skip] {r}: no oof.pt")
            continue
        run = load_run(r)
        if set(run["oof"]) != set(gold):
            print(f"[skip] {r}: out-of-fold documents differ from train+dev")
            continue
        if run["test"] is None and not args.allow_no_test:
            print(f"[skip] {run['name']}: no test.pt (bash run_phase3.sh finalize) - it could not vote on the test set")
            continue
        m = member_votes(run, labelled, test_docs, labels)
        m["name"], m["has_test"] = run["name"], run["test"] is not None
        members.append(m)
        own = pooled(m["oof"], gold_idx, list(gold), k)
        print(f"{m['name']:36s} own decoding OOF {own:.4f}  test.pt {'yes' if m['has_test'] else 'NO'}")
    if not members:
        raise SystemExit("no usable runs")
    docs = list(gold)
    domains = {d: labelled[d]["domain"] for d in docs}

    def score(group):        # plain plurality (+ rules) during selection; thresholds are tuned at the end
        return pooled(vote_all(group, "oof", docs, k, None, forced_oof), gold_idx, docs, k)

    if args.fixed:
        chosen = members
        trace = [(m["name"], None) for m in chosen]
    else:
        ranked = sorted(members, key=lambda m: score([m]), reverse=True)
        chosen, best = [ranked[0]], score([ranked[0]])
        trace = [(chosen[0]["name"], round(best, 5))]
        print(f"start {chosen[0]['name']} {best:.4f}")
        while len(chosen) < args.max_members:
            taken = {c["name"] for c in chosen}
            rest = [m for m in members if m["name"] not in taken]
            if not rest:
                break
            singles = sorted(((score(chosen + [m]), i) for i, m in enumerate(rest)), reverse=True)
            if singles[0][0] > best + 1e-4:
                best, m = singles[0][0], rest[singles[0][1]]
                chosen.append(m)
                trace.append((m["name"], round(best, 5)))
                print(f"+ {m['name']} -> {best:.4f}")
                continue
            top = [rest[i] for _, i in singles[:8]]              # pair look-ahead (phase-2 style)
            pairs = sorted(((score(chosen + list(p)), p) for p in combinations(top, 2)
                            if len(chosen) + 2 <= args.max_members), key=lambda t: t[0], reverse=True)
            if pairs and pairs[0][0] > best + 1e-4:
                best, p = pairs[0]
                chosen += list(p)
                trace.append((f"{p[0]['name']} + {p[1]['name']}", round(best, 5)))
                print(f"+ {p[0]['name']} + {p[1]['name']} -> {best:.4f}  (pair)")
                continue
            break
    honest, plain, th, fit = honest_thresholds(chosen, docs, gold_idx, k, forced_oof, domains)
    use = honest > plain + 0.003
    thresholds = th if use else None
    print(f"vote of {len(chosen)}: plurality OOF {plain:.4f} | per-label thresholds half-swap {honest:.4f} "
          f"(fit {fit:.4f}) -> {'used' if use else 'not used'}"
          + (f": {dict((labels[i], int(t)) for i, t in enumerate(th) if t)}" if use else ""))
    pred_oof = vote_all(chosen, "oof", docs, k, thresholds, forced_oof)
    final = pooled_score(gold, {d: [labels[i] for i in v] for d, v in pred_oof.items()}, labels, docs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"combine": "vote", "members": [m["name"] for m in chosen], "trace": trace,
               "oof_macro_f1": round(final["macro_f1"], 5), "oof_macro_f1_honest": round(honest if use else plain, 5),
               "oof_plurality": round(plain, 5), "thresholds": {labels[i]: int(t) for i, t in enumerate(th) if t} if use else {},
               "rules": rule_set,
               "oof_by_domain": {dom: round(pooled_score(gold, {d: [labels[i] for i in v] for d, v in pred_oof.items()},
                                                         labels, [d for d in docs if domains[d] == dom])["macro_f1"], 4)
                                 for dom in DOMAINS},
               "oof_per_label": {l: round(v["f1"], 4) for l, v in final["per_label"].items()}}
    missing = [m["name"] for m in chosen if not m["has_test"]]
    if missing:
        print(f"no test.pt for {missing}: OOF only (bash run_phase3.sh finalize first)")
    else:
        pred = vote_all(chosen, "test", list(test_docs), k, thresholds, forced_test)
        info = write_prediction({d: [labels[i] for i in v] for d, v in pred.items()}, test_docs, labels, out / "submission")
        summary["test_by_label"] = info["by_label"]
        print(f"submission: {out / 'submission' / 'submission.zip'}  {json.dumps(info['by_label'])}")
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"vote OOF {final['macro_f1']:.4f}  honest {summary['oof_macro_f1_honest']:.4f}  members {summary['members']}")


if __name__ == "__main__":
    main()
