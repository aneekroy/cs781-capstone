"""Per-label voting: each label gets its own member set and vote threshold.

Macro-F1 is the mean of per-label F1, so nothing ties DATE (where most members agree and a high
threshold removes noise) to APP (where a low threshold recovers missed parties). For every label,
members are added greedily from the candidate runs while that label's out-of-fold F1 improves,
with the best threshold k for the set. Spans of all labels are then merged, overlaps going to the
larger vote share.

Honesty: choices are made on one fold's documents and scored on the other's (and back); the
caller compares that swap score with the global vote before using it.
"""
import math
from collections import Counter

from .post import remove_overlaps, strict_scores


def _label_f1(gold, pred, label):
    tp = fp = fn = 0
    for doc, spans in gold.items():
        g = {s for s in spans if s[2] == label}
        p = set(pred.get(doc, ()))
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _vote(lists, k):
    counts = Counter(s for spans in lists for s in set(spans))
    kept = {s: c for s, c in counts.items() if c >= k}
    return remove_overlaps(list(kept), kept)


def _by_label(run_spans, label, docs):
    return {d: [tuple(s) for s in run_spans.get(d, ()) if s[2] == label] for d in docs}


def choose(oof, runs, gold, labels, docs, max_members=7, tol=0.002):
    """{label: (members, k, f1)} chosen on `docs` only."""
    config = {}
    for label in labels:
        g = {d: gold[d] for d in docs}
        per_run = {r: _by_label(oof[r], label, docs) for r in runs}

        def score(members):
            best = (-1.0, 1)
            for k in range(1, len(members) + 1):
                pred = {d: _vote([per_run[m][d] for m in members], k) for d in docs}
                best = max(best, (_label_f1(g, pred, label), -k))
            return best[0], -best[1]

        singles = sorted(((score([r])[0], r) for r in runs), reverse=True)
        members, (f1, k) = [singles[0][1]], score([singles[0][1]])
        while len(members) < max_members:
            trials = [(score(members + [r]), r) for r in runs if r not in members]
            if not trials:
                break
            (f, kk), r = max(trials, key=lambda t: t[0][0])
            if f <= f1 + tol:
                break
            members, f1, k = members + [r], f, kk
        config[label] = (members, k, f1)
    return config


def apply(config, member_spans, docs, extra_spans=(), labels=None):
    """Spans for `docs` from per-label configs. member_spans: {run: {doc: spans}};
    extra_spans: list of {doc: spans} from extra voters (same vote share rule as in vote.py)."""
    out = {}
    for d in docs:
        scored = {}
        for label, (members, k, _) in config.items():
            lists = [[tuple(s) for s in member_spans[m].get(d, ()) if s[2] == label] for m in members]
            extras = [[tuple(s) for s in e.get(d, ()) if s[2] == label] for e in extra_spans]
            mem = Counter(s for spans in lists for s in set(spans))
            total = mem + Counter(s for spans in extras for s in set(spans))
            k_all = max(1, math.ceil(k / len(members) * (len(members) + len(extras)) - 1e-9))
            for s, c in total.items():
                if mem[s] >= k or (extras and c >= k_all):
                    scored[s] = max(mem[s] / len(members), c / (len(members) + len(extras)))
        out[d] = remove_overlaps(list(scored), scored)
    return out


def swap_score(oof, runs, gold, labels, fold_docs, max_members=7):
    """Choose on one fold's documents, predict the other's; pooled macro-F1 over both."""
    preds = {}
    for a, b in ((0, 1), (1, 0)):
        config = choose(oof, runs, gold, labels, fold_docs[a], max_members)
        preds.update(apply(config, {r: oof[r] for r in runs}, fold_docs[b]))
    all_docs = fold_docs[0] + fold_docs[1]
    return strict_scores({d: gold[d] for d in all_docs}, preds, labels)["macro_f1"]
