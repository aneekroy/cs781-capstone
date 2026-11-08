"""CS781 Phase 7 (Prior Case Retrieval) command line.

  stats     queries, pool sizes, relevant per query, case lengths, queries inside their own pool
  lexical   word n-gram BM25 (n = 1..5), parameters and fusion weights tuned on train, scored on dev
            -> lexical.json, summary.json, submission.zip
  train     a Phase-1 retriever (any technique, KD with --teacher) -> model.pt, summary.json,
            submission/ (fused with the lexical scores) and submission_neural/ (Phase-1 model alone)
  rerank    passage-pair re-ranker (Phase-1 model reads query + candidate passages together) over a first stage
  combine   several finished runs (+ lexical) fused with weights chosen on dev (z-score or RRF) -> submission
  check     validate a prediction.json against the test split
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def cmd_stats(args):
    from .data import load_split, stats
    for split in ("train", "dev", "test"):
        q, p = load_split(args.data, split)
        print(split, json.dumps(stats(q, p)))
    tr, dv, te = (set(load_split(args.data, s)[1]) for s in ("train", "dev", "test"))
    print(json.dumps({"pool_overlap_train_dev": len(tr & dv), "pool_overlap_train_test": len(tr & te),
                      "pool_overlap_dev_test": len(dv & te)}))


def cmd_lexical(args):
    from .data import f1_at_k, load_split, rank_from_scores, recall_at, write_submission
    from .lexical import FusedLexical, SplitIndex, tune
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grams = tuple(args.grams)
    tq, tp = load_split(args.data, args.tune_on)
    print(f"tuning BM25 on {args.tune_on}: {len(tq)} queries, {len(tp)} candidates", flush=True)
    from .data import query_texts
    index = SplitIndex(tq, tp)
    views = {w: index.view(query_texts(tq, w)) for w in args.context_windows} if args.context_windows else None
    prf = [(d, t, b) for d in (3, 5, 10) for t in (100, 300) for b in (0.3, 0.7)] if args.prf else None
    tuned = tune(index, grams, qweights=tuple(args.qweights), context_views=views, prf_grid=prf,
                 passage_words=args.passage_words or None)
    tuned = {**tuned, "per_n": {str(k): v for k, v in tuned["per_n"].items()},
             "weights": {str(k): v for k, v in tuned["weights"].items()}, "tuned_on": args.tune_on}
    (out / "lexical.json").write_text(json.dumps(tuned, indent=1))
    lex = FusedLexical(tuned)
    res = {"tuned": tuned}
    for split in ("dev", "test"):
        q, p = load_split(args.data, split)
        index = SplitIndex(q, p)
        s = lex.scores(index)
        ranked = rank_from_scores(s, index.query_ids, index.pool_ids)
        if split == "dev":
            res["dev"] = {**f1_at_k(ranked, index.gold), **recall_at(s, index.query_ids, index.pool_ids, index.gold)}
            res["dev_per_n"] = {}
            for n, v in tuned["per_n"].items():
                sn = index.bm25(int(n), v["k1"], v["b"], v["max_df"], v.get("qweight", "binary"))
                res["dev_per_n"][n] = round(f1_at_k(rank_from_scores(sn, index.query_ids, index.pool_ids), index.gold)["f1"], 4)
            print("dev:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res["dev"].items()}),
                  "| per n:", res["dev_per_n"], flush=True)
        else:
            write_submission(ranked, q, p, out / "submission")
        np.savez_compressed(out / f"scores_{split}.npz", lexical=s, query_ids=np.array(index.query_ids),
                            pool_ids=np.array(index.pool_ids))
    res["minutes"] = round((time.time() - started) / 60, 1)
    (out / "summary.json").write_text(json.dumps({"stage": "lexical", **res}, indent=1))
    print(f"-> {out}/lexical.json, {out}/submission/submission.zip ({res['minutes']} min)", flush=True)


def _ranks(s):
    """1-based rank of every candidate per query (1 = best)."""
    order = np.argsort(-s, axis=1)
    r = np.empty_like(order)
    r[np.arange(s.shape[0])[:, None], order] = np.arange(1, s.shape[1] + 1)[None, :]
    return r


def cmd_combine(args):
    """Fuse the dev/test scores saved by finished runs (retrievers, re-rankers, lexical): per-run weights by
    coordinate ascent on dev F1@7; z-score sum or reciprocal-rank fusion (--fusion rrf)."""
    from .data import f1_at_k, load_split, rank_from_scores, write_submission, zscore_rows
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for split in ("dev", "test"):
        cols = {}
        for run in args.runs:
            z = np.load(Path(run) / f"scores_{split}.npz")
            key = "neural" if "neural" in z.files else "lexical"
            cols[Path(run).name] = (z[key], [str(x) for x in z["query_ids"]], [str(x) for x in z["pool_ids"]])
            if "lexical" in z.files and "lexical" not in cols and not args.no_lexical and key != "lexical":
                cols["lexical"] = (z["lexical"], cols[Path(run).name][1], cols[Path(run).name][2])
        parts[split] = cols
    names = list(parts["dev"])
    ref = parts["dev"][names[0]]
    qd, pd = ref[1], ref[2]

    def aligned(split, n):
        s, rows, cols = parts[split][n]
        want_r, want_c = parts[split][names[0]][1], parts[split][names[0]][2]
        r, c = {q: i for i, q in enumerate(rows)}, {d: j for j, d in enumerate(cols)}
        return s[np.ix_([r[q] for q in want_r], [c[d] for d in want_c])]

    def transform(m):
        return 1.0 / (args.rrf_k + _ranks(m)) if args.fusion == "rrf" else zscore_rows(m)

    dq, dp = load_split(args.data, "dev")
    tq, tp = load_split(args.data, "test")
    gold = {q: dq[q]["relevant"] for q in qd}
    tdev = {n: transform(aligned("dev", n)) for n in names}
    qt, pt = parts["test"][names[0]][1], parts["test"][names[0]][2]
    feats = {"dev": {}, "test": {}}
    if args.prior:
        # how often each case was cited by the LABELLED queries (train for dev; train + dev for test); pools overlap
        from collections import Counter
        trq, _ = load_split(args.data, "train")
        c_train = Counter(c for v in trq.values() for c in (v["relevant"] or []))
        c_all = c_train + Counter(c for v in dq.values() for c in (v["relevant"] or []))
        for split, cnt, q, p in (("dev", c_train, qd, pd), ("test", c_all, qt, pt)):
            row = np.log1p(np.array([cnt.get(c, 0) for c in p], dtype=np.float64))
            feats[split]["prior"] = zscore_rows(np.tile(row, (len(q), 1)) + 1e-9 * np.random.default_rng(0).random((len(q), len(p))))
    if args.years:
        # a candidate mentioning a year later than every year in the query cannot be cited by it
        from .data import max_year
        for split, (qq, pp), q, p in (("dev", (dq, dp), qd, pd), ("test", (tq, tp), qt, pt)):
            yq = np.array([max_year(qq[x]["text"]) for x in q])
            yp = np.array([max_year(pp[x]) for x in p])
            later = (yp[None, :] > yq[:, None]) & (yq[:, None] > 0)
            feats[split]["later_year"] = -later.astype(np.float64)
    if args.knn:
        # citations of the most similar labelled queries (train for dev; train + dev for test), k chosen on dev
        from .lexical import neighbor_citations
        trq, _ = load_split(args.data, "train")
        nb_dev, rows = neighbor_citations(dq, trq, pd)
        nb_dev = {k: m[[rows.index(q) for q in qd]] for k, m in nb_dev.items()}
        kbest = max(nb_dev, key=lambda k: f1_at_k(rank_from_scores(nb_dev[k] + 1e-6 * zscore_rows(tdev[names[0]]), qd, pd),
                                                  gold)["f1"])
        nb_test, rows_t = neighbor_citations(tq, {**trq, **dq}, pt, ks=(kbest,))
        feats["dev"]["neighbours"] = zscore_rows(nb_dev[kbest] + 1e-9)
        feats["test"]["neighbours"] = zscore_rows(nb_test[kbest][[rows_t.index(q) for q in qt]] + 1e-9)
        print(f"neighbour citations: k={kbest}, alone dev F1@7 "
              f"{f1_at_k(rank_from_scores(nb_dev[kbest] + 1e-6 * zscore_rows(tdev[names[0]]), qd, pd), gold)['f1']:.4f}",
              flush=True)
    for k, v in feats["dev"].items():
        tdev[k] = v
    names = names + list(feats["dev"])
    f = lambda w: f1_at_k(rank_from_scores(sum(tdev[n] * w[n] for n in names if w[n]), qd, pd), gold)["f1"]
    w = {n: (0.0 if n in feats["dev"] else 1.0) for n in names}
    cur = f(w)
    single = {n: round(f({m: float(m == n) for m in names}), 4) for n in names if n not in feats["dev"]}
    for _ in range(3):
        for n in names:
            for v in (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
                trial = {**w, n: v}
                if any(trial.values()) and f(trial) > cur + 1e-9:
                    cur, w = f(trial), trial
    test_t = lambda n: feats["test"][n] if n in feats["test"] else transform(aligned("test", n))
    sdev = sum(tdev[n] * w[n] for n in names if w[n])
    st = sum(test_t(n) * w[n] for n in names if w[n])
    hub = 0.0
    if args.hub:
        # hubness correction: mix in each candidate's score standardised ACROSS queries, so a case that scores
        # high for every query (long, generic) loses to one that stands out for this query
        def mix(m, lam):
            col = (m - m.mean(0, keepdims=True)) / np.maximum(m.std(0, keepdims=True), 1e-9)
            return (1 - lam) * zscore_rows(m) + lam * col
        best = (cur, 0.0)
        for lam in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            fh = f1_at_k(rank_from_scores(mix(sdev, lam), qd, pd), gold)["f1"]
            if fh > best[0] + 1e-9:
                best = (fh, lam)
        cur, hub = best
        if hub:
            sdev, st = mix(sdev, hub), mix(st, hub)
    write_submission(rank_from_scores(st, qt, pt), tq, tp, out / "submission")
    lex = {sp: aligned(sp, "lexical") for sp in ("dev", "test")} if "lexical" in names else {}
    for split, mat, q, p in (("dev", sdev, qd, pd), ("test", st, qt, pt)):
        extra = {"lexical": lex[split]} if lex else {}
        np.savez_compressed(out / f"scores_{split}.npz", neural=mat, fused=mat, query_ids=np.array(q),
                            pool_ids=np.array(p), **extra)
    summary = {"stage": "combine", "fusion": args.fusion, "runs": [str(r) for r in args.runs], "weights": w,
               "hub": hub, "features": list(feats["dev"]), "dev_f1_each": single, "dev_f1": cur}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def cmd_check(args):
    from .data import load_split, validate_prediction
    q, p = load_split(args.data, "test")
    validate_prediction(json.loads(Path(args.prediction).read_text()), q, p)
    print(f"OK: {len(q)} queries x 7 candidates from the test pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    lx = sub.add_parser("lexical")
    lx.add_argument("--data", required=True)
    lx.add_argument("--output", required=True)
    lx.add_argument("--grams", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    lx.add_argument("--tune-on", default="train", choices=["train", "dev"])
    lx.add_argument("--qweights", nargs="+", default=["binary", "logtf", "tf"], help="query-side weightings tried")
    lx.add_argument("--context-windows", type=int, nargs="*", default=[],
                    help="citation-context query views tried (sentences around each <CITATION> mark), e.g. 0 1 2 5")
    lx.add_argument("--prf", action="store_true", help="also try pseudo-relevance feedback (12 settings)")
    lx.add_argument("--passage-words", type=int, nargs="*", default=[],
                    help="also try best-passage BM25 with passages of these token lengths, e.g. 150 300")
    from .neural import add_args
    add_args(sub.add_parser("train"))
    c = sub.add_parser("combine")
    c.add_argument("--data", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--no-lexical", action="store_true")
    c.add_argument("--fusion", default="zscore", choices=["zscore", "rrf"])
    c.add_argument("--rrf-k", type=float, default=60.0)
    c.add_argument("--prior", action="store_true", help="feature: how often labelled queries cited the case")
    c.add_argument("--years", action="store_true", help="feature: candidate mentions a year later than the query")
    c.add_argument("--hub", action="store_true", help="hubness correction (column standardisation), tuned on dev")
    c.add_argument("--knn", action="store_true", help="feature: citations of the most similar labelled queries")
    from .rerank import add_args as rerank_args
    rerank_args(sub.add_parser("rerank"))
    k = sub.add_parser("check")
    k.add_argument("--data", required=True)
    k.add_argument("--prediction", required=True)
    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "lexical":
        cmd_lexical(args)
    elif args.command in ("train", "rerank"):
        from phase3.flash import enable
        print(enable(args.attention), flush=True)
        from phase6.model import install_masks
        install_masks()
        if args.command == "train":
            from .neural import train
        else:
            from .rerank import train
        train(args)
    elif args.command == "combine":
        cmd_combine(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
