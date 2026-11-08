"""Lexical retrieval for Phase 7: BM25 over word n-grams (n = 1..5), fused across n.

Why: a judgment that cites a prior case usually quotes it, so long shared word sequences are strong evidence
(on IL-PCR, BM25 over bigrams already beat every transformer retriever in IL-TUR, and 4-5-gram matching on
event-filtered text was best). N-grams are hashed numerically (numpy rolling hash, 2^24 buckets), so a
quad-gram index of 7k long judgments needs no string vocabulary. IDF comes from the split's own candidate pool
(the pools are published with the queries); placeholders like <NAME>/<ORG> are dropped before n-gramming.

Score fusion across n: z-score each n's scores per query, weighted sum. The weights, k1 and b are tuned on the
TRAIN queries (827) and checked on dev.
"""
import itertools
import math

import numpy as np
from scipy import sparse

from .data import f1_at_k, normalize, rank_from_scores, zscore_rows

FEATURES = 1 << 24


class Vocab:
    def __init__(self):
        self.ids = {}

    def encode(self, words):
        ids = self.ids
        return np.fromiter((ids.setdefault(w, len(ids) + 1) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(ids, n, features=FEATURES):
    if len(ids) < n:
        return np.zeros(0, dtype=np.int64)
    m = len(ids) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * np.uint64(1000003) + ids[j:j + m] + np.uint64(j * 7919)
    return (h & np.uint64(features - 1)).astype(np.int64)


def count_matrix(token_ids, n, features=FEATURES, binary=False):
    """CSR [docs, features] of hashed n-gram counts."""
    indptr, indices, data = [0], [], []
    for ids in token_ids:
        u, c = np.unique(ngram_hashes(ids, n, features), return_counts=True)
        indices.append(u)
        data.append(np.ones_like(c, dtype=np.float32) if binary else c.astype(np.float32))
        indptr.append(indptr[-1] + len(u))
    return sparse.csr_matrix((np.concatenate(data) if data else np.zeros(0, np.float32),
                              np.concatenate(indices) if indices else np.zeros(0, np.int64), np.array(indptr)),
                             shape=(len(token_ids), features))


def bm25_weights(counts, k1=1.2, b=0.75, max_df=0.5):
    n_docs = counts.shape[0]
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    idf[df > max_df * n_docs] = 0.0                          # boilerplate shared by most judgments
    dl = np.asarray(counts.sum(1)).ravel()
    avg = max(1e-9, dl.mean())
    w = counts.copy().astype(np.float32)
    rows = np.repeat(np.arange(n_docs), np.diff(w.indptr))
    tf = w.data
    w.data = (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / avg)) * idf[w.indices]).astype(np.float32)
    w.eliminate_zeros()
    return w


class SplitIndex:
    """Token ids of one split's queries and pool, and their n-gram matrices (built lazily, cached).
    query_text: optional {query id: text} replacing the query texts (e.g. citation contexts); view() makes another
    query view that shares the pool side (tokens, counts) with this one."""
    def __init__(self, queries, pool, query_text=None, _shared=None):
        self.query_ids = list(queries)
        self.gold = {q: queries[q]["relevant"] for q in self.query_ids if queries[q]["relevant"] is not None}
        if _shared is None:
            self.vocab = Vocab()
            self.pool_ids = list(pool)
            self.p_tokens = [self.vocab.encode(normalize(pool[c])) for c in self.pool_ids]
            self._counts = {}
        else:
            self.vocab, self.pool_ids, self.p_tokens, self._counts = _shared
        texts = query_text or {q: queries[q]["text"] for q in self.query_ids}
        self.q_tokens = [self.vocab.encode(normalize(texts[q])) for q in self.query_ids]
        self._queries = {}
        self._queries_src = queries

    def view(self, query_text):
        return SplitIndex(self._queries_src, None, query_text, (self.vocab, self.pool_ids, self.p_tokens, self._counts))

    def counts(self, n):
        if n not in self._counts:
            self._counts[n] = count_matrix(self.p_tokens, n)
        return self._counts[n]

    def query_matrix(self, n, qweight="binary"):
        """binary: each query n-gram once; tf: raw counts; logtf: 1 + log(count)."""
        key = (n, qweight)
        if key not in self._queries:
            m = count_matrix(self.q_tokens, n, binary=qweight == "binary")
            if qweight == "logtf":
                m.data = (1 + np.log(m.data)).astype(np.float32)
            self._queries[key] = m
        return self._queries[key]

    def bm25(self, n, k1=1.2, b=0.75, max_df=0.5, qweight="binary", prf=None):
        w = bm25_weights(self.counts(n), k1, b, max_df)
        q = self.query_matrix(n, qweight)
        s = np.asarray((q @ w.T).todense(), dtype=np.float32)
        if prf:
            s = self._prf(s, q, w, **prf)
        return s

    def passage_bm25(self, n, k1=1.2, b=0.75, max_df=0.5, qweight="binary", words=300, block=64):
        """Best-passage BM25: candidates cut into `words`-token passages (half-overlapping), BM25 over passages,
        a candidate scores its best passage (a quoted holding is local; whole-document BM25 dilutes it)."""
        key = ("passage", n, words)
        if key not in self._counts:
            spans, offsets = [], [0]
            step = max(1, words // 2)
            for t in self.p_tokens:
                starts = list(range(0, max(1, len(t) - words + step), step)) or [0]
                spans.extend(t[a:a + words] for a in starts)
                offsets.append(len(spans))
            self._counts[key] = (count_matrix(spans, n), np.array(offsets[:-1]))
        counts, starts = self._counts[key]
        w = bm25_weights(counts, k1, b, max_df).T.tocsc()
        q = self.query_matrix(n, qweight)
        out = np.zeros((q.shape[0], len(self.pool_ids)), dtype=np.float32)
        for a in range(0, q.shape[0], block):
            dense = np.asarray((q[a:a + block] @ w).todense(), dtype=np.float32)
            out[a:a + block] = np.maximum.reduceat(dense, starts, axis=1)
        return out

    def _prf(self, s, q, w, docs=5, terms=200, beta=0.5):
        """Pseudo-relevance feedback (RM3-style): add to each query the `terms` heaviest n-grams of its `docs`
        top-ranked candidates (mean BM25 weight, L1-normalised), weighted beta against the L1-normalised query."""
        index = {c: j for j, c in enumerate(self.pool_ids)}
        rows = []
        qn = sparse.diags(1 / np.maximum(np.asarray(q.sum(1)).ravel(), 1e-9)) @ q
        for i, qid in enumerate(self.query_ids):
            order = np.argsort(-s[i])
            top = [j for j in order[:docs + 1] if j != index.get(qid, -1)][:docs]
            fb = (sparse.csr_matrix(np.full((1, len(top)), 1.0 / len(top), dtype=np.float32)) @ w[top]).tocsr() \
                if top else sparse.csr_matrix((1, w.shape[1]), dtype=np.float32)
            if fb.nnz > terms:
                keep = np.argpartition(-fb.data, terms)[:terms]
                fb = sparse.csr_matrix((fb.data[keep], (np.zeros(terms, int), fb.indices[keep])), shape=fb.shape)
            total = fb.sum()
            rows.append(fb * (beta / total) if total > 0 else fb)
        expanded = qn + sparse.vstack(rows).tocsr()
        return np.asarray((expanded @ w.T).todense(), dtype=np.float32)


def evaluate_scores(scores, index):
    return f1_at_k(rank_from_scores(scores, index.query_ids, index.pool_ids), index.gold)["f1"]


def _ngram_scores(index, per_n, weights, prf=None):
    total = None
    for n, w in weights.items():
        if not w:
            continue
        p = per_n[n] if n in per_n else per_n[str(n)]
        z = zscore_rows(index.bm25(int(n), p["k1"], p["b"], p["max_df"], p.get("qweight", "binary"), prf)) * float(w)
        total = z if total is None else total + z
    return total


def _passage_scores(index, per_n, weights, words, context=None, context_views=None):
    """Best-passage composite over the fused n-gram sizes, with the same citation-context mixing as the documents."""
    def comp(ix):
        total = None
        for n, w in weights.items():
            if not w:
                continue
            p = per_n[n] if n in per_n else per_n[str(n)]
            z = zscore_rows(ix.passage_bm25(int(n), p["k1"], 0.75, p["max_df"], p.get("qweight", "binary"), words))
            total = z * float(w) if total is None else total + z * float(w)
        return total
    full = comp(index)
    if not context:
        return full
    view = (context_views or {}).get(context["window"]) or (context_views or {}).get(str(context["window"]))
    if view is None:
        from .data import query_texts
        view = index.view(query_texts(index._queries_src, int(context["window"])))
    return context["full"] * zscore_rows(full) + context["context"] * zscore_rows(comp(view))


def tune(index, grams=(1, 2, 3, 4, 5), k1s=(0.9, 1.2, 1.6, 2.0), bs=(0.4, 0.75, 0.9), max_dfs=(0.3, 0.5, 1.0),
         qweights=("binary", "logtf", "tf"), context_views=None, prf_grid=None, passage_words=None, log=print):
    """Per n: k1/b, max_df, query weighting. Then fusion weights over n (coordinate ascent over {0, .5, 1, 2}).
    context_views {window: SplitIndex view}: fuse full-query and citation-context scores (weights chosen too).
    prf_grid [(docs, terms, beta)]: pseudo-relevance feedback, kept only if it helps."""
    best_n = {}
    for n in grams:
        top = (-1.0, None)
        for k1, b in itertools.product(k1s, bs):
            f = evaluate_scores(index.bm25(n, k1, b, 0.5), index)
            if f > top[0]:
                top = (f, (k1, b))
        k1, b = top[1]
        cur = {"f1": top[0], "k1": k1, "b": b, "max_df": 0.5, "qweight": "binary"}
        for md in max_dfs:
            if md != 0.5:
                f = evaluate_scores(index.bm25(n, k1, b, md), index)
                if f > cur["f1"]:
                    cur.update(f1=f, max_df=md)
        for qw in qweights:
            if qw != "binary":
                f = evaluate_scores(index.bm25(n, k1, b, cur["max_df"], qw), index)
                if f > cur["f1"]:
                    cur.update(f1=f, qweight=qw)
        best_n[n] = cur
        log(f"  BM25 {n}-grams: F1@7 {cur['f1']:.4f} (k1 {k1}, b {b}, max_df {cur['max_df']}, query {cur['qweight']})")
    z = {n: zscore_rows(index.bm25(n, best_n[n]["k1"], best_n[n]["b"], best_n[n]["max_df"], best_n[n]["qweight"]))
         for n in grams}
    weights = {n: 0.0 for n in grams}
    weights[max(best_n, key=lambda n: best_n[n]["f1"])] = 1.0
    current = evaluate_scores(sum(z[n] * w for n, w in weights.items() if w), index)
    for _ in range(2):
        for n in grams:
            for w in (0.0, 0.5, 1.0, 2.0):
                trial = {**weights, n: w}
                if not any(trial.values()):
                    continue
                f = evaluate_scores(sum(z[m] * v for m, v in trial.items() if v), index)
                if f > current + 1e-9:
                    current, weights = f, trial
    log(f"  fused over n {weights}: F1@7 {current:.4f}")
    result = {"per_n": best_n, "weights": weights, "f1": current, "full_only_f1": current}
    full = sum(z[n] * w for n, w in weights.items() if w)
    if context_views:
        base = full
        best_ctx = None
        for window, view in context_views.items():
            zc = _ngram_scores(view, best_n, weights)
            alone = evaluate_scores(zc, view)
            log(f"  citation context +-{window} sentences alone: F1@7 {alone:.4f}")
            for full_w, ctx_w in ((0.0, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0)):
                f = evaluate_scores(full_w * zscore_rows(base) + ctx_w * zscore_rows(zc), index)
                if f > current + 1e-9:
                    current, best_ctx = f, {"window": window, "full": full_w, "context": ctx_w}
        result["context"] = best_ctx
        log(f"  with citation context {best_ctx}: F1@7 {current:.4f}")
    if passage_words:
        doc = FusedLexical({**result, "prf": None}).scores(index, context_views)
        best_pass = None
        for words in passage_words:
            ps = _passage_scores(index, best_n, weights, words, result.get("context"), context_views)
            log(f"  best passage of {words} tokens alone: F1@7 {evaluate_scores(ps, index):.4f}")
            for dw, pw in ((0.0, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0)):
                f = evaluate_scores(dw * zscore_rows(doc) + pw * zscore_rows(ps), index)
                if f > current + 1e-9:
                    current, best_pass = f, {"words": words, "doc": dw, "passage": pw}
        result["passage"] = best_pass
        log(f"  with best-passage BM25 {best_pass}: F1@7 {current:.4f}")
    if prf_grid:
        best_prf = None
        for docs, terms, beta in prf_grid:
            trial = {**result, "prf": {"docs": docs, "terms": terms, "beta": beta}, "f1": None}
            f = evaluate_scores(FusedLexical(trial).scores(index, context_views), index)
            log(f"  PRF docs {docs} terms {terms} beta {beta}: F1@7 {f:.4f}")
            if f > current + 1e-9:
                current, best_prf = f, {"docs": docs, "terms": terms, "beta": beta}
        result["prf"] = best_prf
    result["f1"] = current
    return result


class FusedLexical:
    """Applies a tune() result: per-n BM25 (+ PRF), fused over n; optionally fused with citation-context scores."""
    def __init__(self, tuned):
        self.tuned = tuned

    def scores(self, index, context_views=None):
        t = self.tuned
        prf = t.get("prf")
        full = _ngram_scores(index, t["per_n"], t["weights"], prf)
        ctx = t.get("context")
        doc = full if not ctx else None
        if ctx:
            window = ctx["window"]
            view = (context_views or {}).get(window) or (context_views or {}).get(str(window))
            if view is None:
                from .data import query_texts
                view = index.view(query_texts(index._queries_src, int(window)))
            zc = _ngram_scores(view, t["per_n"], t["weights"], prf)
            doc = ctx["full"] * zscore_rows(full) + ctx["context"] * zscore_rows(zc)
        pas = t.get("passage")
        if not pas:
            return doc.astype(np.float32)
        ps = _passage_scores(index, t["per_n"], t["weights"], pas["words"], ctx, context_views)
        return (pas["doc"] * zscore_rows(doc) + pas["passage"] * zscore_rows(ps)).astype(np.float32)

    def scores_split(self, queries, pool):
        return self.scores(SplitIndex(queries, pool))


# --------------------------------------------------------------------------- chunk-level TF-IDF (pair mining)
def chunk_tfidf(chunk_tokens, grams=(2, 3), idf=None):
    """L2-normalised hashed TF-IDF of word 2/3-grams per chunk (shared phrasing between passages)."""
    mats = [count_matrix(chunk_tokens, n, features=1 << 20) for n in grams]
    x = sparse.hstack(mats).tocsr() if len(mats) > 1 else mats[0]
    if idf is None:
        df = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float64)
        idf = np.log((1 + x.shape[0]) / (1 + df)) + 1
    x = _scale_cols(x, idf)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return sparse.diags(1 / np.maximum(norms, 1e-9)) @ x, idf


def _scale_cols(x, w):
    x = x.copy()
    x.data = x.data * w[x.indices]
    return x


# --------------------------------------------------------------------------- citations of similar labelled queries
def neighbor_citations(target, source, pool_ids, ks=(5, 10, 20, 50), grams=(2, 3), temperature=1.0):
    """{k: [n target, n pool]}: each candidate gets the (softmax-weighted) votes of the k labelled queries most
    similar to the target query (BM25 over word 2/3-grams of the full texts) that cited it. Similar judgments cite
    the same precedents, and the candidate pools overlap across splits."""
    src = {s: v for s, v in source.items() if v.get("relevant")}
    index = SplitIndex(target, {s: v["text"] for s, v in src.items()})
    sim = sum(zscore_rows(index.bm25(n)) for n in grams)
    col = {c: j for j, c in enumerate(pool_ids)}
    cited = [[col[c] for c in src[s]["relevant"] if c in col] for s in index.pool_ids]
    out = {}
    for k in ks:
        f = np.zeros((len(index.query_ids), len(pool_ids)), dtype=np.float32)
        for i, q in enumerate(index.query_ids):
            order = [j for j in np.argsort(-sim[i]) if index.pool_ids[j] != q][:k]
            if not order:
                continue
            w = np.exp((sim[i][order] - sim[i][order].max()) / temperature)
            for j, wj in zip(order, w):
                f[i, cited[j]] += wj
        out[k] = f
    return out, index.query_ids
