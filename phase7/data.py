"""Phase-7 data (Prior Case Retrieval, IL-PCR): queries, candidate pools, word chunks, Micro-F1@7, submission.

Layout: Phase_7_PCR/{Train,Dev,Test}/{split}.json (queries: id, text = list of sentences, relevant_candidates)
and {split}_candidates.json (the split's candidate pool). Each split's queries are ranked ONLY against that
split's pool. The loader accepts a list of records, an id-keyed object, or JSONL, and text as a list of
sentences or a string. A query that also appears in its own pool is never ranked against itself.

Micro-F1@7: over all queries, TP = sum |top7 & gold|, P = TP / (7 x queries), R = TP / sum |gold|.
Submission: prediction.json = {query id: [7 unique candidate ids from that split's pool]} in submission.zip.
"""
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

K = 7
FOLDERS = {"train": "Train", "dev": "Dev", "test": "Test"}
_TAG = re.compile(r"<[A-Za-z_]+>")          # <NAME>, <ORG>, <CITATION> ... anonymisation placeholders
_WORD = re.compile(r"[a-z0-9]+")


def _records(path):
    path = Path(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        out = []
        for key, value in data.items():
            if isinstance(value, dict):
                out.append({"id": value.get("id", key), **value})
            else:
                out.append({"id": key, "text": value})
        return out
    return data


def _text(r):
    for key in ("text", "document", "content", "sentences", "body"):
        v = r.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
    raise ValueError(f"record without text (keys {sorted(r)})")


def _id(r):
    for key in ("id", "query_id", "doc_id", "case_id", "candidate_id"):
        if key in r:
            return str(r[key])
    raise ValueError(f"record without id (keys {sorted(r)})")


def _path(data_dir, split, candidates):
    folder = Path(data_dir) / FOLDERS[split]
    name = f"{split}_candidates" if candidates else split
    for suffix in (".json", ".jsonl"):
        if (folder / f"{name}{suffix}").exists():
            return folder / f"{name}{suffix}"
    raise FileNotFoundError(folder / f"{name}.json")


def load_split(data_dir, split):
    """(queries {id: {"text", "relevant": [ids] | None}}, pool {id: text})."""
    queries = {}
    for r in _records(_path(data_dir, split, False)):
        rel = r.get("relevant_candidates", r.get("relevant", r.get("labels")))
        queries[_id(r)] = {"text": _text(r), "relevant": [str(x) for x in rel] if rel is not None else None}
    pool = {}
    for r in _records(_path(data_dir, split, True)):
        pool[_id(r)] = _text(r)
    return queries, pool


def stats(queries, pool):
    rel = [len(q["relevant"]) for q in queries.values() if q["relevant"] is not None]
    missing = sum(1 for q in queries.values() for c in (q["relevant"] or []) if c not in pool)
    self_in_pool = sum(1 for d in queries if d in pool)
    qwords = [len(q["text"].split()) for q in queries.values()]
    cwords = [len(t.split()) for t in pool.values()]
    marks = [citation_context(q["text"], 0)[1] for q in queries.values()]
    out = {"queries": len(queries), "pool": len(pool), "query_words_median": int(np.median(qwords)),
           "citation_marks_per_query_median": float(np.median(marks)),
           "queries_without_citation_marks": int(sum(m == 0 for m in marks)),
           "candidate_words_median": int(np.median(cwords)), "candidate_words_p90": int(np.percentile(cwords, 90)),
           "queries_in_own_pool": self_in_pool}
    if rel:
        cited = Counter(c for q in queries.values() for c in q["relevant"])
        out.update({"relevant_per_query_mean": round(float(np.mean(rel)), 2),
                    "relevant_per_query_median": float(np.median(rel)), "relevant_max": int(max(rel)),
                    "queries_with_more_than_7": int(sum(r > K for r in rel)),
                    "relevant_not_in_pool": missing,
                    "pool_never_relevant": int(sum(1 for c in pool if c not in cited)),
                    "most_cited_candidate_queries": cited.most_common(1)[0][1] if cited else 0,
                    "oracle_f1_at_7": f1_at_k({d: q["relevant"][:K] for d, q in queries.items()},
                                              {d: q["relevant"] for d, q in queries.items()})["f1"]})
    return out


# --------------------------------------------------------------------------- text
def normalize(text):
    """Lower-cased alphanumeric words, anonymisation placeholders removed (they would match everywhere)."""
    return _WORD.findall(_TAG.sub(" ", text).lower())


def word_chunks(text, chunk_words=200, max_chunks=64):
    """Raw-text chunks of chunk_words words (case kept, for the tokenizer). Documents with more chunks keep
    max_chunks evenly spaced ones, so the whole judgment is still represented."""
    w = _TAG.sub(" ", text).split()
    chunks = [" ".join(w[i:i + chunk_words]) for i in range(0, max(1, len(w)), chunk_words)] or [""]
    if max_chunks and len(chunks) > max_chunks:
        pick = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in sorted(set(pick.tolist()))]
    return chunks


CITE = re.compile(r"<\s*CITATION\s*>", re.I)


def sentences_of(text):
    """The split's sentences (texts arrive as sentence lists joined by newlines); long lines also at '. '."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(p for p in re.split(r"(?<=[.?!])\s+(?=[A-Z(\"'])", line) if p.strip())
    return out


def citation_context(text, window=1, pattern=CITE):
    """Only the sentences within `window` of a sentence that holds a citation mark (where the judgment discusses
    the cases it cites). Returns (context text, number of marks); the full text when there is no mark."""
    sents = sentences_of(text)
    hits = [i for i, s in enumerate(sents) if pattern.search(s)]
    if not hits:
        return text, 0
    keep = sorted({j for i in hits for j in range(max(0, i - window), min(len(sents), i + window + 1))})
    return "\n".join(sents[j] for j in keep), len(hits)


def query_texts(queries, window=None):
    """{id: text} of the queries, reduced to citation contexts when window is not None."""
    if window is None or window < 0:
        return {q: v["text"] for q, v in queries.items()}
    return {q: citation_context(v["text"], window)[0] for q, v in queries.items()}


_YEAR = re.compile(r"\b(18[5-9]\d|19\d\d|20[0-3]\d)\b")


def max_year(text):
    """Latest year mentioned (a judgment cannot mention years after its own decision); 0 when none."""
    years = [int(y) for y in _YEAR.findall(text)]
    return max(years) if years else 0


# --------------------------------------------------------------------------- ranking and metric
def rank_from_scores(scores, query_ids, pool_ids, k=K):
    """scores [nq, npool] -> {query id: top-k pool ids}; a query never retrieves itself."""
    index = {c: j for j, c in enumerate(pool_ids)}
    out = {}
    for i, q in enumerate(query_ids):
        s = np.asarray(scores[i], dtype=np.float64).copy()
        if q in index:
            s[index[q]] = -np.inf
        top = np.argpartition(-s, min(k, len(s) - 1))[:k + 1]
        top = top[np.argsort(-s[top], kind="stable")]
        out[q] = [pool_ids[j] for j in top if pool_ids[j] != q][:k]
    return out


def f1_at_k(ranked, gold, k=K):
    """Micro P/R/F1 @k over the queries in gold (the complete gold sets, not truncated)."""
    tp = n_pred = n_gold = 0
    per = []
    for q, rel in gold.items():
        rel = set(rel or [])
        got = list(ranked.get(q, []))[:k]
        hit = len(rel & set(got))
        tp, n_pred, n_gold = tp + hit, n_pred + k, n_gold + len(rel)
        per.append(hit)
    p, r = tp / max(1, n_pred), tp / max(1, n_gold)
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r, "tp": tp,
            "queries": len(gold), "queries_with_no_hit": int(sum(h == 0 for h in per))}


def recall_at(scores, query_ids, pool_ids, gold, ks=(7, 20, 50, 100)):
    """Recall@K for several K (how much a reranker could still recover)."""
    out = {}
    for k in ks:
        r = rank_from_scores(scores, query_ids, pool_ids, k)
        tp = sum(len(set(gold[q]) & set(r[q])) for q in query_ids if gold.get(q))
        out[f"recall@{k}"] = tp / max(1, sum(len(gold[q]) for q in query_ids if gold.get(q)))
    return out


def validate_prediction(pred, queries, pool, k=K):
    if not isinstance(pred, dict):
        raise ValueError("prediction must be an object {query id: [candidate ids]}")
    missing, extra = set(queries) - set(pred), set(pred) - set(queries)
    if missing or extra:
        raise ValueError(f"query ids differ from the split: {len(missing)} missing, {len(extra)} unknown")
    for q, ids in pred.items():
        if not isinstance(ids, list) or len(ids) != k:
            raise ValueError(f"{q}: needs exactly {k} candidate ids, got {len(ids) if isinstance(ids, list) else ids}")
        if len(set(ids)) != k:
            raise ValueError(f"{q}: duplicate candidate ids")
        bad = [c for c in ids if c not in pool]
        if bad:
            raise ValueError(f"{q}: ids outside the candidate pool, e.g. {bad[:2]}")
    return pred


def write_submission(ranked, queries, pool, out_dir, k=K):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = {q: [str(c) for c in ranked[q][:k]] for q in queries}
    validate_prediction(pred, queries, pool, k)
    (out_dir / "prediction.json").write_text(json.dumps(pred, indent=1))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", "prediction.json")
    return {"queries": len(pred)}


def zscore_rows(s):
    s = np.asarray(s, dtype=np.float64)
    mu, sd = s.mean(1, keepdims=True), s.std(1, keepdims=True)
    return (s - mu) / np.maximum(sd, 1e-9)
