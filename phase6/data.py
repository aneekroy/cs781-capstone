"""Phase-6 data (abstractive legal summarization, IN-Abs style): loading, held-out split, sentences, the
summary-to-chunk alignment, chunking, length budgets, post-processing and the submission file.

Layout: Phase_6_Summ/Train/train.json ({"id", "document", "summary"}), Phase_6_Summ/Test/test.json ({"id",
"document"}). The loader also accepts an id-keyed object, JSONL and other common field names (text / judgment,
headnote / abstract). No dev split is published, so a fixed held-out set (hash of the id) is cut from train.

Long documents (IN-Abs: ~4.4k words) and long summaries (~1k words, ~24% of the document) do not fit one
Phase-1 context, so the document is cut into sentence-aligned chunks and every summary sentence is assigned to the
chunk holding its most similar document sentence (TF-IDF cosine; the divide-and-conquer recipe of Gidiotis &
Tsoumakas 2020 that Shukla et al. 2022 found best on IN-Abs). A chunk's training target is the summary sentences
assigned to it, in summary order. At inference each chunk gets a word budget (the predicted summary length split
over the chunks, uniformly by size or by the alignment density learned on train) and the chunk summaries are
joined in document order, de-duplicated and trimmed.

Submission: prediction.json = [{"id": ..., "summary": "..."}, ...], one non-empty summary per test document,
zipped as submission.zip.
"""
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- loading
DOC_KEYS = ("document", "text", "judgment", "judgement", "source", "doc", "input")
SUM_KEYS = ("summary", "headnote", "abstract", "target", "reference", "output")


def _records(path):
    path = Path(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        for line in path.open(encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            yield from data["data"]
            return
        for key, value in data.items():
            yield {"id": value.get("id", key), **value} if isinstance(value, dict) else {"id": key, "document": value}
        return
    yield from data


def _field(record, keys):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return "\n".join(value)
    return None


def split_path(data_dir, split):
    data_dir = Path(data_dir)
    folder = "Test" if split == "test" else "Train"
    for suffix in (".json", ".jsonl"):
        p = data_dir / folder / f"{split}{suffix}"
        if p.exists():
            return p
    found = sorted((data_dir / folder).glob("*.json*")) if (data_dir / folder).exists() else []
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"{data_dir}/{folder}/{split}.json")


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _shingles(text, n=8, step=16, limit=400):
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    starts = list(range(0, max(0, len(words) - n + 1), step))[:limit]
    return {hash(tuple(words[i:i + n])) for i in starts}


def load_raw(data_dir, split):
    """{id: {"document", "summary" or None}} in file order. Repeated ids: exact repeats dropped, else renamed."""
    out = {}
    for i, r in enumerate(_records(split_path(data_dir, split))):
        doc = _field(r, DOC_KEYS)
        if doc is None:
            raise ValueError(f"{split} record {i}: no document field (keys {sorted(r)})")
        did = str(r.get("id", r.get("doc_id", r.get("name", i))))
        summ = _field(r, SUM_KEYS)
        if split != "test" and not summ:
            continue                                            # nothing to learn from
        rec = {"document": doc, "summary": summ}
        if did in out:
            if out[did] == rec:
                continue
            if split == "test":
                raise ValueError(f"test id {did} appears twice with different documents")
            k = 2
            while f"{did}#{k}" in out:
                k += 1
            did = f"{did}#{k}"
        out[did] = rec
    return out


def test_overlap(train, test, min_share=0.5, min_shingles=10):
    """{train id: (kind, test id)} for training documents that duplicate a test document."""
    exact = {hashlib.md5(_norm(d["document"]).encode()).hexdigest(): t for t, d in test.items()}
    index, sizes = {}, {}
    for t, d in test.items():
        sh = _shingles(d["document"])
        sizes[t] = len(sh)
        for x in sh:
            index.setdefault(x, set()).add(t)
    out = {}
    for d, doc in train.items():
        h = hashlib.md5(_norm(doc["document"]).encode()).hexdigest()
        if h in exact:
            out[d] = ("exact", exact[h])
            continue
        sh = _shingles(doc["document"])
        if not sh:
            continue
        for t, c in Counter(t for x in sh for t in index.get(x, ())).items():
            if (len(sh) >= min_shingles and c / len(sh) >= min_share) or \
                    (sizes[t] >= min_shingles and c / sizes[t] >= min_share):
                out[d] = ("near", t)
                break
    return out


def is_dev(doc_id, dev_fraction):
    h = int(hashlib.md5(f"phase6-dev:{doc_id}".encode()).hexdigest()[:8], 16) / 16 ** 8
    return h < dev_fraction


def load_splits(data_dir, dev_size=200, verbose=True):
    """(train, dev, test): dev = a fixed hash-chosen held-out part of train (about dev_size documents).
    Training documents that copy a test document are dropped (PHASE6_KEEP_TEST_OVERLAP=1 keeps them)."""
    full = load_raw(data_dir, "train")
    test = load_raw(data_dir, "test")
    overlap = test_overlap(full, test)
    if overlap and os.environ.get("PHASE6_KEEP_TEST_OVERLAP") != "1":
        full = {d: r for d, r in full.items() if d not in overlap}
    frac = min(0.5, dev_size / max(1, len(full)))
    dev = {d: r for d, r in full.items() if is_dev(d, frac)}
    train = {d: r for d, r in full.items() if d not in dev}
    if verbose:
        exact = sum(1 for k, _ in overlap.values() if k == "exact")
        print(f"data: train {len(train)} / held-out dev {len(dev)} / test {len(test)}; "
              f"{len(overlap)} training documents copy a test document ({exact} exact)"
              + (" - dropped" if overlap and os.environ.get("PHASE6_KEEP_TEST_OVERLAP") != "1" else ""), flush=True)
    return train, dev, test


# --------------------------------------------------------------------------- sentences
_ABBR = {"v", "vs", "no", "nos", "s", "ss", "sec", "secs", "art", "arts", "cl", "cls", "para", "paras", "r", "rr",
         "o", "ch", "vol", "p", "pp", "j", "jj", "mr", "mrs", "ms", "dr", "ltd", "co", "corp", "inc", "pvt", "govt",
         "hon", "ors", "anr", "etc", "viz", "cf", "ibid", "st", "sri", "smt", "shri", "ex", "exh", "cr", "crl", "cri",
         "rs", "sch", "rule", "sub", "ib", "supra", "approx", "dt", "dated", "addl", "asstt", "dy", "spl", "jt"}
_BOUND = re.compile(r"[.?!][\"')\]]*\s+")


def split_sentences(text, max_words=60):
    """Legal-text sentence splitter: no split after abbreviations (v., No., s., Art., ...) or dotted initialisms
    (I.P.C.); sentences longer than max_words are cut at '; ' and then by length (alignment units stay short)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    out, start = [], 0
    for m in _BOUND.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'(["):
            continue
        word = text[start:m.start()].rsplit(" ", 1)[-1].strip("(\"'[").lower()
        if word in _ABBR or "." in word or "/" in word or (len(word) == 1 and word.isalpha()):
            continue
        piece = text[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    final = []
    for s in out:
        words = s.split()
        if len(words) <= max_words:
            final.append(s)
            continue
        parts, cur = [], []
        for piece in re.split(r"(?<=;)\s+", s):
            if cur and len(cur) + len(piece.split()) > max_words:
                parts.append(cur)
                cur = []
            cur += piece.split()
        if cur:
            parts.append(cur)
        for p in parts:
            for i in range(0, len(p), max_words * 2):
                final.append(" ".join(p[i:i + max_words * 2]))
    return final


def words(text):
    return len(text.split())


# --------------------------------------------------------------------------- alignment
def align(doc_sents, summ_sents, method="tfidf"):
    """For every summary sentence, the index of the document sentence it came from (most similar)."""
    n, m = len(doc_sents), len(summ_sents)
    if not n or not m:
        return [0] * m
    proportional = [min(n - 1, int((i + 0.5) / m * n)) for i in range(m)]
    if method == "position":
        return proportional
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b")
    try:
        x = vec.fit_transform(list(doc_sents) + list(summ_sents))
    except ValueError:                                          # empty vocabulary
        return proportional
    sim = (x[n:] @ x[:n].T).toarray()
    out = []
    for i, row in enumerate(sim):
        out.append(int(row.argmax()) if row.max() > 0 else proportional[i])
    return out


def oracle_labels(doc_sents, summ_sents, top=3):
    """Extractive oracle (Avr-style, Shukla et al. 2022): the `top` most similar document sentences of every summary
    sentence are labelled 1."""
    n = len(doc_sents)
    y = np.zeros(n, dtype=np.int8)
    if not n or not summ_sents:
        return y
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b")
    try:
        x = vec.fit_transform(list(doc_sents) + list(summ_sents))
    except ValueError:
        return y
    sim = (x[n:] @ x[:n].T).toarray()
    for row in sim:
        for j in np.argsort(-row)[:top]:
            if row[j] > 0:
                y[j] = 1
    return y


# --------------------------------------------------------------------------- chunks
def encode_sentences(tokenizer, sents):
    return [e.ids for e in tokenizer.encode_batch([" " + s for s in sents])] if sents else []


def make_chunks(sent_ids, chunk_tokens):
    """Pack sentences (token id lists) into chunks of at most chunk_tokens, never splitting a sentence unless it
    alone is longer than a chunk. Returns [{"sents": [i, ...], "ids": [...]}]."""
    chunks, cur, cur_ids = [], [], []
    for i, ids in enumerate(sent_ids):
        pieces = [ids[j:j + chunk_tokens] for j in range(0, len(ids), chunk_tokens)] or [[]]
        for piece in pieces:
            if cur_ids and len(cur_ids) + len(piece) > chunk_tokens:
                chunks.append({"sents": cur, "ids": cur_ids})
                cur, cur_ids = [], []
            if not cur or cur[-1] != i:
                cur.append(i)
            cur_ids = cur_ids + piece
    if cur_ids:
        chunks.append({"sents": cur, "ids": cur_ids})
    # a very small last chunk is merged into the previous one when that still fits
    if len(chunks) > 1 and len(chunks[-1]["ids"]) < chunk_tokens // 4 and \
            len(chunks[-2]["ids"]) + len(chunks[-1]["ids"]) <= chunk_tokens:
        last = chunks.pop()
        chunks[-1]["sents"] += [s for s in last["sents"] if s not in chunks[-1]["sents"]]
        chunks[-1]["ids"] += last["ids"]
    return chunks


def prepare_doc(tokenizer, text, chunk_tokens, overlap=0, keep=None):
    """Sentences, token ids and chunks of one document. keep: optional set of sentence indices (extract-then-
    abstract: the rest is dropped before chunking). overlap: tokens of the previous chunk prepended as context."""
    sents = split_sentences(text)
    if not sents:
        sents = [text.strip() or "."]
    ids = encode_sentences(tokenizer, sents)
    order = [i for i in range(len(sents)) if keep is None or i in keep] or list(range(len(sents)))
    chunks = make_chunks([ids[i] for i in order], chunk_tokens)
    total = sum(len(c["ids"]) for c in chunks)
    at = 0
    for k, c in enumerate(chunks):
        c["sents"] = [order[i] for i in c["sents"]]
        c["ctx"] = chunks[k - 1]["ids"][-overlap:] if overlap and k else []
        c["center"] = (at + len(c["ids"]) / 2) / max(1, total)
        at += len(c["ids"])
    return {"sents": sents, "ids": ids, "chunks": chunks, "words": words(text), "tokens": total}


def chunk_targets(doc, summary, method="tfidf"):
    """Summary sentences assigned to each chunk (lists of sentence strings, in summary order)."""
    summ_sents = split_sentences(summary)
    where = {}
    for k, c in enumerate(doc["chunks"]):
        for s in c["sents"]:
            where.setdefault(s, k)
    kept = sorted(where)
    assign = align([doc["sents"][i] for i in kept], summ_sents, method)
    out = [[] for _ in doc["chunks"]]
    for sent, j in zip(summ_sents, assign):
        out[where[kept[j]]].append(sent)
    return out


# --------------------------------------------------------------------------- length model and budgets
class LengthModel:
    """log(summary words) = a + b log(document words), fitted on train; prediction clamped to the train range."""
    def __init__(self, a=0.0, b=1.0, lo=50, hi=3000, r2=0.0, ratio=0.24):
        self.a, self.b, self.lo, self.hi, self.r2, self.ratio = a, b, lo, hi, r2, ratio

    @classmethod
    def fit(cls, docs):
        dw = np.array([max(1, words(d["document"])) for d in docs.values()], dtype=float)
        sw = np.array([max(1, words(d["summary"])) for d in docs.values()], dtype=float)
        x, y = np.log(dw), np.log(sw)
        b, a = np.polyfit(x, y, 1) if len(set(x)) > 1 else (1.0, float(np.mean(y - x)))
        r2 = 1 - float(np.var(y - (a + b * x))) / max(1e-9, float(np.var(y)))
        return cls(float(a), float(b), float(np.percentile(sw, 2)), float(np.percentile(sw, 98)), r2,
                   float(np.median(sw / dw)))

    def predict(self, doc_words, scale=1.0):
        w = math.exp(self.a + self.b * math.log(max(1, doc_words)))
        return int(round(scale * min(self.hi, max(self.lo, w))))

    def state(self):
        return dict(a=self.a, b=self.b, lo=self.lo, hi=self.hi, r2=self.r2, ratio=self.ratio)


def density_profile(prepared, targets, buckets=10):
    """Summary words aligned per document token, by relative position of the chunk (learned budget allocation)."""
    a, t = np.zeros(buckets), np.zeros(buckets)
    for d, doc in prepared.items():
        for c, tg in zip(doc["chunks"], targets[d]):
            b = min(buckets - 1, int(c["center"] * buckets))
            a[b] += sum(words(s) for s in tg)
            t[b] += len(c["ids"])
    dens = a / np.maximum(t, 1)
    dens[t == 0] = dens[t > 0].mean() if (t > 0).any() else 1.0
    return [float(x) for x in dens / max(1e-9, dens.mean())]


def budgets(doc, total_words, profile=None, floor=12):
    """Split total_words over the chunks: proportional to chunk tokens (uniform) or to tokens x learned density."""
    w = []
    for c in doc["chunks"]:
        dens = profile[min(len(profile) - 1, int(c["center"] * len(profile)))] if profile else 1.0
        w.append(len(c["ids"]) * dens)
    s = sum(w) or 1.0
    return [max(floor, int(round(total_words * x / s))) for x in w]


# --------------------------------------------------------------------------- post-processing
def _tri(s):
    t = re.findall(r"[a-z0-9]+", s.lower())
    return {tuple(t[i:i + 3]) for i in range(len(t) - 2)}


def clean_piece(text):
    """Whitespace, and an unfinished last sentence dropped when that keeps most of the text."""
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".?!\"')":
        cut = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
        if cut > 0.6 * len(text):
            text = text[:cut + 1]
        elif text:
            text += "."
    return text


def assemble(pieces, total_words, dedup=0.6, slack=1.05):
    """Join chunk summaries, drop sentences whose trigrams are mostly contained in earlier ones (dedup > 0),
    trim to slack x total_words at a sentence boundary."""
    seen, out, n = set(), [], 0
    limit = int(total_words * slack) if total_words else None
    for piece in pieces:
        for s in split_sentences(clean_piece(piece)):
            tri = _tri(s)
            if dedup and tri and len(tri & seen) / len(tri) >= dedup:
                continue
            w = words(s)
            if limit and n + w > limit and n >= 0.8 * total_words:
                return " ".join(out)
            out.append(s)
            seen |= tri
            n += w
    return " ".join(out)


def redundancy(text, n=3):
    """Fraction of repeated word n-grams (0 = none repeated)."""
    t = text.lower().split()
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def lead(text, total_words):
    out, n = [], 0
    for s in split_sentences(text):
        if n >= total_words:
            break
        out.append(s)
        n += words(s)
    return " ".join(out)


# --------------------------------------------------------------------------- submission
def validate_prediction(pred, test_docs):
    if isinstance(pred, dict):
        pred = [{"id": k, "summary": v} for k, v in pred.items()]
    if not isinstance(pred, list):
        raise ValueError("prediction must be a list of {id, summary} (or an id -> summary object)")
    ids = [str(p.get("id")) for p in pred]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise ValueError(f"duplicate ids: {dup[:5]}")
    missing, extra = set(test_docs) - set(ids), set(ids) - set(test_docs)
    if missing or extra:
        raise ValueError(f"ids differ from the test set: {len(missing)} missing, {len(extra)} unknown")
    empty = [p["id"] for p in pred if not isinstance(p.get("summary"), str) or not p["summary"].strip()]
    if empty:
        raise ValueError(f"{len(empty)} empty summaries, e.g. {empty[:3]}")
    return pred


def write_submission(summaries, test_docs, out_dir, fallback_words=None):
    """summaries {id: text} -> prediction.json + submission.zip (empty outputs replaced by the lead sentences)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, filled = [], 0
    for d, doc in test_docs.items():
        s = (summaries.get(d) or "").strip()
        if not s:
            s = lead(doc["document"], (fallback_words or {}).get(d, 300)) or doc["document"][:2000]
            filled += 1
        pred.append({"id": d, "summary": s})
    validate_prediction(pred, test_docs)
    (out_dir / "prediction.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", "prediction.json")
    return {"documents": len(pred), "filled_with_lead": filled,
            "words_mean": round(float(np.mean([words(p["summary"]) for p in pred])), 1)}
