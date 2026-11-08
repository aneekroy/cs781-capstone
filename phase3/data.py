"""Phase-3 data: loading both domains, the label vocabulary, the official pooled score, and a
prediction writer that refuses anything the competition scorer would reject.

Layout (Phase_3_RR/):
  Train/{CL,IT}_{train,dev}.json   list of {"id", "text": [sentences], "labels": [roles]}
  Train/label_vocab.json           {label: id} (13 labels)
  Test/{CL,IT}_test.json           {doc_id: [sentences]}

Scoring (evaluate_phase3.py): every sentence of every CL and IT document goes into ONE y_true/y_pred
list; macro-F1 over all 13 labels with zero_division=0, so a label with no gold sentence scores 0.
"""
import json
import re
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

DOMAINS = ("CL", "IT")
EXPECTED_LABELS = 13


# --------------------------------------------------------------------------- loading
def load_vocab(path_or_data):
    data = json.loads(Path(path_or_data).read_text()) if not isinstance(path_or_data, (list, dict)) else path_or_data
    if isinstance(data, list):
        labels = list(data)
    elif isinstance(data, dict) and all(isinstance(v, int) and not isinstance(v, bool) for v in data.values()):
        labels = [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
    elif isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        labels = [v for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    else:
        raise ValueError("label_vocab.json: expected a list, a label->id or an id->label mapping")
    if len(labels) != EXPECTED_LABELS or len(set(labels)) != len(labels):
        raise ValueError(f"expected {EXPECTED_LABELS} unique labels, got {labels}")
    return labels


def normalize_id(doc_id):
    doc_id = str(doc_id)
    return doc_id[:-4] if doc_id.endswith(".txt") else doc_id


def load_docs(path, domain):
    """OrderedDict id -> {"sentences", "labels" (None for test), "domain"} from any of the shipped formats."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    if isinstance(data, list):
        for row in data:
            sentences = row.get("text", row.get("sentences"))
            items.append((row["id"], sentences, row.get("labels")))
    elif isinstance(data, dict):
        for doc_id, value in data.items():
            if isinstance(value, dict):
                items.append((doc_id, value.get("text", value.get("sentences")), value.get("labels")))
            else:
                items.append((doc_id, value, None))
    else:
        raise ValueError(f"{path}: unsupported format")
    docs = OrderedDict()
    for doc_id, sentences, labels in items:
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError(f"{path}:{doc_id}: sentences must be a list of strings")
        if labels is not None and len(labels) != len(sentences):
            raise ValueError(f"{path}:{doc_id}: {len(sentences)} sentences but {len(labels)} labels")
        key = normalize_id(doc_id)
        if key in docs:
            raise ValueError(f"{path}: duplicate document id {key}")
        docs[key] = {"sentences": sentences, "labels": labels, "domain": domain, "raw_id": doc_id}
    return docs


def split_paths(data_dir, split):
    data_dir = Path(data_dir)
    folder = "Test" if split == "test" else "Train"
    return {d: data_dir / folder / f"{d}_{split}.json" for d in DOMAINS}


def load_split(data_dir, split, labels=None):
    """Both domains of one split in one dict. Cross-domain id collisions are fatal: the flat
    prediction.json could not hold both documents."""
    merged = OrderedDict()
    for domain, path in split_paths(data_dir, split).items():
        docs = load_docs(path, domain)
        clash = set(docs) & set(merged)
        if clash:
            raise ValueError(f"{split}: CL and IT share document ids {sorted(clash)[:5]} - raise this with the TAs")
        merged.update(docs)
    if labels is not None:
        allowed = set(labels)
        for doc_id, doc in merged.items():
            if doc["labels"] is not None:
                bad = set(doc["labels"]) - allowed
                if bad:
                    raise ValueError(f"{doc_id}: labels outside the vocabulary: {sorted(bad)}")
    return merged


def label_counts(docs):
    return Counter(l for d in docs.values() for l in (d["labels"] or ()))


# --------------------------------------------------------------------------- scoring
def per_label_scores(y_true, y_pred, labels):
    """Exactly sklearn's precision/recall/f1 with labels=labels, zero_division=0."""
    out = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        n_pred = sum(1 for p in y_pred if p == label)
        n_gold = sum(1 for t in y_true if t == label)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        f = 2 * tp / (n_pred + n_gold) if tp else 0.0
        out[label] = {"p": p, "r": r, "f1": f, "support": n_gold, "pred": n_pred}
    return out


def pooled_score(gold, pred, labels, docs=None):
    """Official Phase-3 number: one pooled list over every document of both domains."""
    y_true, y_pred = [], []
    for doc_id in (docs or gold):
        g, p = gold[doc_id], pred[doc_id]
        if len(g) != len(p):
            raise ValueError(f"{doc_id}: {len(g)} gold vs {len(p)} predicted sentences")
        y_true += list(g)
        y_pred += list(p)
    per = per_label_scores(y_true, y_pred, labels)
    k = len(labels)
    return {"macro_f1": sum(v["f1"] for v in per.values()) / k,
            "macro_p": sum(v["p"] for v in per.values()) / k,
            "macro_r": sum(v["r"] for v in per.values()) / k,
            "per_label": per, "sentences": len(y_true)}


class FastScorer:
    """Vectorized pooled macro-F1 for tuning loops: gold as label indices, predictions as index arrays."""
    def __init__(self, gold_idx, k):
        self.gold = np.asarray(gold_idx)
        self.k = k
        self.n_gold = np.bincount(self.gold, minlength=k)

    def __call__(self, pred_idx):
        pred_idx = np.asarray(pred_idx)
        tp = np.bincount(self.gold[pred_idx == self.gold], minlength=self.k)
        n_pred = np.bincount(pred_idx, minlength=self.k)
        denom = n_pred + self.n_gold
        f1 = np.where(tp > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        return float(f1.mean())


# --------------------------------------------------------------------------- output
def write_truth(docs, path):
    Path(path).write_text(json.dumps({k: d["labels"] for k, d in docs.items()}, ensure_ascii=False, indent=1))


def validate_prediction(pred, reference_docs, labels):
    """Every check the official scorer applies, against the test (or dev) documents."""
    allowed = set(labels)
    pred = {normalize_id(k): v for k, v in pred.items()}
    missing = sorted(set(reference_docs) - set(pred))
    extra = sorted(set(pred) - set(reference_docs))
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={missing[:5]} ({len(missing)}) extra={extra[:5]} ({len(extra)})")
    for doc_id, doc in reference_docs.items():
        labs = pred[doc_id]
        if not isinstance(labs, list) or len(labs) != len(doc["sentences"]):
            raise ValueError(f"{doc_id}: {len(doc['sentences'])} sentences but "
                             f"{len(labs) if isinstance(labs, list) else type(labs).__name__} predictions")
        bad = set(labs) - allowed
        if bad:
            raise ValueError(f"{doc_id}: unknown label strings {sorted(bad)}")
    return True


def write_prediction(pred, reference_docs, labels, out_dir):
    """prediction.json (keys = the raw test ids) + submission.zip, after validation."""
    validate_prediction(pred, reference_docs, labels)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {reference_docs[k]["raw_id"]: list(pred[k]) for k in reference_docs}
    (out_dir / "prediction.json").write_text(json.dumps(payload, ensure_ascii=False))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", "prediction.json")
    counts = Counter(l for v in payload.values() for l in v)
    summary = {"documents": len(payload), "sentences": sum(len(v) for v in payload.values()),
               "by_domain": dict(Counter(reference_docs[k]["domain"] for k in reference_docs)),
               "by_label": {l: counts.get(l, 0) for l in labels}}
    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- unlabeled text (KD)
_SENT = re.compile(r"(?<=[.?!])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text, min_chars=20):
    """Plain sentence split for teacher-labelled unlabeled judgments (never used on Phase-3 data)."""
    out = []
    for part in _SENT.split(" ".join(text.split())):
        part = part.strip()
        if len(part) >= min_chars:
            out.append(part)
    return out


_NORM = [(re.compile(r"\bno\s*\.\s*(?=\d)"), "number "), (re.compile(r"\bnos\s*\.\s*(?=\d)"), "numbers "),
         (re.compile(r"[^a-z0-9 ]+"), " ")]
GRAM, GRAM_STRIDE, MIN_HITS = 8, 4, 8


def _norm_words(text):
    text = text.lower()
    for rx, rep in _NORM:
        text = rx.sub(rep, text)
    return text.split()


class Fingerprints:
    """Word 8-grams of the Phase-3 documents after a normalisation that undoes the dataset's own edits
    ("No. 5" -> "Number 5", punctuation and apostrophes dropped), so a corpus judgment that IS a Phase-3
    document is found although the two texts are not byte-identical."""
    def __init__(self, texts):
        self.grams = set()
        for text in texts:
            w = _norm_words(text)
            self.grams.update(hash(tuple(w[i:i + GRAM])) for i in range(len(w) - GRAM + 1))

    def overlaps(self, text):
        w = _norm_words(text)
        hits = 0
        for i in range(0, len(w) - GRAM + 1, GRAM_STRIDE):
            if hash(tuple(w[i:i + GRAM])) in self.grams:
                hits += 1
                if hits >= MIN_HITS:
                    return True
        return False


def _records(path):
    """Raw records of any course json: a list, an id-keyed object, or JSONL (streamed with ijson when present,
    since the Phase-1 files are > 1 GB). Phase 4 (CJPE), 6 (Summ) and 7 (PCR) files work too."""
    path = Path(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        yield from (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        return
    try:
        import ijson
    except ImportError:
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else data.values())
        return
    with path.open("rb") as handle:
        first = b""
        while not first.strip():
            first = handle.read(1)
            if not first:
                return
        handle.seek(0)
        if first == b"[":
            yield from ijson.items(handle, "item")
        else:
            yield from (value for _, value in ijson.kvitems(handle, ""))


def record_text(record):
    """Judgment text of a record: Phase 1/4 {"text": str}, Phase 6 {"document": str} (never the summary),
    Phase 7 sentence lists ({"text": [...]}, {"sentences": [...]} or a bare list)."""
    if isinstance(record, str):
        return record
    if isinstance(record, list):
        return " ".join(x for x in record if isinstance(x, str))
    if isinstance(record, dict):
        for key in ("text", "document", "judgment", "sentences", "content"):
            value = record.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join(x for x in value if isinstance(x, str))
    return None


def unlabeled_documents(paths, n_docs, seed, phase3_texts, min_sents=20, max_sents=400, take_prob=0.1,
                        select=False, max_records=300000):
    """Judgments from the official Phase-1 corpus (one or more json files), sentence-split. A judgment that
    overlaps any Phase-3 train/dev/test document is dropped: test text never reaches training, and a
    validation-fold document never comes back as teacher-labelled text (that would leak into the student's CV).
    select=True keeps only judgments whose vocabulary marks them as competition-law or tax/banking
    (phase3.mixing.domain_of), tagged with that domain, at most half of n_docs per domain."""
    import random
    from .mixing import domain_of
    rng = random.Random(seed)
    fingerprints = Fingerprints(phase3_texts)
    docs, dropped, per_domain = OrderedDict(), 0, {}
    cap = (n_docs + 1) // 2
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        for i, record in enumerate(_records(path)):
            if len(docs) >= n_docs or i >= max_records:     # each file scanned up to max_records records
                break
            text = record_text(record)
            if not isinstance(text, str) or rng.random() > take_prob:
                continue
            domain = None
            if select:
                domain = domain_of(text)
                if domain is None or per_domain.get(domain, 0) >= cap:
                    continue
            sentences = split_sentences(text)
            if not (min_sents <= len(sentences) <= max_sents):
                continue
            if fingerprints.overlaps(text):
                dropped += 1
                continue
            key = f"unlabeled_{Path(path).stem}_{i}"
            docs[key] = {"sentences": sentences, "labels": None, "domain": domain, "raw_id": key}
            per_domain[domain] = per_domain.get(domain, 0) + 1
    return docs, dropped


def select_similar(candidates, reference_docs, n_docs):
    """Silver judgments ranked by TF-IDF cosine similarity to the CL / IT centroids of the reference (training)
    documents; the top n_docs/2 per domain are kept. Run per CV fold with that fold's training documents only,
    so a validation document's text never decides which silver judgments its fold model sees."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not candidates:
        return OrderedDict()
    cand_ids = list(candidates)
    cand_text = [" ".join(candidates[d]["sentences"]) for d in cand_ids]
    ref = [(doc["domain"], " ".join(doc["sentences"])) for doc in reference_docs.values()]
    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=2, stop_words="english")
    vec.fit(cand_text + [t for _, t in ref])
    x = vec.transform(cand_text)
    out, cap = OrderedDict(), (n_docs + 1) // 2
    centroids = {}
    for dom in DOMAINS:
        rows = [t for d, t in ref if d == dom]
        if rows:
            c = np.asarray(vec.transform(rows).mean(0)).ravel()
            centroids[dom] = c / (np.linalg.norm(c) + 1e-9)
    if not centroids:
        return OrderedDict()
    sims = {dom: x @ c for dom, c in centroids.items()}
    assigned = {}
    for i, d in enumerate(cand_ids):
        dom = candidates[d]["domain"] or max(sims, key=lambda k: sims[k][i])
        assigned.setdefault(dom, []).append((float(sims[dom][i]) if dom in sims else -1.0, d))
    for dom, rows in assigned.items():
        for _, d in sorted(rows, reverse=True)[:cap]:
            out[d] = {**candidates[d], "domain": dom}
    return out
