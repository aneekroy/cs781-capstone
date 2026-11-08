"""Phase-4 data: loading, the official score, and a submission writer that refuses anything the scorer would.

Layout (Phase_4_CJPE/): Train/train.json, Train/dev.json, Test/test.json. Train/dev records carry
`id`, `text`, `label` (0 = rejected, 1 = accepted; at least one appeal accepted counts as 1); test records
have no label. The loader accepts a list of records, an id-keyed object, or JSONL.
Submission: prediction.json = [{"id": ..., "pred": 0|1}, ...] for exactly the test ids, inside submission.zip.
Score: macro-F1 over the two classes (sklearn f1_score average="macro").
"""
import json
import re
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np


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
            if isinstance(value, dict):
                yield {"id": value.get("id", key), **value}
            else:
                yield {"id": key, "text": value}
        return
    yield from data


def _text(record):
    for key in ("text", "document", "judgment", "facts"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(x for x in value if isinstance(x, str))
    raise ValueError(f"record {record.get('id')}: no text field (keys {sorted(record)})")


def _label(record):
    for key in ("label", "labels", "decision", "y"):
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, list):
                value = value[0]
            if isinstance(value, str):
                value = value.strip().lower()
                value = {"accepted": 1, "accept": 1, "rejected": 0, "reject": 0}.get(value, value)
            value = int(value)
            if value not in (0, 1):
                raise ValueError(f"record {record.get('id')}: label {value} is not 0/1")
            return value
    return None


def split_path(data_dir, split):
    data_dir = Path(data_dir)
    folder = "Test" if split == "test" else "Train"
    for suffix in (".json", ".jsonl"):
        p = data_dir / folder / f"{split}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{data_dir}/{folder}/{split}.json")


LOAD_REPORT = {}
_TEST_INDEX = {}


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def test_overlap(data_dir, docs, min_share=0.5, min_shingles=10):
    """{id: (kind, test id)} for documents in `docs` that duplicate a TEST judgment: the same normalised text
    ("exact"), or >= min_share of sampled 8-grams shared in either direction ("near")."""
    import hashlib
    key = str(Path(data_dir).resolve())
    if key not in _TEST_INDEX:
        test = load_split(data_dir, "test", verbose=False)
        exact = {hashlib.md5(_norm(d["text"]).encode()).hexdigest(): t for t, d in test.items()}
        index, sizes = {}, {}
        for t, d in test.items():
            sh = _shingles(d["text"])
            sizes[t] = len(sh)
            for x in sh:
                index.setdefault(x, set()).add(t)
        _TEST_INDEX[key] = (exact, index, sizes)
    exact, index, sizes = _TEST_INDEX[key]
    out = {}
    for d, doc in docs.items():
        h = hashlib.md5(_norm(doc["text"]).encode()).hexdigest()
        if h in exact:
            out[d] = ("exact", exact[h])
            continue
        sh = _shingles(doc["text"])
        if not sh:
            continue
        hits = Counter(t for x in sh for t in index.get(x, ()))
        for t, c in hits.items():
            # a short document (a few boilerplate lines) shares phrases with many unrelated cases: "near" needs
            # min_shingles sampled phrases (~160+ words) on the side whose share is measured
            if (len(sh) >= min_shingles and c / len(sh) >= min_share) or \
                    (sizes[t] >= min_shingles and c / sizes[t] >= min_share):
                out[d] = ("near", t)
                break
    return out


def load_split(data_dir, split, verbose=True, exclude_test=True):
    """OrderedDict id -> {"text", "label" (None for test), "raw_id"}; ids are strings.

    ILDC repeats some ids. Train/dev: an exact repeat (same text, same label) is dropped; the same text with a
    different label keeps the first copy (counted as a conflict); a different judgment under a repeated id is
    kept as "<id>#<k>" (raw_id unchanged). Test: an exact repeat is merged (one prediction per id, as the
    scorer expects); two different test judgments under one id cannot both be submitted -> error."""
    docs = OrderedDict()
    seen_text = {}
    report = {"records": 0, "exact_repeats": 0, "label_conflicts": 0, "renamed": 0}
    for record in _records(split_path(data_dir, split)):
        report["records"] += 1
        raw = record.get("id", record.get("doc_id", report["records"] - 1))
        doc_id = str(raw)
        text, label = _text(record), _label(record)
        if doc_id in docs or doc_id in seen_text:
            same = [k for k in seen_text.get(doc_id, []) if docs[k]["text"] == text]
            if same:
                if docs[same[0]]["label"] == label:
                    report["exact_repeats"] += 1
                else:
                    report["label_conflicts"] += 1
                continue
            if split == "test":
                raise ValueError(f"test: id {doc_id} names two different judgments; a submission can hold only one")
            key = f"{doc_id}#{len(seen_text[doc_id]) + 1}"
            report["renamed"] += 1
        else:
            key = doc_id
        seen_text.setdefault(doc_id, []).append(key)
        docs[key] = {"text": text, "label": label, "raw_id": raw}
    if split != "test" and any(d["label"] is None for d in docs.values()):
        raise ValueError(f"{split}: records without a label")
    import os
    if split != "test" and exclude_test and os.environ.get("CJPE_KEEP_TEST_OVERLAP") != "1":
        # a labelled copy of a test judgment would train (or select thresholds/epochs/members) on the test answer
        overlap = test_overlap(data_dir, docs)
        for d in overlap:
            del docs[d]
        report["test_copies_dropped"] = len(overlap)
        report["test_copies_exact"] = sum(1 for k, _ in overlap.values() if k == "exact")
        if verbose and overlap:
            print(f"[{split}] dropped {len(overlap)} documents that duplicate a test judgment "
                  f"({report['test_copies_exact']} exact, {len(overlap) - report['test_copies_exact']} near) "
                  f"-> {len(docs)} left", flush=True)
    LOAD_REPORT[split] = report
    if verbose and (report["exact_repeats"] or report["label_conflicts"] or report["renamed"]):
        print(f"[{split}] {report['records']} records -> {len(docs)} documents: {report['exact_repeats']} exact repeats "
              f"dropped, {report['label_conflicts']} same-text label conflicts (first kept), {report['renamed']} "
              f"different judgments under a repeated id kept as <id>#k", flush=True)
    return docs


def macro_f1(y_true, y_pred):
    """sklearn f1_score(average="macro", labels=[0,1], zero_division=0)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in (0, 1):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def best_threshold(y_true, prob, grid=None):
    """Probability threshold maximising macro-F1 (ties -> closest to 0.5)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scored = [(macro_f1(y_true, (prob >= t).astype(int)), -abs(t - 0.5), t) for t in grid]
    best = max(scored)
    return float(best[2]), float(best[0])


def honest_threshold(y_true, prob, seed=0):
    """Choose the threshold on one half of the documents, score the other half, swap; pooled macro-F1."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_true))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    pred = np.zeros(len(y_true), dtype=int)
    for fit, ev in ((a, b), (b, a)):
        t, _ = best_threshold(y_true[fit], prob[fit])
        pred[ev] = (prob[ev] >= t).astype(int)
    return macro_f1(y_true, pred)


def validate_prediction(pred, test_docs):
    """pred: list of {"id", "pred"} (or {id: pred}); same id set as the test file, values 0/1."""
    if isinstance(pred, dict):
        pred = [{"id": k, "pred": v} for k, v in pred.items()]
    ids = [str(p["id"]) for p in pred]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in prediction")
    missing, extra = set(test_docs) - set(ids), set(ids) - set(test_docs)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={sorted(missing)[:5]} ({len(missing)}) extra={sorted(extra)[:5]} ({len(extra)})")
    bad = [p for p in pred if p["pred"] not in (0, 1) or isinstance(p["pred"], bool)]
    if bad:
        raise ValueError(f"predictions must be the integers 0/1: {bad[:3]}")
    return True


def write_prediction(pred, test_docs, out_dir, prob=None):
    """pred: {id: 0/1}. Writes prediction.json (ids as in the test file), submission.zip, a summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [{"id": test_docs[d]["raw_id"], "pred": int(pred[d])} for d in test_docs]
    validate_prediction([{"id": str(p["id"]), "pred": p["pred"]} for p in payload], test_docs)
    (out_dir / "prediction.json").write_text(json.dumps(payload))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", "prediction.json")
    counts = Counter(p["pred"] for p in payload)
    summary = {"documents": len(payload), "accepted_1": counts.get(1, 0), "rejected_0": counts.get(0, 0)}
    if prob is not None:
        (out_dir / "probabilities.json").write_text(json.dumps({d: round(float(prob[d]), 6) for d in test_docs}))
    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- hygiene
def _shingles(text, n=8, step=16, limit=400):
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    starts = list(range(0, max(0, len(words) - n + 1), step))[:limit]
    return {hash(tuple(words[i:i + n])) for i in starts}


def near_duplicates(docs_a, docs_b, min_share=0.5):
    """Pairs (a, b) whose sampled 8-gram shingles overlap by >= min_share of a's (train-test leakage check)."""
    index = {}
    for d, doc in docs_b.items():
        for s in _shingles(doc["text"]):
            index.setdefault(s, set()).add(d)
    out = []
    for d, doc in docs_a.items():
        sh = _shingles(doc["text"])
        if not sh:
            continue
        hits = Counter(x for s in sh for x in index.get(s, ()))
        for other, c in hits.items():
            if c / len(sh) >= min_share:
                out.append((d, other, round(c / len(sh), 3)))
    return out
