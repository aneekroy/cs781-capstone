"""Phase-5 data (IL-TUR Legal Statute Identification): loading, the official metrics, thresholds, submissions.

Layout: Phase_5_LSI/Train/{train,dev}.json, Phase_5_LSI/Test/test.json. A record is
{"id": ..., "text": ..., "labels": [statute ids]} (test: no labels). The loader also accepts an id-keyed object,
JSONL, text given as a list of sentences, and labels given as strings of integers.

Submission: prediction.json = [{"id": ..., "pred": [ids]}, ...], one object per test document, integer ids,
no duplicates, only ids of the label space; zipped as submission.zip.
Scores: Macro-F1 (primary), Micro-F1, subset accuracy over the label space (sklearn, zero_division=0).
Macro-F1 is the mean of per-statute F1, so it decomposes by label: a per-statute threshold that maximises that
statute's F1 maximises Macro-F1 on the documents it was tuned on. Rare statutes have few dev positives, so
per-label thresholds are shrunk toward the global one and the whole choice is made with an honest split-half check.
"""
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- loading
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
            yield {"id": value.get("id", key), **value} if isinstance(value, dict) else {"id": key, "text": value}
        return
    yield from data


def _text(record):
    for key in ("text", "facts", "document", "doc"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(x for x in value if isinstance(x, str))
    raise ValueError(f"record {record.get('id')}: no text field (keys {sorted(record)})")


def _labels(record):
    for key in ("labels", "label", "statutes", "y"):
        if key in record and record[key] is not None:
            value = record[key]
            if not isinstance(value, (list, tuple)):
                value = [value]
            out = []
            for v in value:
                if isinstance(v, str):
                    v = v.strip()
                    if not re.fullmatch(r"-?\d+", v):
                        raise ValueError(f"record {record.get('id')}: statute id {v!r} is not an integer")
                out.append(int(v))
            return sorted(set(out))
    return None


def split_path(data_dir, split):
    data_dir = Path(data_dir)
    folder = "Test" if split == "test" else "Train"
    for suffix in (".json", ".jsonl"):
        p = data_dir / folder / f"{split}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{data_dir}/{folder}/{split}.json")


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _shingles(text, n=8, step=16, limit=400):
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    starts = list(range(0, max(0, len(words) - n + 1), step))[:limit]
    return {hash(tuple(words[i:i + n])) for i in starts}


LOAD_REPORT = {}
_TEST_INDEX = {}


def test_overlap(data_dir, docs, min_share=0.5, min_shingles=10):
    """{id: (kind, test id)}: documents duplicating a TEST document (same normalised text = "exact", or >= min_share
    of sampled 8-grams shared in either direction = "near")."""
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
    """OrderedDict id -> {"text", "labels" (sorted int list; None for test), "raw_id"}.
    Repeated ids: exact repeats dropped; same text with other labels -> labels merged (union, counted);
    a different document under a repeated id kept as "<id>#k". Test: exact repeats merged, two different test
    documents under one id -> error (a submission holds one prediction per id).
    Train/dev documents that duplicate a test document are dropped (they would carry the test answer into
    training or into threshold / epoch / member selection); LSI_KEEP_TEST_OVERLAP=1 keeps them."""
    docs, seen = OrderedDict(), {}
    report = {"records": 0, "exact_repeats": 0, "label_merges": 0, "renamed": 0}
    for record in _records(split_path(data_dir, split)):
        report["records"] += 1
        raw = record.get("id", record.get("case_id", report["records"] - 1))
        doc_id = str(raw)
        text, labels = _text(record), _labels(record)
        if doc_id in seen:
            same = [k for k in seen[doc_id] if docs[k]["text"] == text]
            if same:
                k = same[0]
                if labels is not None and docs[k]["labels"] != labels:
                    docs[k]["labels"] = sorted(set(docs[k]["labels"] or []) | set(labels))
                    report["label_merges"] += 1
                else:
                    report["exact_repeats"] += 1
                continue
            if split == "test":
                raise ValueError(f"test: id {doc_id} names two different documents; a submission can hold only one")
            key = f"{doc_id}#{len(seen[doc_id]) + 1}"
            report["renamed"] += 1
        else:
            key = doc_id
        seen.setdefault(doc_id, []).append(key)
        docs[key] = {"text": text, "labels": labels, "raw_id": raw}
    if split != "test":
        missing = [d for d, doc in docs.items() if not doc["labels"]]
        if missing:
            report["no_label_dropped"] = len(missing)
            for d in missing:
                del docs[d]
    if split != "test" and exclude_test and os.environ.get("LSI_KEEP_TEST_OVERLAP", os.environ.get("CJPE_KEEP_TEST_OVERLAP")) != "1":
        overlap = test_overlap(data_dir, docs)
        for d in overlap:
            del docs[d]
        report["test_copies_dropped"] = len(overlap)
    LOAD_REPORT[split] = report
    if verbose and any(report.get(k) for k in ("exact_repeats", "label_merges", "renamed", "test_copies_dropped",
                                               "no_label_dropped")):
        print(f"[{split}] {report['records']} records -> {len(docs)} documents: {report}", flush=True)
    return docs


def label_space(*splits):
    """Sorted statute ids seen in the labelled splits."""
    return sorted({l for docs in splits for d in docs.values() for l in (d["labels"] or [])})


def multi_hot(docs, labels):
    index = {l: i for i, l in enumerate(labels)}
    y = np.zeros((len(docs), len(labels)), dtype=np.int8)
    for r, d in enumerate(docs.values()):
        for l in d["labels"] or []:
            if l in index:
                y[r, index[l]] = 1
    return y


# --------------------------------------------------------------------------- metrics
def scores(y_true, y_pred):
    """y_true, y_pred: [n, L] 0/1. Macro-F1 (mean per-label F1, 0 for a label with no true and no predicted
    positive, as sklearn zero_division=0), Micro-F1, subset accuracy, per-label F1."""
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    tp = (y_true & y_pred).sum(0)
    fp = (~y_true & y_pred).sum(0)
    fn = (y_true & ~y_pred).sum(0)
    denom = 2 * tp + fp + fn
    per = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    micro_d = 2 * tp.sum() + fp.sum() + fn.sum()
    return {"macro_f1": float(per.mean()) if per.size else 0.0,
            "micro_f1": float(2 * tp.sum() / micro_d) if micro_d else 0.0,
            "subset_accuracy": float((y_true == y_pred).all(1).mean()) if len(y_true) else 0.0,
            "per_label_f1": per}


def per_label_f1_curve(y, p, grid):
    """F1 of one label for every threshold of the grid (vectorised). y [n] bool, p [n]."""
    pred = p[None, :] >= grid[:, None]
    tp = (pred & y[None, :]).sum(1)
    fp = (pred & ~y[None, :]).sum(1)
    fn = (~pred & y[None, :]).sum(1)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


GRID = np.round(np.concatenate([np.arange(0.02, 0.1, 0.01), np.arange(0.1, 0.951, 0.025)]), 3)


def apply_thresholds(p, t, min_one=False):
    """p [n, L] probabilities, t scalar or [L] -> 0/1 predictions; min_one: every document keeps its top statute."""
    pred = p >= np.asarray(t)[None, ...] if np.ndim(t) else p >= t
    if min_one:
        empty = ~pred.any(1)
        if empty.any():
            pred[np.where(empty)[0], p[empty].argmax(1)] = True
    return pred.astype(np.int8)


def fit_thresholds(y, p, kind="per_label", shrink=10.0, grid=GRID, min_one=False):
    """Thresholds maximising Macro-F1 on (y, p). kind: 'global' (one value) or 'per_label' (each statute its own,
    a weighted mean with the global value: w = n_pos / (n_pos + shrink), so rare statutes stay near global)."""
    y = np.asarray(y, dtype=bool)
    best_g, best_s = 0.5, -1.0
    for t in grid:
        s = scores(y, apply_thresholds(p, t, min_one))["macro_f1"]
        if s > best_s + 1e-12:
            best_g, best_s = float(t), s
    if kind == "global":
        return best_g
    out = np.full(y.shape[1], best_g)
    for j in range(y.shape[1]):
        n_pos = int(y[:, j].sum())
        if n_pos == 0:
            continue
        curve = per_label_f1_curve(y[:, j], p[:, j], grid)
        tj = float(grid[int(np.argmax(curve))])
        w = n_pos / (n_pos + shrink) if shrink else 1.0
        out[j] = w * tj + (1 - w) * best_g
    return out


def halves(n, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2:]


def choose_thresholds(y, p, log=print, margin=0.002):
    """Pick the thresholding scheme by the honest split-half Macro-F1 (tune on one half, score the other, swap,
    pool). Candidates: global 0.5, tuned global, per-label with shrinkage 3 / 10 / 30, each with and without
    the min-one rule. A tuned scheme must beat the plain 0.5 by `margin`. Returns (config, thresholds fit on
    all of y, table of honest scores)."""
    y = np.asarray(y, dtype=bool)
    a, b = halves(len(y))
    table = []
    candidates = [("fixed", None), ("global", None)] + [("per_label", s) for s in (3.0, 10.0, 30.0)]
    for min_one in (False, True):
        for kind, shrink in candidates:
            pred = np.zeros_like(y, dtype=np.int8)
            for fit, ev in ((a, b), (b, a)):
                t = 0.5 if kind == "fixed" else fit_thresholds(y[fit], p[fit], kind, shrink or 10.0, min_one=min_one)
                pred[ev] = apply_thresholds(p[ev], t, min_one)
            s = scores(y, pred)
            table.append({"kind": kind, "shrink": shrink, "min_one": min_one, "honest_macro_f1": round(s["macro_f1"], 5),
                          "honest_micro_f1": round(s["micro_f1"], 5), "honest_subset_acc": round(s["subset_accuracy"], 5)})
    base = next(r for r in table if r["kind"] == "fixed" and not r["min_one"])
    best = max(table, key=lambda r: r["honest_macro_f1"])
    if best is not base and best["honest_macro_f1"] < base["honest_macro_f1"] + margin:
        best = base
    t = 0.5 if best["kind"] == "fixed" else fit_thresholds(y, p, best["kind"], best["shrink"] or 10.0,
                                                           min_one=best["min_one"])
    for r in table:
        log(f"  thresholds {r['kind']:9s} shrink {str(r['shrink']):5s} min_one {str(r['min_one']):5s} -> honest "
            f"macro-F1 {r['honest_macro_f1']:.4f} micro {r['honest_micro_f1']:.4f} subset {r['honest_subset_acc']:.4f}")
    return best, t, table


# --------------------------------------------------------------------------- submissions
def validate_prediction(pred, test_docs, labels):
    """pred: list of {"id", "pred": [ints]} -> True or ValueError (ids, types, duplicates, label space)."""
    if isinstance(pred, dict):
        pred = [{"id": k, "pred": v} for k, v in pred.items()]
    ids = [str(p["id"]) for p in pred]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in prediction")
    expected = {str(d["raw_id"]) for d in test_docs.values()}
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={sorted(missing)[:5]} ({len(missing)}) extra={sorted(extra)[:5]} ({len(extra)})")
    valid = set(labels)
    for p in pred:
        v = p["pred"]
        if not isinstance(v, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in v):
            raise ValueError(f"{p['id']}: pred must be a list of integers, got {v!r}")
        if len(v) != len(set(v)):
            raise ValueError(f"{p['id']}: duplicate statute ids {v}")
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"{p['id']}: statute ids outside the label space {bad}")
    return True


def write_prediction(pred_matrix, test_docs, labels, out_dir):
    """pred_matrix [n_test, L] 0/1 in test_docs order -> prediction.json + submission.zip + summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for row, doc in zip(np.asarray(pred_matrix), test_docs.values()):
        payload.append({"id": doc["raw_id"], "pred": [int(labels[j]) for j in np.flatnonzero(row)]})
    validate_prediction(payload, test_docs, labels)
    (out_dir / "prediction.json").write_text(json.dumps(payload))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", "prediction.json")
    sizes = [len(p["pred"]) for p in payload]
    info = {"documents": len(payload), "labels_per_doc": round(float(np.mean(sizes)), 3) if sizes else 0,
            "empty": int(sum(1 for s in sizes if s == 0)),
            "distinct_statutes_predicted": len({l for p in payload for l in p["pred"]})}
    (out_dir / "prediction_summary.json").write_text(json.dumps(info, indent=1))
    return info


def save_probs(path, dev_ids, dev_p, test_ids, test_p, labels):
    np.savez_compressed(path, dev_ids=np.array(dev_ids, dtype=object), dev=dev_p.astype(np.float32),
                        test_ids=np.array(test_ids, dtype=object), test=test_p.astype(np.float32),
                        labels=np.array(labels))


def load_probs(path):
    z = np.load(path, allow_pickle=True)
    return {"dev_ids": list(z["dev_ids"]), "dev": z["dev"], "test_ids": list(z["test_ids"]), "test": z["test"],
            "labels": [int(x) for x in z["labels"]]}


def finish(y_dev, p_dev, p_test, dev, test, labels, out, summary, log=print):
    """Threshold choice (honest), dev scores, test submission, probs.npz, summary.json."""
    out = Path(out)
    config, t, table = choose_thresholds(y_dev, p_dev, log=log)
    dev_pred = apply_thresholds(p_dev, t, config["min_one"])
    s = scores(y_dev, dev_pred)
    s05 = scores(y_dev, apply_thresholds(p_dev, 0.5))
    info = write_prediction(apply_thresholds(p_test, t, config["min_one"]), test, labels, out / "submission")
    save_probs(out / "probs.npz", list(dev), p_dev, list(test), p_test, labels)
    support = np.asarray(y_dev).sum(0)
    rare = support <= np.percentile(support, 25)
    summary.update({"dev_macro_f1_at_0.5": round(s05["macro_f1"], 5), "dev_macro_f1_fit": round(s["macro_f1"], 5),
                    "dev_macro_f1_honest": config["honest_macro_f1"], "dev_micro_f1_honest": config["honest_micro_f1"],
                    "dev_subset_acc_honest": config["honest_subset_acc"], "thresholds": config,
                    "threshold_table": table, "dev_macro_f1_rare_quartile_fit": round(float(s["per_label_f1"][rare].mean()), 4),
                    "dev_per_label_f1": {int(l): round(float(f), 4) for l, f in zip(labels, s["per_label_f1"])},
                    "test": info})
    if isinstance(t, np.ndarray):
        np.save(out / "thresholds.npy", t)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"dev macro-F1 {s05['macro_f1']:.4f} @0.5 | honest {config['honest_macro_f1']:.4f} ({config['kind']}"
        f"{'' if config['shrink'] is None else ' shrink ' + str(config['shrink'])}{' +min_one' if config['min_one'] else ''}) "
        f"micro {config['honest_micro_f1']:.4f} | submission: {out / 'submission' / 'submission.zip'} {info}")
    return summary
