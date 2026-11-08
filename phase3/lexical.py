"""Lexical member for the ensembles: TF-IDF + logistic regression with neighbour sentences and position.

  python -m phase3.lexical --data Phase_3_RR --output runsP3/L1_tfidf

No neural model and no pretraining: word 1-2 grams and character 3-5 grams of the sentence, the same features
of the previous and next sentence (down-weighted), relative-position bins and the domain. It is much weaker
than the taggers on its own, but its errors are lexical rather than contextual, which is what an averaging
or voting ensemble needs from an extra member. Same output files as a tagger run (oof.pt, test.pt,
cv_summary.json, submission/), so ensemble.py, vote.py and stack.py take it like any other run.
Out-of-fold: 5 document folds over train+dev, stratified by domain; the test model is fitted on all 90.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import DOMAINS, load_split, load_vocab, pooled_score, write_prediction
from .decode import apply_decoding, finalize_decoding, jsonable


def folds(docs, k, seed):
    rng = np.random.RandomState(seed)
    out = [[] for _ in range(k)]
    for dom in DOMAINS:
        ids = sorted(d for d, doc in docs.items() if doc["domain"] == dom)
        rng.shuffle(ids)
        for i, d in enumerate(ids):
            out[i % k].append(d)
    return out


def position_features(docs, bins=10):
    from scipy import sparse
    rows = []
    for doc in docs.values():
        n = len(doc["sentences"])
        for i in range(n):
            r = np.zeros(bins + 4)
            r[min(bins - 1, int(bins * i / n))] = 1
            r[bins] = float(doc["domain"] == "CL")
            r[bins + 1] = float(i == 0)
            r[bins + 2] = float(i == n - 1)
            r[bins + 3] = np.log1p(len(doc["sentences"][i].split())) / 5
            rows.append(r)
    return sparse.csr_matrix(np.array(rows))


class Featurizer:
    def __init__(self, neighbour_weight=0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True,
                                    lowercase=True, token_pattern=r"(?u)\b\w[\w.]*\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000,
                                    sublinear_tf=True)
        self.w = neighbour_weight

    def _text(self, docs):
        return [s for doc in docs.values() for s in doc["sentences"]]

    def fit(self, docs):
        text = self._text(docs)
        self.word.fit(text)
        self.char.fit(text)
        return self

    def transform(self, docs):
        from scipy import sparse
        text = self._text(docs)
        base = sparse.hstack([self.word.transform(text), self.char.transform(text)]).tocsr()
        prev_rows, next_rows, at = [], [], 0
        for doc in docs.values():
            n = len(doc["sentences"])
            idx = np.arange(at, at + n)
            prev_rows.append(np.r_[idx[:1], idx[:-1]])
            next_rows.append(np.r_[idx[1:], idx[-1:]])
            at += n
        prev = base[np.concatenate(prev_rows)] * self.w
        nxt = base[np.concatenate(next_rows)] * self.w
        return sparse.hstack([base, prev, nxt, position_features(docs)]).tocsr()


def fit_predict(train_docs, eval_docs, labels, c):
    from sklearn.linear_model import LogisticRegression
    index = {l: i for i, l in enumerate(labels)}
    feat = Featurizer().fit(train_docs)
    x = feat.transform(train_docs)
    y = np.array([index[l] for doc in train_docs.values() for l in doc["labels"]])
    model = LogisticRegression(C=c, max_iter=3000, solver="saga" if x.shape[0] > 200000 else "lbfgs")
    model.fit(x, y)
    p = np.full((sum(len(d["sentences"]) for d in eval_docs.values()), len(labels)), 1e-6)
    p[:, model.classes_] = model.predict_proba(feat.transform(eval_docs))
    p /= p.sum(1, keepdims=True)
    out, at = {}, 0
    for d, doc in eval_docs.items():
        n = len(doc["sentences"])
        lp = np.log(p[at:at + n])
        out[d] = {"emissions": lp, "logp": lp, "domain": doc["domain"], "crf": None}
        at += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=4.0)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    started = time.time()
    labels = load_vocab(Path(args.data) / "Train" / "label_vocab.json")
    docs = {d: doc for split in ("train", "dev") for d, doc in load_split(args.data, split, labels).items()}
    gold = {d: doc["labels"] for d, doc in docs.items()}
    oof = {}
    for i, val in enumerate(folds(docs, args.folds, args.seed)):
        tr = {d: docs[d] for d in docs if d not in set(val)}
        va = {d: docs[d] for d in val}
        for d, r in fit_predict(tr, va, labels, args.c).items():
            oof[d] = {**r, "fold": i}
        plain = pooled_score({d: gold[d] for d in va}, {d: [labels[j] for j in oof[d]["logp"].argmax(1)] for d in va},
                             labels, list(va))["macro_f1"]
        print(f"fold {i}: {len(va)} docs, argmax macro-F1 {plain:.4f}", flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"oof": oof, "labels": labels}, out / "oof.pt")
    config, _ = finalize_decoding(oof, gold, labels, docs, log=print, modes=["marginal"])
    decoded = apply_decoding(oof, config, labels, docs)
    score = pooled_score(gold, decoded, labels, list(gold))
    summary = {"method": "tfidf-logreg", "backbone": "none", "oof_macro_f1": round(score["macro_f1"], 5),
               "oof_macro_f1_honest": round(float(config["honest"]), 5), "decoding": jsonable(config),
               "oof_by_domain": {dom: round(pooled_score(gold, decoded, labels, [d for d in gold if docs[d]["domain"] == dom])
                                            ["macro_f1"], 4) for dom in DOMAINS},
               "oof_per_label": {l: round(v["f1"], 4) for l, v in score["per_label"].items()},
               "minutes": round((time.time() - started) / 60, 1)}
    print(f"OOF {score['macro_f1']:.4f} honest {config['honest']:.4f}", flush=True)
    if not args.no_test:
        test_docs = load_split(args.data, "test")
        test = fit_predict(docs, test_docs, labels, args.c)
        torch.save({"test": test, "labels": labels}, out / "test.pt")
        info = write_prediction(apply_decoding(test, config, labels, test_docs), test_docs, labels, out / "submission")
        summary["test_by_label"] = info["by_label"]
        print(f"submission: {out / 'submission' / 'submission.zip'}", flush=True)
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
