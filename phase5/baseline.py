"""Stage 0: TF-IDF + one-vs-rest logistic regression (one binary classifier per statute; no neural model).

The facts of a criminal case name what happened ("armed with lathis", "dowry", "succumbed to injuries"), and
each IPC section has its own vocabulary, so a linear model over word 1-2-grams is a strong reference and a
different kind of ensemble member. Statute citations are masked in this dataset, so it cannot just read them.
class_weight="balanced" per statute (rare statutes matter for Macro-F1); C chosen on dev Macro-F1 after
thresholding; thresholds chosen with the honest split-half check (phase5.data.choose_thresholds).
"""
import json
import time
from pathlib import Path

import numpy as np

from .data import choose_thresholds, finish, label_space, load_split, multi_hot


def run(args):
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    if args.max_train_docs:
        train = dict(list(train.items())[:args.max_train_docs])
    labels = label_space(train, dev)
    y_train, y_dev = multi_hot(train, labels), multi_hot(dev, labels)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=args.max_features,
                          sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b\w[\w'-]*\b")
    words = lambda docs: [" ".join(d["text"].split()[: args.max_words]) for d in docs.values()]
    x_train = vec.fit_transform(words(train))
    x_dev, x_test = vec.transform(words(dev)), vec.transform(words(test))
    # max_features pruning leaves the column indices unsorted; every LogisticRegression.fit would sort them IN
    # PLACE, and with thread workers sharing one matrix that corrupts it ("indices array has fewer than nnz
    # elements", then a segfault). Make all three canonical once, before any thread touches them.
    for x in (x_train, x_dev, x_test):
        x.sum_duplicates()
        x.sort_indices()
    assert x_train.has_canonical_format and x_dev.has_canonical_format and x_test.has_canonical_format
    print(f"TF-IDF: {x_train.shape[1]} features, {len(train)} train / {len(dev)} dev / {len(test)} test, "
          f"{len(labels)} statutes ({time.time() - started:.0f}s)", flush=True)

    def fit_one(j, c):
        yj = y_train[:, j]
        if yj.min() == yj.max():                       # statute never (or always) present in train
            return None
        clf = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(x_train, yj)
        return clf

    def probs(models, x):
        p = np.zeros((x.shape[0], len(models)), dtype=np.float32)
        for j, m in enumerate(models):
            if m is not None:
                p[:, j] = m.predict_proba(x)[:, 1]
        return p

    best = None
    for c in args.c_grid:
        models = Parallel(n_jobs=args.jobs, prefer="threads")(   # threads share x_train; process workers get a read-only memmap that liblinear rejects
            delayed(fit_one)(j, c) for j in range(len(labels)))
        p_dev = probs(models, x_dev)
        cfg, _, _ = choose_thresholds(y_dev, p_dev, log=lambda *a: None)
        print(f"C={c:<6} dev honest macro-F1 {cfg['honest_macro_f1']:.4f} ({cfg['kind']}) "
              f"[{time.time() - started:.0f}s]", flush=True)
        if best is None or cfg["honest_macro_f1"] > best[0]:
            best = (cfg["honest_macro_f1"], c, models, p_dev)
    _, c, models, p_dev = best
    p_test = probs(models, x_test)
    finish(y_dev, p_dev, p_test, dev, test, labels, out,
           {"stage": "baseline_tfidf_ovr", "C": c, "features": int(x_train.shape[1]), "train_docs": len(train),
            "statutes": len(labels), "minutes": round((time.time() - started) / 60, 1)})
