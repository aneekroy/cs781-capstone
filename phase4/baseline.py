"""Stage 0: TF-IDF + logistic regression on the decision-bearing parts of the judgment (no neural model).

Judgments state the outcome's reasoning near the end, so the features come from the LAST --tail-words words
(plus the first --head-words, which name the appeal and the parties). Word 1-2-grams, sublinear tf. The
regularisation C is chosen on dev, the decision threshold on dev with the honest half-swap check.
A fast, strong reference point, and a useful, different member for the final ensemble.
"""
import json
import time
from pathlib import Path

import numpy as np

from .data import best_threshold, honest_threshold, load_split, macro_f1, write_prediction


def excerpt(text, head_words, tail_words):
    words = text.split()
    if len(words) <= head_words + tail_words:
        return " ".join(words)
    return " ".join(words[:head_words] + ["[...]"] + words[-tail_words:])


def run(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    if args.max_train_docs:
        train = dict(list(train.items())[:args.max_train_docs])
    texts = {s: [excerpt(d["text"], args.head_words, args.tail_words) for d in docs.values()]
             for s, docs in (("train", train), ("dev", dev), ("test", test))}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=args.max_features,
                          sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b\w[\w'-]*\b")
    x_train = vec.fit_transform(texts["train"])
    x_dev, x_test = vec.transform(texts["dev"]), vec.transform(texts["test"])
    y_train = np.array([d["label"] for d in train.values()])
    y_dev = np.array([d["label"] for d in dev.values()])
    best = None
    for c in args.c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="liblinear")
        clf.fit(x_train, y_train)
        p = clf.predict_proba(x_dev)[:, 1]
        f1_05 = macro_f1(y_dev, (p >= 0.5).astype(int))
        print(f"C={c:<6} dev macro-F1 @0.5 {f1_05:.4f}", flush=True)
        if best is None or f1_05 > best[0]:
            best = (f1_05, c, clf, p)
    f1_05, c, clf, p_dev = best
    threshold, f1_tuned = best_threshold(y_dev, p_dev)
    honest = honest_threshold(y_dev, p_dev)
    use = honest > f1_05 + 0.003
    t = threshold if use else 0.5
    p_test = clf.predict_proba(x_test)[:, 1]
    probs = {"dev": dict(zip(dev, map(float, p_dev))), "test": dict(zip(test, map(float, p_test)))}
    (out / "probs.json").write_text(json.dumps(probs))
    info = write_prediction({d: int(p >= t) for d, p in probs["test"].items()}, test, out / "submission",
                            probs["test"])
    summary = {"stage": "baseline_tfidf", "C": c, "dev_macro_f1_at_0.5": round(f1_05, 5),
               "dev_macro_f1_tuned": round(f1_tuned, 5), "dev_macro_f1_honest": round(honest if use else f1_05, 5),
               "threshold": round(t, 3), "features": len(vec.vocabulary_), "train_docs": len(train),
               "minutes": round((time.time() - started) / 60, 1), "test": info}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return summary
