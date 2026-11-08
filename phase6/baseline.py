"""Non-neural references for Phase 6 (CPU), and the sentence-salience model used for extract-then-abstract.

  lead        the first sentences of the judgment up to the predicted summary length
  extractive  supervised sentence scorer (TF-IDF 1-2-grams + relative-position and length features, logistic
              loss), trained on extractive oracle labels derived from the abstractive summaries (each summary
              sentence marks its 3 most similar document sentences: the "Avr" conversion of Shukla et al. 2022);
              the summary is the top-scored sentences up to the predicted length, in document order.
The fitted scorer (extractor.pkl) is reused by the neural runs with --strategy extract (importance-based
context selection: only the most salient part of a long judgment is chunked and summarized).
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .data import LengthModel, lead, load_splits, oracle_labels, split_sentences, write_submission
from .evaluate import evaluate

BUCKETS = 10


class Extractor:
    def __init__(self, vec, clf):
        self.vec, self.clf = vec, clf

    @staticmethod
    def _extra(sents):
        from scipy import sparse
        n = len(sents)
        rows = []
        for i, s in enumerate(sents):
            pos = np.zeros(BUCKETS + 4, dtype=np.float32)
            pos[min(BUCKETS - 1, int(i / max(1, n) * BUCKETS))] = 1
            w = len(s.split())
            pos[BUCKETS:] = [min(w, 80) / 80, float(i < 3), float(i >= n - 3), float(any(c.isdigit() for c in s))]
            rows.append(pos)
        return sparse.csr_matrix(np.stack(rows)) if rows else sparse.csr_matrix((0, BUCKETS + 4))

    def features(self, sents):
        from scipy import sparse
        return sparse.hstack([self.vec.transform(sents), self._extra(sents)]).tocsr()

    def scores(self, sents):
        return self.clf.decision_function(self.features(sents)) if sents else np.zeros(0)

    @classmethod
    def train(cls, docs, max_docs=0, log=print):
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        items = list(docs.values())[:max_docs or None]
        sents, labels = [], []
        for d in items:
            s = split_sentences(d["document"])
            sents.append(s)
            labels.append(oracle_labels(s, split_sentences(d["summary"])))
        flat = [x for s in sents for x in s]
        y = np.concatenate(labels) if labels else np.zeros(0)
        log(f"extractor: {len(items)} documents, {len(flat)} sentences, {y.mean():.3f} oracle-positive")
        rng = np.random.default_rng(0)
        fit_on = [flat[i] for i in rng.choice(len(flat), min(len(flat), 300000), replace=False)]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=200000, sublinear_tf=True,
                              token_pattern=r"(?u)\b\w[\w'-]*\b").fit(fit_on)
        x = sparse.vstack([sparse.hstack([vec.transform(s), cls._extra(s)]) for s in sents if s]).tocsr()
        clf = SGDClassifier(loss="log_loss", alpha=2e-6, class_weight="balanced", max_iter=8, tol=None,
                            random_state=0).fit(x, y)
        return cls(vec, clf)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self, text, total_words):
        sents = split_sentences(text)
        if not sents:
            return text[:2000]
        order = np.argsort(-self.scores(sents))
        keep, n = set(), 0
        for i in order:
            if n >= total_words:
                break
            keep.add(int(i))
            n += len(sents[i].split())
        return " ".join(sents[i] for i in sorted(keep))

    def keep_sets(self, docs, ratio):
        """{id: set(sentence index)}: most salient sentences covering `ratio` of each document's words."""
        out = {}
        for d, doc in docs.items():
            sents = split_sentences(doc["document"])
            total = sum(len(s.split()) for s in sents)
            keep, n = set(), 0
            for i in np.argsort(-self.scores(sents)):
                if n >= ratio * total:
                    break
                keep.add(int(i))
                n += len(sents[i].split())
            out[d] = keep
        return out


def run(args):
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test = load_splits(args.data, args.dev_size)
    length = LengthModel.fit(train)
    print(f"length model: log(summary words) = {length.a:.2f} + {length.b:.2f} log(document words), "
          f"R^2 {length.r2:.2f}, clamp [{length.lo:.0f}, {length.hi:.0f}]", flush=True)
    refs = {d: r["summary"] for d, r in dev.items()}
    want = {d: length.predict(len(r["document"].split())) for d, r in {**dev, **test}.items()}
    results = {}
    lead_dev = {d: lead(r["document"], want[d]) for d, r in dev.items()}
    docs = {d: r["document"] for d, r in dev.items()}
    results["lead"] = evaluate(lead_dev, refs, use_bertscore=not args.no_bertscore, docs=docs)
    print("lead:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in results["lead"].items()}),
          flush=True)
    write_submission({d: lead(r["document"], want[d]) for d, r in test.items()}, test, out / "lead" / "submission")
    if not args.lead_only:
        ext = Extractor.train(train, args.max_train_docs)
        ext.save(out / "extractor.pkl")
        ext_dev = {d: ext.summary(r["document"], want[d]) for d, r in dev.items()}
        results["extractive"] = evaluate(ext_dev, refs, use_bertscore=not args.no_bertscore, docs=docs)
        print("extractive:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in results["extractive"].items()}), flush=True)
        write_submission({d: ext.summary(r["document"], want[d]) for d, r in test.items()}, test,
                         out / "submission")
        (out / "dev_pred.json").write_text(json.dumps(ext_dev, ensure_ascii=False, indent=0))
    (out / "summary.json").write_text(json.dumps({"stage": "baseline", "length_model": length.state(),
                                                  "dev": results, "minutes": round((time.time() - started) / 60, 1)},
                                                 indent=1))
    print(f"-> {out}/summary.json, submissions in {out}/submission and {out}/lead/submission", flush=True)
