"""Held-out scoring for Phase 6: ROUGE-1/2/L F1 and BERTScore F1 (the primary competition metric).

ROUGE: the `rouge_score` package (Google's implementation, Porter stemming) when it is installed, else the same
tokenisation (lower case, runs of [a-z0-9]) without stemming and a bit-parallel LCS. The implementation used is
recorded with the scores.
BERTScore: the `bert_score` package when installed. The scorer model is used only to SCORE held-out summaries
(as the course scorer did in Phase 1); it never touches training. Model: $BERTSCORE_MODEL, else the model_type
named in the course evaluator ($PROJECT/evaluators/evaluate_phase6.py or evaluate_phase1.py), else the package
default for English (roberta-large). Note: bert_score truncates each text to the scorer's 512 sub-words, so with
~1k-word summaries the BERTScore is decided by roughly the first 350-400 words of each summary.
"""
import os
import re
from pathlib import Path

import numpy as np

from .data import redundancy, words

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOK.findall(text.lower())


def lcs(a, b):
    """Length of the longest common subsequence (Hyyro's bit-parallel algorithm)."""
    if not a or not b:
        return 0
    masks = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for t in b:
        u = v & masks.get(t, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def _f(overlap, n_pred, n_ref):
    if not overlap:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _ngrams(t, n):
    from collections import Counter
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def rouge_own(pred, ref):
    a, b = _tokens(pred), _tokens(ref)
    out = {}
    for n in (1, 2):
        pa, pb = _ngrams(a, n), _ngrams(b, n)
        out[f"rouge{n}"] = _f(sum((pa & pb).values()), max(1, sum(pa.values())), max(1, sum(pb.values())))
    out["rougeL"] = _f(lcs(a, b), max(1, len(a)), max(1, len(b)))
    return out


def rouge(preds, refs):
    """Mean ROUGE-1/2/L F1 over aligned lists."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rows = [{k: v.fmeasure for k, v in scorer.score(r, p).items()} for p, r in zip(preds, refs)]
        impl = "rouge_score (stemmed)"
    except ImportError:
        rows = [rouge_own(p, r) for p, r in zip(preds, refs)]
        impl = "built-in (no stemming; pip install rouge-score for the official one)"
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("rouge1", "rouge2", "rougeL")}
    out["rouge_impl"] = impl
    return out


def scorer_model():
    if os.environ.get("BERTSCORE_MODEL"):
        return os.environ["BERTSCORE_MODEL"]
    root = Path(os.environ.get("PROJECT", "."))
    for name in ("evaluate_phase6.py", "evaluate_phase1.py"):
        p = root / "evaluators" / name
        if p.exists():
            m = re.search(r"model_type\s*=\s*['\"]([^'\"]+)['\"]", p.read_text(errors="ignore"))
            if m:
                return m.group(1)
    return None


def bertscore(preds, refs, batch_size=8, device=None):
    """(mean P, R, F1, model) or None when bert_score is not installed / the scorer model cannot be loaded."""
    try:
        from bert_score import score
    except ImportError:
        return None
    model = scorer_model()
    try:
        kw = {"model_type": model} if model else {"lang": "en"}
        p, r, f = score(list(preds), list(refs), batch_size=batch_size, device=device, verbose=False, **kw)
    except Exception as e:                                       # offline without the model, CUDA OOM, ...
        print(f"BERTScore unavailable: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None
    return float(p.mean()), float(r.mean()), float(f.mean()), model or "default (lang=en)"


def faithfulness(pred, doc):
    """Cheap hallucination / abstractiveness proxies: share of the summary's numbers (sections, years, amounts)
    that occur in the document, and share of its word 3-grams that do not occur in the document."""
    nums = re.findall(r"\d+", pred)
    doc_nums = set(re.findall(r"\d+", doc))
    t, s = _tokens(pred), _tokens(doc)
    grams = {tuple(t[i:i + 3]) for i in range(len(t) - 2)}
    doc_grams = {tuple(s[i:i + 3]) for i in range(len(s) - 2)}
    return (sum(n in doc_nums for n in nums) / len(nums) if nums else 1.0,
            len(grams - doc_grams) / len(grams) if grams else 0.0)


def evaluate(pred, refs, use_bertscore=True, device=None, docs=None):
    """pred, refs: {id: text} (scored on refs' ids); docs {id: document} adds the faithfulness proxies."""
    ids = [d for d in refs if d in pred]
    p, r = [pred[d] or "." for d in ids], [refs[d] for d in ids]
    out = {"documents": len(ids), **rouge(p, r)}
    if docs:
        f = [faithfulness(pred[d] or "", docs[d]) for d in ids]
        out["number_precision"] = float(np.mean([x[0] for x in f]))
        out["novel_3gram"] = float(np.mean([x[1] for x in f]))
        out["ref_novel_3gram"] = float(np.mean([faithfulness(refs[d], docs[d])[1] for d in ids]))
    out["words_pred"] = float(np.mean([words(x) for x in p])) if p else 0.0
    out["words_ref"] = float(np.mean([words(x) for x in r])) if r else 0.0
    out["length_ratio"] = out["words_pred"] / max(1.0, out["words_ref"])
    out["redundancy_3gram"] = float(np.mean([redundancy(x) for x in p])) if p else 0.0
    bs = bertscore(p, r, device=device) if use_bertscore else None
    if bs:
        out.update({"bertscore_p": bs[0], "bertscore_r": bs[1], "bertscore_f1": bs[2], "bertscore_model": bs[3]})
    out["selection_score"] = out.get("bertscore_f1", (out["rouge1"] + out["rougeL"]) / 2)
    out["selection_metric"] = "bertscore_f1" if bs else "mean(rouge1, rougeL)"
    return out
