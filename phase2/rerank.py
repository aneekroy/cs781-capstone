"""Second-stage span reranker: keep / relabel / drop each candidate span after voting and rules.

The models find person names but often give the wrong party (APP over-predicted, RESP missed),
and propagation rules add spans with no model support. A gradient-boosted classifier looks at
every candidate span with features a tagger window does not see together:
  * model support: share of ensemble members predicting this exact span, any overlapping span,
    and this label; whether the span came from a post-processing rule only;
  * form: length, word count, case, digits, position in the document (header or body);
  * cue words to the left and right ("appellant", "respondent", "witness", "PW-", "learned
    counsel", "Section", "SCC", ...), and the same cues around every other mention of the same
    string in the document (document-level role evidence);
  * how the ensemble labelled the other mentions of the same string.
Target: the gold label of exactly this span, or NONE. Trained on out-of-fold candidates; its own
score is measured by document-fold swap (train on fold-1 documents, score fold-2, and back).
"""
import math
import re
from collections import Counter

import numpy as np

from .post import remove_overlaps, strict_scores

CUES = {
    "appellant": r"appellant", "petitioner": r"petitioner|applicant", "respondent": r"respondent",
    "accused": r"accused|convict", "witness": r"witness|\bp\.?\s?w\.?\s?-?\s?\d|\bd\.?\s?w\.?\s?-?\s?\d|deposed",
    "complainant": r"complainant|informant|\bfir\b", "deceased": r"deceased|victim|murder",
    "counsel": r"counsel|advocate|\badv\b|\bsr\.", "learned": r"learned", "judge": r"justice|\bj\.|\bjj\.|hon'?ble|bench",
    "court": r"court|tribunal", "statute": r"\bsection\b|\bact\b|\bs\.\s?\d|article|\brule\b|code",
    "prec": r"\bv\.|\bvs\.?|versus|\bscc\b|\bair\b|\bscr\b", "date": r"\bdated\b|\bon\b",
    "state": r"\bstate\b|union of india|government", "office": r"officer|police|inspector|collector|commissioner|"
                                                                 r"authority|department|station",
    "mr": r"\bmr\.?\b|\bms\.?\b|\bmrs\.?\b|\bshri\b|\bsmt\.?\b|\bdr\.?\b",
}
CUE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in CUES.items()}
CUE_NAMES = list(CUES)


def _cues(snippet):
    return [1.0 if rx.search(snippet) else 0.0 for rx in CUE_RE.values()]


def _occurrences(text, surface, limit=20):
    words = surface.split()
    if not words:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    out = []
    for m in re.finditer(pattern, text, re.IGNORECASE):
        out.append(m.span())
        if len(out) >= limit:
            break
    return out


def document_rows(text, candidates, member_spans, rule_only, labels, window=60):
    """Feature rows for one document. candidates: [(s,e,label)]; member_spans: list per member."""
    n_members = max(1, len(member_spans))
    exact = Counter(tuple(s) for m in member_spans for s in set(map(tuple, m)))
    bounds = Counter((s, e) for m in member_spans for s, e, _ in set(map(tuple, m)))
    flat = [tuple(s) for m in member_spans for s in m]
    surface_labels = {}
    for s, e, label in candidates:
        surface_labels.setdefault(" ".join(text[s:e].split()).lower(), Counter())[label] += 1
    doc_len = max(1, len(text))
    occurrence_cache = {}
    rows = []
    for s, e, label in candidates:
        surface = text[s:e]
        key = " ".join(surface.split()).lower()
        left, right = text[max(0, s - window):s], text[e:e + window]
        overlap = sum(1 for a, b, _ in flat if a < e and s < b) / n_members
        if key not in occurrence_cache:
            occ = _occurrences(text, surface)
            agg = np.zeros(len(CUE_NAMES))
            for a, b in occ:
                agg += np.array(_cues(text[max(0, a - window):a] + " || " + text[b:b + window]))
            occurrence_cache[key] = (len(occ), agg / max(1, len(occ)))
        n_occ, doc_cues = occurrence_cache[key]
        lab_counts = surface_labels[key]
        words = surface.split()
        row = [1.0 if label == l else 0.0 for l in labels]
        row += [exact[(s, e, label)] / n_members, bounds[(s, e)] / n_members, overlap,
                1.0 if (s, e, label) in rule_only else 0.0,
                math.log1p(e - s), len(words), float(surface.isupper()), float(surface.istitle()),
                float(any(c.isdigit() for c in surface)), float("." in surface),
                s / doc_len, float(s < 2000), math.log1p(n_occ),
                lab_counts[label] / max(1, sum(lab_counts.values())), math.log1p(sum(lab_counts.values()))]
        row += _cues(left) + _cues(right) + list(doc_cues)
        rows.append(row)
    return rows


def build(texts, candidates, members, rule_only, labels, gold=None):
    """X, y (gold label index+1 or 0 = NONE), and (doc, span) keys for every candidate."""
    X, y, keys = [], [], []
    index = {l: i + 1 for i, l in enumerate(labels)}
    for doc, spans in candidates.items():
        spans = sorted(set(map(tuple, spans)))
        if not spans:
            continue
        X += document_rows(texts[doc], spans, [m.get(doc, []) for m in members], rule_only.get(doc, set()), labels)
        keys += [(doc, sp) for sp in spans]
        if gold is not None:
            truth = {(a, b): l for a, b, l in map(tuple, gold.get(doc, []))}
            y += [index.get(truth.get((sp[0], sp[1])), 0) for sp in spans]
    return np.asarray(X, dtype=np.float32), np.asarray(y), keys


def _model(seed=0):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:   # pragma: no cover
        raise SystemExit("the reranker needs scikit-learn: pip install scikit-learn") from exc
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=10,
                                          l2_regularization=1.0, random_state=seed)


def decide(model, X, keys, labels):
    """Spans after reranking: argmax class, NONE dropped, overlaps resolved by probability."""
    out = {}
    if len(keys) == 0:
        return out
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    scored = {}
    for (doc, (s, e, _)), p in zip(keys, proba):
        c = classes[int(np.argmax(p))]
        out.setdefault(doc, [])
        if c == 0:
            continue
        span = (s, e, labels[c - 1])
        scored.setdefault(doc, {})[span] = float(p.max())
    return {doc: remove_overlaps(list(scored.get(doc, {})), scored.get(doc, {})) for doc in out}


def cross_validate(fold_data, labels):
    """fold_data: [(texts, candidates, members, rule_only, gold)] for fold 1 and fold 2 documents.
    Returns pooled macro-F1 with the reranker trained on the other fold, and the predictions."""
    built = [build(t, c, m, r, labels, g) for t, c, m, r, g in fold_data]
    if any(len(set(y.tolist())) < 2 for _, y, _ in built):
        return -1.0, {}                          # a fold without both kept and dropped candidates
    preds, gold = {}, {}
    for i, j in ((0, 1), (1, 0)):
        X_tr, y_tr, _ = built[i]
        X_te, _, keys = built[j]
        model = _model().fit(X_tr, y_tr)
        preds.update(decide(model, X_te, keys, labels))
        gold.update(fold_data[j][4])
    for doc in gold:
        preds.setdefault(doc, [])
    return strict_scores(gold, preds, labels)["macro_f1"], preds


def fit_all(fold_data, labels):
    X = np.concatenate([build(t, c, m, r, labels, g)[0] for t, c, m, r, g in fold_data])
    y = np.concatenate([build(t, c, m, r, labels, g)[1] for t, c, m, r, g in fold_data])
    return _model().fit(X, y)
