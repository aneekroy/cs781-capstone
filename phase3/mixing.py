"""Data mixing for Phase 3 (official data only; everything built from the training fold of each run).

1. Label-targeted oversampling (--repeat-labels Dissent=4,Issue=3,...): a training segment is repeated as
   often as the largest weight among the labels it contains. Macro-F1 gives Dissent (284 sentences, 2
   documents) the same weight as Fact (4542), so rare roles get more gradient.
2. Domain weights (--domain-weights CL=1,IT=1.3): fractional repetition of a domain's segments.
3. Block splicing (--splice R): R x (training documents) synthetic documents per epoch. A document is a
   sequence of role blocks (maximal runs of one label); a synthetic document takes a real document of
   one domain and (a) swaps some blocks for same-label blocks of other documents of that domain and
   (b) inserts a rare-role block (Dissent, Issue, PRE-NR, ...) from another document at the block
   boundary closest to where that role sits in its source (a Dissent note goes to the end, Issues after
   the facts). Rhetorical order is preserved, so the context model sees new, plausible neighbourhoods.
4. In-domain silver data for KD (phase3.data.unlabeled_documents with select=True): Phase-1 judgments
   scored by competition-law / tax-and-banking vocabulary, tagged with that domain, teacher-labelled.
"""
import random


def parse_weights(spec):
    out = {}
    for part in (spec or "").split(","):
        if part.strip():
            key, value = part.split("=")
            out[key.strip()] = float(value)
    return out


def repeats(weight, rng):
    """Fractional repetition count: 1.3 -> 1 or 2 (30%)."""
    whole = int(weight)
    return whole + (1 if rng.random() < weight - whole else 0)


def blocks(labels):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out


def splice_documents(recs, n_new, rng, rare_labels, swap_prob=0.3, max_sentences=800):
    """recs: {doc: {"ids", "labels", "domain"}} (training fold). Returns {synthetic id: rec}."""
    by_domain = {}
    for doc, rec in recs.items():
        if rec["labels"]:
            by_domain.setdefault(rec["domain"], []).append(doc)
    pools = {}                                   # (domain, label) -> [(doc, start, end, relative position)]
    for doc, rec in recs.items():
        if not rec["labels"]:
            continue
        n = len(rec["labels"])
        for label, s, e in blocks(rec["labels"]):     # anchor: 1.0 = closes the document (a dissent note)
            pools.setdefault((rec["domain"], label), []).append((doc, s, e, 1.0 if e == n else s / n))
    out = {}
    domains = [d for d in by_domain if len(by_domain[d]) > 1]
    if not domains:
        return out
    for k in range(n_new):
        domain = rng.choice(domains)
        base_id = rng.choice(by_domain[domain])
        base = recs[base_id]
        pieces = []                              # list of (ids list, labels list)
        for label, s, e in blocks(base["labels"]):
            pool = [p for p in pools.get((domain, label), []) if p[0] != base_id]
            if pool and rng.random() < swap_prob:
                d, ps, pe, _ = rng.choice(pool)
                pieces.append((recs[d]["ids"][ps:pe], recs[d]["labels"][ps:pe]))
            else:
                pieces.append((base["ids"][s:e], base["labels"][s:e]))
        rare = [l for l in rare_labels if pools.get((domain, l))]
        if rare:
            label = rng.choice(rare)
            d, ps, pe, rel = rng.choice(pools[(domain, label)])
            if d != base_id or len(pools[(domain, label)]) == 1:
                at = len(pieces) if rel >= 1.0 else min(len(pieces) - 1, max(1, round(rel * len(pieces))))
                pieces.insert(at, (recs[d]["ids"][ps:pe], recs[d]["labels"][ps:pe]))
        ids, labels = [], []
        for i_, l_ in pieces:
            ids += list(i_)
            labels += list(l_)
        if len(ids) > max_sentences:             # keep the end (rulings, dissent) and the start
            head = max_sentences // 2
            ids, labels = ids[:head] + ids[-(max_sentences - head):], labels[:head] + labels[-(max_sentences - head):]
        out[f"splice_{k}_{base_id}"] = {"ids": ids, "labels": labels, "domain": domain}
    return out


# --------------------------------------------------------------------------- in-domain silver data
CL_TERMS = ("competition commission", "competition act", "dominant position", "abuse of dominan", "cartel",
            "anti-competitive", "anticompetitive", "relevant market", "director general", "informant",
            "opposite part", "compat", "appreciable adverse effect", "bid rigging", "monopolies and restrictive",
            "mrtp", "combination", "predatory pric")
IT_TERMS = ("income tax", "income-tax", "assessee", "assessment year", "commissioner of income", "reassessment",
            "taxable", "capital gain", "depreciation", "appellate tribunal", "wealth tax", "deduction under section",
            "banking regulation", "recovery of debts", "sarfaesi", "non-performing", "reserve bank", "negotiable instrument")


def domain_of(text, min_hits=8, min_rate=1.5):
    """'CL' / 'IT' / None from vocabulary hits (per 1000 words)."""
    low = text.lower()
    words = max(1, low.count(" "))
    cl = sum(low.count(t) for t in CL_TERMS)
    it = sum(low.count(t) for t in IT_TERMS)
    best, hits = ("CL", cl) if cl >= it else ("IT", it)
    if hits >= min_hits and 1000 * hits / words >= min_rate:
        return best
    return None
