"""Decoding choices tuned on out-of-fold predictions for the pooled macro-F1.

Every label is worth 1/13 of the score whatever its frequency, so a small additive offset per label
(on CRF emissions before Viterbi, or on sentence log-posteriors before argmax) that trades a few Fact
or ROD sentences for Issue / RPC / PRE-NR recall moves the score. Offsets are chosen by coordinate
ascent on the pooled OOF score; whether they are used at all is decided honestly: choose on half of
the documents, score the other half, swap, pool, and compare with no offsets.

A record is one document: {"emissions": [n,K], "logp": [n,K], "crf": {"trans","start","end"} | None,
"domain": "CL"/"IT"}.
"""
import numpy as np

from .data import FastScorer
from .model import viterbi_np

GRID = (-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0)


try:                                   # optional: ~100x faster decoding for offset tuning (pip install numba)
    from numba import njit

    @njit(cache=True)
    def _viterbi_one(em, trans, start, end):
        n, k = em.shape
        score = start + em[0]
        back = np.zeros((n, k), dtype=np.int64)
        for i in range(1, n):
            new = np.empty(k)
            for j in range(k):
                best, arg = -1e300, 0
                for a in range(k):
                    v = score[a] + trans[a, j]
                    if v > best:
                        best, arg = v, a
                new[j] = best + em[i, j]
                back[i, j] = arg
            score = new
        path = np.zeros(n, dtype=np.int64)
        path[n - 1] = np.argmax(score + end)
        for i in range(n - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        return path
except ImportError:
    _viterbi_one = None


def viterbi_batch(ems, crfs):
    """Viterbi for many documents at once (each with its own CRF): lists of [n_i,K] and crf dicts."""
    if _viterbi_one is not None:
        return [_viterbi_one(np.ascontiguousarray(e, dtype=np.float64), np.asarray(c["trans"], dtype=np.float64),
                             np.asarray(c["start"], dtype=np.float64), np.asarray(c["end"], dtype=np.float64))
                for e, c in zip(ems, crfs)]
    b, k = len(ems), ems[0].shape[1]
    lengths = np.array([len(e) for e in ems])
    n = int(lengths.max())
    em = np.zeros((b, n, k))
    for i, e in enumerate(ems):
        em[i, :len(e)] = e
    trans = np.stack([c["trans"] for c in crfs])
    start = np.stack([c["start"] for c in crfs])
    end = np.stack([c["end"] for c in crfs])
    score = start + em[:, 0]
    back = np.zeros((b, n, k), dtype=np.int64)
    rows = np.arange(b)
    for i in range(1, n):
        total = score[:, :, None] + trans
        best = total.argmax(1)
        value = np.take_along_axis(total, best[:, None, :], 1)[:, 0] + em[:, i]
        active = i < lengths
        back[:, i] = best
        score = np.where(active[:, None], value, score)
    cur = (score + end).argmax(1)
    path = np.zeros((b, n), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        active = i < lengths
        path[active, i] = cur[active]
        if i > 0:
            cur = np.where(active, back[rows, i, cur], cur)
    return [path[i, :lengths[i]] for i in range(b)]


def _offset(offsets, domain):
    return offsets.get(domain) if isinstance(offsets, dict) else offsets


def predict_indices(records, mode, offsets=None):
    """offsets: None, array [K] (global) or dict domain -> [K]. A record's optional "forced" array (label index
    or -1 per sentence, from phase3.rules) overrides the decoder, so offsets are tuned with the rules in place."""
    if mode == "viterbi":
        ems = [r["emissions"] if _offset(offsets, r["domain"]) is None else r["emissions"] + _offset(offsets, r["domain"])
               for r in records]
        per_doc = viterbi_batch(ems, [r["crf"] for r in records])
    else:
        per_doc = []
        for r in records:
            off = _offset(offsets, r["domain"])
            x = r["logp"] if off is None else r["logp"] + off
            per_doc.append(x.argmax(1))
    out = []
    for r, idx in zip(records, per_doc):
        forced = r.get("forced")
        idx = idx if forced is None else np.where(forced >= 0, forced, idx)
        post = r.get("post")                 # label-dependent rules (speaker tracking) on the decoded labels
        out.append(post(idx) if post is not None else idx)
    return np.concatenate(out)


def tune_offsets(records, gold_idx, k, mode, per_domain=False, rounds=2, grid=GRID):
    scorer = FastScorer(gold_idx, k)
    domains = sorted({r["domain"] for r in records}) if per_domain else [None]
    offsets = {d: np.zeros(k) for d in domains}

    def current():
        return offsets if per_domain else offsets[None]

    best = scorer(predict_indices(records, mode, current()))
    for _ in range(rounds):
        improved = False
        for d in domains:
            for label in range(k):
                keep = offsets[d][label]
                for value in grid:
                    if value == keep:
                        continue
                    offsets[d][label] = value
                    s = scorer(predict_indices(records, mode, current()))
                    if s > best + 1e-9:
                        best, keep, improved = s, value, True
                offsets[d][label] = keep
        if not improved:
            break
    return current(), best


def halves(doc_ids, domains):
    """Two document halves, each domain split evenly (deterministic)."""
    a, b, seen = [], [], {}
    for doc in doc_ids:
        d = domains[doc]
        (a if seen.get(d, 0) % 2 == 0 else b).append(doc)
        seen[d] = seen.get(d, 0) + 1
    return a, b


def choose_decoding(oof, gold, labels, log=print, modes=None, max_tune_docs=None, tune=True):
    """oof: {doc: record}; gold: {doc: [label strings]}. Returns the config with the best honest score."""
    index = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    docs = [d for d in oof if d in gold]
    records = {d: oof[d] for d in docs}
    gold_idx = {d: np.array([index[l] for l in gold[d]]) for d in docs}
    has_crf = all(r.get("crf") is not None for r in records.values())
    modes = modes or (["viterbi", "marginal"] if has_crf else ["marginal"])
    domains = {d: records[d]["domain"] for d in docs}
    half_a, half_b = halves(docs, domains)

    def pooled(doc_list, mode, offsets=None):
        g = np.concatenate([gold_idx[d] for d in doc_list])
        return FastScorer(g, k)(predict_indices([records[d] for d in doc_list], mode, offsets))

    results = []
    for mode in modes:
        base = pooled(docs, mode)
        results.append({"mode": mode, "offsets": "none", "honest": base, "fit": base})
        log(f"  decode {mode:8s} no offsets          OOF {base:.4f}")
        for per_domain in ((False, True) if tune else ()):
            preds, golds = [], []
            for fit_docs, eval_docs in ((half_a, half_b), (half_b, half_a)):
                g = np.concatenate([gold_idx[d] for d in fit_docs])
                off, _ = tune_offsets([records[d] for d in fit_docs], g, k, mode, per_domain)
                preds.append(predict_indices([records[d] for d in eval_docs], mode, off))
                golds.append(np.concatenate([gold_idx[d] for d in eval_docs]))
            honest = FastScorer(np.concatenate(golds), k)(np.concatenate(preds))
            g_all = np.concatenate([gold_idx[d] for d in docs])
            off_all, fit = tune_offsets([records[d] for d in docs], g_all, k, mode, per_domain)
            name = "domain" if per_domain else "global"
            results.append({"mode": mode, "offsets": name, "honest": honest, "fit": fit, "values": off_all})
            log(f"  decode {mode:8s} {name:6s} offsets  half-swap {honest:.4f}  (fit on all {fit:.4f})")
    # offsets must beat "none" by a margin on the honest score; otherwise keep the simpler decoder
    best_none = max((r for r in results if r["offsets"] == "none"), key=lambda r: r["honest"])
    best = max(results, key=lambda r: r["honest"])
    if best["offsets"] != "none" and best["honest"] < best_none["honest"] + 0.003:
        best = best_none
    return best, results


def with_rules(records, docs, rules, labels):
    """Copies of the records carrying the rules' forced labels (records without a document are left as is)."""
    from .rules import forced_indices, post_rule
    index = {l: i for i, l in enumerate(labels)}
    return {d: ({**r, "forced": forced_indices(docs[d]["sentences"], rules, index),
                 "post": post_rule(rules, docs[d]["sentences"], d, labels)} if rules and d in docs else r)
            for d, r in records.items()}


def finalize_decoding(oof, gold, labels, docs, log=print, modes=None, rules="auto"):
    """Decoder + offsets chosen WITH the document-structure rules in place (phase3.rules): offsets tuned without
    them learn to push labels the rules already settle (a +3 Dissent offset scatters Dissent into documents
    without a dissent note). Each rule set is a full candidate; with rules="auto" one is kept only if it
    beats decoding without rules. The rules are fixed text patterns, not fitted on these labels."""
    from .rules import RULES, TEXT_RULES, VARIANTS
    if rules == "auto":
        variants = [list(v) for v in VARIANTS] if docs is not None else [[]]
    else:
        names = [r for r in (rules or "").split(",") if r and r != "none"]
        unknown = [r for r in names if r not in RULES]
        if unknown:
            raise ValueError(f"unknown rules {unknown}; known: {list(RULES)}")
        variants = [names] if names and docs is not None else [[]]
    best = None
    for variant in variants:
        recs = with_rules(oof, docs, variant, labels) if variant else oof
        # the no-rule baseline only decides whether the rules are kept: plain decoders, no offset tuning (fast)
        config, results = choose_decoding(recs, gold, labels, log=log if variant or len(variants) == 1 else (lambda *a, **k: None),
                                          modes=modes, tune=bool(variant) or len(variants) == 1)
        config = dict(config)
        config["rules"] = list(variant)
        log(f"  rules {variant or 'none'}: {config['mode']} + {config['offsets']} offsets, honest {config['honest']:.4f}")
        # a label-dependent rule (speaker tracking) has to earn its place: +0.002 over the best simpler variant
        margin = 0.002 if any(r not in TEXT_RULES for r in variant) else 1e-9
        if best is None or config["honest"] > best[0]["honest"] + margin:
            best = (config, results)
    return best


def apply_decoding(records, config, labels, docs=None):
    """{doc: [label strings]} with a chosen config (mode + offsets [+ rules, which need the documents])."""
    offsets = config.get("values")
    if isinstance(offsets, dict):
        offsets = {d: np.asarray(v) for d, v in offsets.items()}
    elif offsets is not None:
        offsets = np.asarray(offsets)
    out = {}
    for doc, r in records.items():
        idx = predict_indices([r], config["mode"], offsets)
        out[doc] = [labels[i] for i in idx]
    if config.get("rules"):
        if docs is None:
            raise ValueError("decoding config has rules: pass the documents")
        from .rules import apply_rules
        out = apply_rules(out, docs, config["rules"])
    return out


def jsonable(config):
    c = dict(config)
    v = c.get("values")
    if isinstance(v, dict):
        c["values"] = {d: [round(float(x), 3) for x in arr] for d, arr in v.items()}
    elif v is not None:
        c["values"] = [round(float(x), 3) for x in v]
    c["rules"] = list(c.get("rules") or [])
    for key in ("honest", "fit"):
        if key in c:
            c[key] = round(float(c[key]), 5)
    return c


# --------------------------------------------------------------------------- sequence decoding for ensembles
# Averaging posteriors throws the members' CRF structure away: the average is decoded sentence by sentence
# (argmax), so a document's role blocks get isolated flips that cost precision on both labels. Two ways back:
#   * "crfavg": average every member's CRF emissions and transition/start/end scores (a log-linear pool of
#     the CRFs; all members must have a CRF) and run Viterbi;
#   * "smooth": Viterbi over the averaged log-posteriors with lam x a label-transition prior
#     log P(y_t | y_t-1) - log P(y_t) counted from the gold training sequences (13x13 counts; works for any
#     members, including stackers and non-CRF runs). lam in {0.25, 0.5, 1} is chosen like everything else.
# The candidates are compared on the same honest half-swap score; "marginal" (the old decoder) stays one of them.

def transition_prior(gold_seqs, labels, smoothing=0.5):
    k = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    big, first, last, uni = (np.full((k, k), smoothing), np.full(k, smoothing), np.full(k, smoothing),
                             np.full(k, smoothing))
    for seq in gold_seqs:
        idx = [index[l] for l in seq]
        if not idx:
            continue
        first[idx[0]] += 1
        last[idx[-1]] += 1
        for a, b in zip(idx[:-1], idx[1:]):
            big[a, b] += 1
        for a in idx:
            uni[a] += 1
    log_uni = np.log(uni / uni.sum())
    return {"trans": np.log(big / big.sum(1, keepdims=True)) - log_uni[None, :],
            "start": np.log(first / first.sum()) - log_uni,
            "end": np.log(last / last.sum()) - log_uni}


def smoothed(records, prior, lam):
    crf = {key: lam * np.asarray(v) for key, v in prior.items()}
    return {d: {**r, "emissions": r["logp"], "crf": crf} for d, r in records.items()}


def crf_average(parts):
    """parts: list of {doc: record with emissions + crf}; None if any member lacks a CRF."""
    if any(r.get("crf") is None for part in parts for r in part.values()):
        return None
    out = {}
    for doc in parts[0]:
        recs = [part[doc] for part in parts]
        p = np.mean([np.exp(r["logp"]) for r in recs], axis=0)
        out[doc] = {"emissions": np.mean([r["emissions"] for r in recs], axis=0),
                    "logp": np.log(np.clip(p, 1e-12, None)), "domain": recs[0]["domain"],
                    "crf": {key: np.mean([np.asarray(r["crf"][key]) for r in recs], axis=0)
                            for key in ("trans", "start", "end")}}
    return out


def sequence_candidates(avg, crf_avg, prior, lams=(0.25, 0.5, 1.0)):
    """[(spec, records, modes)]; spec says how to rebuild the records for the test set."""
    cands = [({"kind": "marginal"}, avg, ["marginal"])]
    for lam in lams:
        cands.append(({"kind": "smooth", "lam": lam}, smoothed(avg, prior, lam), ["viterbi"]))
    if crf_avg is not None:
        cands.append(({"kind": "crfavg"}, crf_avg, ["viterbi"]))
    return cands


def build_sequence_records(spec, avg, crf_avg, prior):
    if spec["kind"] == "smooth":
        return smoothed(avg, prior, spec["lam"])
    if spec["kind"] == "crfavg":
        if crf_avg is None:
            raise ValueError("crfavg decoding needs CRF records for every member")
        return crf_avg
    return avg


def choose_sequence_decoding(avg, crf_avg, gold, labels, docs, prior, log=print, rules="auto", kinds=None):
    """Best (config, records) over the sequence candidates by the honest score; config["sequence"] holds the spec."""
    best = None
    for spec, recs, modes in sequence_candidates(avg, crf_avg, prior):
        if kinds and spec["kind"] not in kinds:
            continue
        config, _ = finalize_decoding(recs, gold, labels, docs, log=lambda *a, **k: None, modes=modes, rules=rules)
        tag = spec["kind"] + (f" lam={spec['lam']}" if "lam" in spec else "")
        log(f"  sequence {tag:16s}: {config['mode']} + {config['offsets']} offsets + rules {config['rules']}  "
            f"honest {config['honest']:.4f}")
        if best is None or config["honest"] > best[0]["honest"] + 1e-9:
            best = ({**config, "sequence": spec}, recs)
    return best
