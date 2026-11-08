"""Decoding for the Phase-6 summarizers: batched greedy / top-p sampling / beam search on the Phase-1 KV caches,
with per-prompt minimum and maximum lengths, n-gram blocking and repetition penalty; and the document-level
procedure (chunks -> word budgets -> chunk summaries -> joined, de-duplicated, trimmed summary).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from phase2.run import autocast

from .data import assemble, budgets, prepare_doc


@dataclass
class Decoding:
    beams: int = 4
    temperature: float = 0.0          # > 0 with beams == 1: top-p sampling
    top_p: float = 0.9
    no_repeat_ngram: int = 5          # token n-grams never repeated within one output (0 = off)
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0       # beam score / length**alpha
    batch: int = 8                    # prompts per batch (x beams rows)
    seed: int = 42
    mbr: int = 0                      # > 1: minimum-Bayes-risk choice among the top-mbr finished beam hypotheses


def _overlap_f(a, b):
    from collections import Counter
    f = 0.0
    for n in (1, 2):
        ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
        cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
        hit = sum((ca & cb).values())
        if hit:
            p, r = hit / sum(ca.values()), hit / sum(cb.values())
            f += 2 * p * r / (p + r)
    return f / 2


def mbr_pick(cands):
    """The candidate with the highest mean unigram+bigram F1 against the other candidates (consensus)."""
    if len(cands) < 3:
        return cands[0]
    best, score = cands[0], -1.0
    for i, c in enumerate(cands):
        s = sum(_overlap_f(c, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
        if s > score:
            best, score = c, s
    return best


def _banned(table, hist, n):
    if n < 2 or len(hist) < n - 1:
        return ()
    return table.get(tuple(hist[len(hist) - n + 1:]), ())


def _add_ngram(table, hist, n):
    if n >= 2 and len(hist) >= n:
        key = tuple(hist[-n:-1])
        table[key] = table.get(key, ()) + (hist[-1],)


@torch.no_grad()
def _batch(model, prompts, min_new, max_new, dec, device, dtype, generator):
    b, nb = len(prompts), max(1, dec.beams)
    with autocast(device, dtype):
        logits, state = model.start(prompts, device)
    if nb > 1:
        expand = torch.arange(b, device=device).repeat_interleave(nb)
        state, logits = model.reorder(state, expand), logits[expand]
    rows = b * nb
    eos = model.eos_id
    forbid = [t for t in {model.pad_id, model.bos_id, model.unk_id} if t is not None and t != eos]
    mins = [min_new[r // nb] for r in range(rows)]
    maxs = [max(1, max_new[r // nb]) for r in range(rows)]
    hist = [[] for _ in range(rows)]
    ngr = [dict() for _ in range(rows)]
    seen = torch.zeros(rows, logits.size(-1), dtype=torch.bool, device=device) \
        if dec.repetition_penalty != 1.0 else None
    beam_scores = torch.zeros(b, nb, device=device)
    beam_scores[:, 1:] = -float("inf")
    finished = [[] for _ in range(b)]
    done = [False] * b
    for t in range(max(maxs)):
        lg = logits.float()
        if seen is not None:
            rp = dec.repetition_penalty
            lg = torch.where(seen, torch.where(lg < 0, lg * rp, lg / rp), lg)
        if forbid:
            lg[:, forbid] = -float("inf")
        early = [r for r in range(rows) if t < mins[r]]
        if early:
            lg[early, eos] = -float("inf")
        last = [r for r in range(rows) if t >= maxs[r] - 1]
        if last:                                                # budget reached: only <eos> remains
            keep = lg[last, eos].clone()
            lg[last] = -float("inf")
            lg[last, eos] = torch.where(torch.isfinite(keep), keep, torch.zeros_like(keep))
        if dec.no_repeat_ngram:
            for r in range(rows):
                ban = _banned(ngr[r], hist[r], dec.no_repeat_ngram)
                if ban:
                    lg[r, list(ban)] = -float("inf")
        if nb == 1:
            if dec.temperature > 0:
                probs = F.softmax(lg / dec.temperature, -1)
                sp, si = probs.sort(-1, descending=True)
                sp[(sp.cumsum(-1) - sp) > dec.top_p] = 0
                tok = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=generator)).squeeze(1)
            else:
                tok = lg.argmax(-1)
            toks = tok.tolist()
            for r in range(rows):
                if done[r]:
                    continue
                if toks[r] == eos:
                    done[r] = True
                    finished[r] = [(0.0, hist[r])]
                    continue
                hist[r].append(toks[r])
                _add_ngram(ngr[r], hist[r], dec.no_repeat_ngram)
                if len(hist[r]) >= maxs[r]:
                    done[r] = True
                    finished[r] = [(0.0, hist[r])]
            if seen is not None:
                seen[torch.arange(rows, device=device), tok] = True
            if all(done):
                break
            with autocast(device, dtype):
                logits, state = model.step(tok, state)
            continue
        # ---- beam search
        logp = F.log_softmax(lg, -1)
        vocab = logp.size(-1)
        cand = (beam_scores.view(-1, 1) + logp).view(b, nb * vocab)
        top_v, top_i = cand.topk(2 * nb, dim=1)
        top_v, top_i = top_v.tolist(), top_i.tolist()
        src, new_tok, new_score = [], [], []
        for x in range(b):
            picked = []
            if not done[x]:
                for rank, (v, i) in enumerate(zip(top_v[x], top_i[x])):
                    if v == -float("inf"):
                        break
                    j, tok = divmod(i, vocab)
                    r = x * nb + j
                    if tok == eos:
                        if rank < nb:
                            finished[x].append((v / max(1, len(hist[r]) + 1) ** dec.length_penalty, hist[r]))
                        continue
                    picked.append((r, tok, v))
                    if len(picked) == nb:
                        break
                if len(finished[x]) >= nb or not picked or t + 1 >= max(maxs[x * nb:(x + 1) * nb]):
                    done[x] = True
                    for r, tok, v in picked[:1] if not finished[x] else []:
                        finished[x].append((v / max(1, len(hist[r]) + 1) ** dec.length_penalty, hist[r] + [tok]))
            while len(picked) < nb:                             # finished rows / too few live candidates
                r = x * nb + len(picked)
                picked.append((r, eos, -float("inf")))
            for r, tok, v in picked:
                src.append(r)
                new_tok.append(tok)
                new_score.append(v)
        if all(done):
            break
        idx = torch.tensor(src, device=device)
        hist = [hist[r] + [tok] for r, tok in zip(src, new_tok)]
        if dec.no_repeat_ngram:
            ngr = [dict(ngr[r]) for r in src]
            for r in range(rows):
                _add_ngram(ngr[r], hist[r], dec.no_repeat_ngram)
        tok_t = torch.tensor(new_tok, device=device)
        if seen is not None:
            seen = seen.index_select(0, idx)
            seen[torch.arange(rows, device=device), tok_t] = True
        beam_scores = torch.tensor(new_score, device=device).view(b, nb)
        state = model.reorder(state, idx)
        with autocast(device, dtype):
            logits, state = model.step(tok_t, state)
    if nb == 1:
        return [finished[r][0][1] if finished[r] else hist[r] for r in range(rows)]
    out = []
    for x in range(b):
        if not finished[x]:
            r = x * nb + int(beam_scores[x].argmax())
            finished[x].append((0.0, hist[r]))
        ranked = [h for _, h in sorted(finished[x], key=lambda s: -s[0])]
        out.append(mbr_pick(ranked[:dec.mbr]) if dec.mbr > 1 else ranked[0])
    return out


def _size(p):
    return sum(len(c) for c in p) if p and isinstance(p[0], list) else len(p)


@torch.no_grad()
def generate(model, prompts, min_new, max_new, dec, device, dtype, log=None):
    """Token id lists (without <eos>), one per prompt. Prompts are batched in order of length."""
    model.eval()
    fid = bool(prompts) and isinstance(prompts[0], list) and bool(prompts[0]) and isinstance(prompts[0][0], list)
    size = max(1, dec.batch // 4) if fid else max(1, dec.batch)
    order = sorted(range(len(prompts)), key=lambda i: _size(prompts[i]))
    out = [None] * len(prompts)
    g = torch.Generator(device=device).manual_seed(dec.seed) if dec.temperature > 0 else None
    for n, s in enumerate(range(0, len(order), size)):
        idx = order[s:s + size]
        res = _batch(model, [prompts[i] for i in idx], [min_new[i] for i in idx], [max_new[i] for i in idx],
                     dec, device, dtype, g)
        for i, r in zip(idx, res):
            out[i] = r
        if log and (n + 1) % 20 == 0:
            log(f"  generated {min(len(order), s + size)}/{len(order)} prompts")
    return out


# --------------------------------------------------------------------------- documents -> summaries
def doc_prompts(model, doc, plan, total_words):
    """[(prompt, budget_words)] for one prepared document under the plan's strategy."""
    chunks = doc["chunks"]
    if plan["strategy"] == "fid":
        n = len(chunks)
        k = plan["fid_chunks"]
        pick = range(n) if n <= k else sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})
        return [([chunks[i]["ids"] for i in pick], total_words)]
    if plan["strategy"] == "head":
        return [(chunks[0]["ids"], total_words)]
    split = budgets(doc, total_words, plan["profile"] if plan["budget"] == "learned" else None)
    return [(c["ctx"] + c["ids"], w) for c, w in zip(chunks, split)]


def summarize(model, tokenizer, docs, plan, dec, device, dtype, keep=None, log=print):
    """{id: summary} and per-document diagnostics. plan: strategy, chunk, overlap, fid_chunks, budget, profile,
    length (LengthModel), len_scale, tpw (summary tokens per word), min_ratio, max_ratio, dedup."""
    prompts, mins, maxs, owner, words_for = [], [], [], [], {}
    for d, doc in docs.items():
        prep = prepare_doc(tokenizer, doc["document"], plan["chunk"], plan["overlap"], (keep or {}).get(d))
        total = plan["length"].predict(len(doc["document"].split()), plan["len_scale"])
        words_for[d] = total
        for p, w in doc_prompts(model, prep, plan, total):
            room = model.room(p)
            hi = min(room, int(w * plan["tpw"] * plan["max_ratio"]) + 8)
            lo = min(hi - 1, int(w * plan["tpw"] * plan["min_ratio"]))
            prompts.append(p)
            mins.append(max(0, lo))
            maxs.append(max(1, hi))
            owner.append(d)
    log(f"decoding {len(prompts)} prompts for {len(docs)} documents ({plan['strategy']}, {asdict(dec)})")
    outs = generate(model, prompts, mins, maxs, dec, device, dtype, log)
    pieces, diag = {}, {}
    for d, ids, hi in zip(owner, outs, maxs):
        pieces.setdefault(d, []).append(tokenizer.decode(ids, skip_special_tokens=True))
        info = diag.setdefault(d, {"budget_words": words_for[d], "prompts": 0, "hit_max": 0})
        info["prompts"] += 1
        info["hit_max"] += int(len(ids) >= hi - 1)          # the last slot is reserved for <eos>
    summaries = {d: assemble(pieces.get(d, []), words_for[d], plan["dedup"]) for d in docs}
    return summaries, diag
