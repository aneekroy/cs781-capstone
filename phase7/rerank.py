"""Passage-level re-ranker for Phase 7: the Phase-1 model reads a (query passage, candidate passage) pair together
and scores whether the query cites the candidate there; it re-scores the top candidates of any finished run.

Why. A bi-encoder compresses each passage alone; a pair model sees both texts at once, so it can tell a quoted or
applied holding from a passage that is merely on the same topic. Full judgments are far too long for one pass,
so a candidate is judged by its best passage pairs: for each of the first stage's top --rerank-depth candidates,
the --pair-k query/candidate passage pairs with the highest word 2-3-gram TF-IDF cosine are scored, and the
candidate gets the max (or mean of the best two, --pair-agg mean2).

Input: [query passage] "\\n\\nCited case:" [candidate passage] -> pooled state (last token for causal decoders,
mean for bidirectional / the encoder) -> linear score. Training: the mined positive passage pairs of cited cases
(the retriever's pairs) against their hard negatives (similar passages of uncited BM25-top cases) and a random
passage, softmax cross-entropy per group. Any technique (--method).

Final score of the re-ranked candidates: w z(first stage) + (1 - w) z(re-ranker), w chosen on dev; candidates
below the depth keep their first-stage order underneath.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase2.run import autocast, device_and_amp

from .data import f1_at_k, load_split, query_texts, rank_from_scores, recall_at, write_submission, zscore_rows
from .lexical import Vocab, chunk_tfidf
from .neural import (LR_DEFAULT, Chunks, build_retriever, lexical_scores, log, mine_pairs, pad, trainable_state)

SEP = "\n\nCited case:"


class CrossScorer(nn.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.head = nn.Linear(retriever.encoder.dim, 1)

    def forward(self, ids, mask):
        return self.head(self.retriever.pooled(ids, mask).float()).squeeze(-1)


def pair_ids(q, c, sep, side):
    return q[:side] + sep + c[:side]


def first_stage(run, split):
    """(scores [nq, npool], query ids, pool ids, lexical or None) saved by a retriever, lexical or combine run."""
    z = np.load(Path(run) / f"scores_{split}.npz")
    keys = set(z.files)
    if "fused" in keys:
        s = z["fused"]
    elif "neural" in keys:
        s = z["neural"]
    else:
        s = z["lexical"]
    return s, [str(x) for x in z["query_ids"]], [str(x) for x in z["pool_ids"]], \
        (z["lexical"] if "lexical" in keys else None)


def align(scores, rows, cols, want_rows, want_cols):
    r = {q: i for i, q in enumerate(rows)}
    c = {d: j for j, d in enumerate(cols)}
    return scores[np.ix_([r[q] for q in want_rows], [c[d] for d in want_cols])]


class SplitPairs:
    """A split's chunks and the TF-IDF used to pick passage pairs."""
    def __init__(self, tokenizer, queries, pool, args):
        self.qc = Chunks(tokenizer, query_texts(queries, args.query_context), args.chunk_words, args.max_chunks,
                         args.max_tokens)
        self.pc = Chunks(tokenizer, pool, args.chunk_words, args.max_chunks, args.max_tokens)
        vocab = Vocab()
        self.xp, idf = chunk_tfidf([vocab.encode(w) for w in self.pc.words])
        self.xq, _ = chunk_tfidf([vocab.encode(w) for w in self.qc.words], idf=idf)

    def candidates(self, first, depth):
        """Per query: the first stage's top-`depth` pool indices (never the query itself)."""
        out = []
        for i, q in enumerate(self.qc.doc_ids):
            order = [j for j in np.argsort(-first[i]) if self.pc.doc_ids[j] != q][:depth]
            out.append(order)
        return out

    def pairs(self, qi, j, k):
        qr, cr = list(self.qc.doc_range(qi)), list(self.pc.doc_range(j))
        sim = (self.xq[qr] @ self.xp[cr].T).toarray()
        flat = np.argsort(-sim, axis=None)[:k]
        return [(qr[f // sim.shape[1]], cr[f % sim.shape[1]]) for f in flat]


@torch.no_grad()
def score_pairs(model, seqs, device, dtype, batch_tokens):
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros(len(seqs), dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * len(seqs[order[end]]) <= batch_tokens:
            end += 1
        idx = order[start:end]
        ids, mask = pad([seqs[i] for i in idx], device)
        with autocast(device, dtype):
            out[idx] = model(ids, mask).float().cpu().numpy()
        start = end
    return out


def rerank_matrix(model, sp, first, sep, args, device, dtype):
    """Re-ranker scores [nq, npool] for the top-depth candidates (NaN elsewhere)."""
    cands = sp.candidates(first, args.rerank_depth)
    seqs, owner = [], []
    for qi, js in enumerate(cands):
        for j in js:
            for a, b in sp.pairs(qi, j, args.pair_k):
                seqs.append(pair_ids(sp.qc.ids[a], sp.pc.ids[b], sep, args.side_tokens))
                owner.append((qi, j))
    s = score_pairs(model, seqs, device, dtype, args.encode_tokens)
    per = {}
    for (qi, j), v in zip(owner, s):
        per.setdefault((qi, j), []).append(v)
    out = np.full(first.shape, np.nan, dtype=np.float32)
    for (qi, j), vals in per.items():
        vals = sorted(vals, reverse=True)
        out[qi, j] = vals[0] if args.pair_agg == "max" else float(np.mean(vals[:2]))
    return out


def combine(first, rr, w):
    """Re-ranked candidates on top (w z(first) + (1-w) z(rerank) within them), the rest underneath."""
    out = zscore_rows(first) - 1000.0
    for i in range(first.shape[0]):
        js = np.where(~np.isnan(rr[i]))[0]
        if len(js) < 2:
            continue
        f, r = first[i, js], rr[i, js]
        zf = (f - f.mean()) / max(f.std(), 1e-9)
        zr = (r - r.mean()) / max(r.std(), 1e-9)
        out[i, js] = w * zf + (1 - w) * zr
    return out


def best_weight(first, rr, qids, pids, gold):
    rows = [(float(w), f1_at_k(rank_from_scores(combine(first, rr, w), qids, pids), gold)["f1"])
            for w in np.round(np.arange(0, 1.0001, 0.1), 2)]
    return max(rows, key=lambda x: (x[1], x[0])), rows


def train(args):
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device, dtype = device_and_amp()
    retriever, tokenizer, info = build_retriever(args)
    model = CrossScorer(retriever).to(device)
    sep = tokenizer.encode(SEP).ids
    trq, trp = load_split(args.data, "train")
    dvq, dvp = load_split(args.data, "dev")
    teq, tep = load_split(args.data, "test")
    # first stage on dev/test
    first = {}
    for split, (q, p) in (("dev", (dvq, dvp)), ("test", (teq, tep))):
        s, rows, cols, _ = first_stage(args.first, split)
        first[split] = align(s, rows, cols, list(q), list(p))
    dv = SplitPairs(tokenizer, dvq, dvp, args)
    gold = {q: dvq[q]["relevant"] for q in dv.qc.doc_ids}

    def dev_eval():
        rr = rerank_matrix(model, dv, first["dev"], sep, args, device, dtype)
        (w, f), _ = best_weight(first["dev"], rr, dv.qc.doc_ids, dv.pc.doc_ids, gold)
        only = f1_at_k(rank_from_scores(combine(first["dev"], rr, 0.0), dv.qc.doc_ids, dv.pc.doc_ids), gold)["f1"]
        return f, only, w, rr

    base = f1_at_k(rank_from_scores(first["dev"], dv.qc.doc_ids, dv.pc.doc_ids), gold)
    log({"first_stage": str(args.first), "dev_f1@7_first": round(base["f1"], 4),
         **{k: round(v, 4) for k, v in recall_at(first["dev"], dv.qc.doc_ids, dv.pc.doc_ids, gold,
                                                   (7, args.rerank_depth)).items()}, **info["parameters"]})
    history, best = [], {"f1": -1.0}
    if args.epochs > 0 and args.method != "frozen":
        tr_qc = Chunks(tokenizer, query_texts(trq, args.query_context), args.chunk_words, args.max_chunks, args.max_tokens)
        tr_pc = Chunks(tokenizer, trp, args.chunk_words, args.max_chunks, args.max_tokens)
        bm25, _, _ = lexical_scores(args, "train", trq, trp)
        pairs, pst, _ = mine_pairs(tr_qc, tr_pc, trq, bm25, args, rng)
        pairs = [p for p in pairs if p["negs"]]
        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs = rng.sample(pairs, args.max_pairs)
        log({"rerank_train_groups": len(pairs), **pst})
        rel_of = {q: set(v["relevant"] or []) for q, v in trq.items()}
        lr = args.lr_full if args.method in ("full", "partial") else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        per_epoch = max(1, len(pairs) // args.groups)
        total = per_epoch * args.epochs
        warm = max(1, int(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.05, (total - s) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(pairs)
            losses = []
            for s in range(0, len(pairs) - args.groups + 1, args.groups):
                seqs, size = [], None
                for p in pairs[s:s + args.groups]:
                    negs = list(p["negs"][:3])
                    while True:                                      # one random passage of an uncited case
                        r = rng.randrange(len(tr_pc))
                        if tr_pc.doc_ids[tr_pc.owner[r]] not in rel_of[p["query"]]:
                            break
                    group = [p["p"]] + negs + [r]
                    group += [group[-1]] * (5 - len(group))
                    size = len(group)
                    seqs += [pair_ids(tr_qc.ids[p["q"]], tr_pc.ids[c], sep, args.side_tokens) for c in group]
                ids, mask = pad(seqs, device)
                with autocast(device, dtype):
                    logits = model(ids, mask).float().view(-1, size)
                loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                losses.append(float(loss))
                if step % 50 == 0:
                    log({"epoch": epoch, "step": step, "of": total, "loss": round(float(np.mean(losses[-50:])), 4),
                         "minutes": round((time.time() - started) / 60, 1)})
            f, only, w, _ = dev_eval()
            row = {"epoch": epoch, "loss": round(float(np.mean(losses or [0])), 4), "dev_f1@7_fused": round(f, 4),
                   "dev_f1@7_rerank_only": round(only, 4), "weight_first": w,
                   "minutes": round((time.time() - started) / 60, 1)}
            log(row)
            history.append(row)
            if f > best["f1"]:
                best = {"f1": f, "epoch": epoch, "state": {k: v.clone() for k, v in trainable_state(model).items()}}
        model.load_state_dict(best["state"], strict=False)
    torch.save({"args": vars(args), "state": trainable_state(model)}, out / "model.pt")
    train_minutes = round((time.time() - started) / 60, 1)
    peak = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None
    # final: dev weight, test submission
    t0 = time.time()
    rr_dev = rerank_matrix(model, dv, first["dev"], sep, args, device, dtype)
    (w, f), curve = best_weight(first["dev"], rr_dev, dv.qc.doc_ids, dv.pc.doc_ids, gold)
    dev_seconds = time.time() - t0
    final_dev = combine(first["dev"], rr_dev, w)
    te = SplitPairs(tokenizer, teq, tep, args)
    rr_test = rerank_matrix(model, te, first["test"], sep, args, device, dtype)
    final_test = combine(first["test"], rr_test, w)
    write_submission(rank_from_scores(final_test, te.qc.doc_ids, te.pc.doc_ids), teq, tep, out / "submission")
    for split, mat, sp in (("dev", final_dev, dv), ("test", final_test, te)):
        _, _, _, lex = first_stage(args.first, split)
        extra = {"lexical": lex} if lex is not None else {}
        np.savez_compressed(out / f"scores_{split}.npz", neural=mat, fused=mat, query_ids=np.array(sp.qc.doc_ids),
                            pool_ids=np.array(sp.pc.doc_ids), **extra)
    res = {"dev": {"first_stage": base,
                   "fused": f1_at_k(rank_from_scores(final_dev, dv.qc.doc_ids, dv.pc.doc_ids), gold),
                   "rerank_only": f1_at_k(rank_from_scores(combine(first["dev"], rr_dev, 0.0), dv.qc.doc_ids,
                                                           dv.pc.doc_ids), gold),
                   "weight_curve": {str(k): round(v, 4) for k, v in curve}},
           "weight_first": w}
    summary = {"stage": "rerank", "method": args.method, "architecture": info["architecture"], "first": str(args.first),
               "rerank_depth": args.rerank_depth, "pair_k": args.pair_k, "pair_agg": args.pair_agg,
               "history": history, "best_epoch": best.get("epoch"), "parameters": info["parameters"],
               "train_minutes": train_minutes, "peak_gpu_gib": peak, "dev_rerank_seconds": round(dev_seconds, 1),
               **res, "minutes": round((time.time() - started) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"dev F1@7: first stage {base['f1']:.4f} -> re-ranked {res['dev']['fused']['f1']:.4f} (first-stage weight {w}; "
        f"re-ranker alone {res['dev']['rerank_only']['f1']:.4f}) -> {out}/summary.json")


def add_args(p):
    from .neural import add_args as retriever_args
    retriever_args(p)
    p.add_argument("--first", required=True, help="first-stage run directory (retriever, lexical or combine)")
    p.add_argument("--rerank-depth", type=int, default=30)
    p.add_argument("--pair-k", type=int, default=4, help="passage pairs scored per candidate")
    p.add_argument("--pair-agg", default="max", choices=["max", "mean2"])
    p.add_argument("--side-tokens", type=int, default=256, help="tokens per side of a pair")
    p.add_argument("--groups", type=int, default=8, help="(positive + 4 negatives) groups per step")
