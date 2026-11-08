"""Phase-1 models as prior-case retrievers (bi-encoder over passages), trained contrastively; KD; fusion.

Representation. A case is cut into word chunks (--chunk-words 200, up to --max-chunks 64, evenly spaced when
longer). Each chunk is encoded by the Phase-1 model (any of the four architectures; decoder-only models causal by
default, --mode bidirectional to drop the mask; the encoder stack of the encoder-decoder) and pooled (--pool
mean | last | max) into a unit vector. A query-candidate score aggregates the chunk-pair cosines (--agg):
  maxsim      for every query chunk the best candidate chunk, averaged over query chunks (late interaction)
  maxsim_top  the same, averaged over the 8 best query chunks only (a citation is discussed in a few passages)
  max         the single best chunk pair
  mean        cosine of the mean chunk vectors (one vector per case)
All are computed once embeddings exist; the dev set chooses (rank command re-scores without re-encoding).

Training signal. Citation links are document-level; the passages that carry them are found lexically: for every
(query, cited case) the query chunk / candidate chunk pairs with the highest word 2-3-gram TF-IDF cosine (quoted
or paraphrased holdings) become positive pairs. Negatives: in-batch, plus hard negatives = the most similar
chunks of the top BM25 candidates that are NOT cited (--negatives bm25 | random | none). Candidates relevant to
another query of the batch are masked out (no false negatives). Loss: InfoNCE at temperature --tau.

Knowledge distillation (--teacher RUN): the teacher scores the same batch; the student matches the teacher's
similarity distribution over the batch's candidates (KL at temperature T, mixed with InfoNCE by --kd-alpha), and
with --kd-embed w also the teacher's chunk vectors (MSE; needs the same width, e.g. a truncated dense student).

Final ranking: neural scores alone, and fused with the lexical n-gram BM25 (z-scores, weight chosen on dev).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase2.model import Encoder, load_phase1
from phase2.peft import count_parameters
from phase2.run import autocast, device_and_amp
from phase3.peft import apply_technique

from .data import (K, f1_at_k, load_split, normalize, query_texts, rank_from_scores, recall_at, word_chunks,
                   write_submission, zscore_rows)
from .lexical import FusedLexical, SplitIndex, Vocab, chunk_tfidf

LR_DEFAULT = {"lora": 3e-4, "rslora": 3e-4, "qlora": 3e-4, "lorma": 3e-4, "adalora": 1e-3, "ia3": 3e-3,
              "prefix": 1e-3, "prompt": 3e-3, "adapter": 1e-3}
AGGS = ("maxsim", "maxsim_top", "max", "mean")


def log(msg):
    print(msg if isinstance(msg, str) else json.dumps(msg, default=str), flush=True)


# --------------------------------------------------------------------------- model
class Retriever(nn.Module):
    def __init__(self, encoder, pool="mean", proj=0):
        super().__init__()
        self.encoder, self.pool = encoder, pool
        self.proj = nn.Linear(encoder.dim, proj) if proj else None
        self.out_dim = proj or encoder.dim

    def forward(self, ids, mask):
        v = self.pooled(ids, mask)
        if self.proj is not None:
            v = self.proj(v)
        return F.normalize(v.float(), dim=-1)

    def pooled(self, ids, mask):
        """Pooled hidden states [B, dim] (not normalised)."""
        from phase6.model import masks, norm_mask
        enc = self.encoder
        if enc.mode == "causal":                       # right padding never reaches a real token
            h = enc(ids)
        else:
            extra = (enc.prompt.size(0) if enc.prompt is not None else 0) + \
                (enc.prefix.length if getattr(enc, "prefix", None) is not None else 0)
            full = torch.cat((torch.ones(ids.size(0), extra, dtype=torch.bool, device=ids.device), mask), 1)
            with masks(src=norm_mask(full)):
                h = enc(ids)
        m = mask.unsqueeze(-1).to(h.dtype)
        if self.pool == "mean":
            v = (h * m).sum(1) / m.sum(1).clamp_min(1)
        elif self.pool == "last":
            v = h[torch.arange(h.size(0), device=h.device), mask.sum(1) - 1]
        else:
            v = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        return v


def build_retriever(args):
    lm, tokenizer, resolved = load_phase1(args.checkpoint, args.project_root)
    if getattr(args, "random_init", False):
        from phase1.models import build_model
        lm = build_model(resolved["model"])
    encoder = Encoder(lm, mode=args.mode, prompt_tokens=args.prompt_tokens if args.method == "prompt" else 0,
                      gradient_checkpointing=args.gradient_checkpointing)
    del lm
    kept = None
    if args.keep_layers:
        n, k = len(encoder.blocks), args.keep_layers
        kept = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)}) if args.layer_pick == "spread" and 1 < k < n \
            else list(range(min(k, n)))
        encoder.blocks = nn.ModuleList([encoder.blocks[i] for i in kept])
    peft, factory = apply_technique(encoder, args.method, args)
    model = Retriever(encoder, args.pool, args.proj)
    if model.proj is not None:
        for p in model.proj.parameters():
            p.requires_grad = True
    info = {"checkpoint": str(args.checkpoint), "architecture": encoder.architecture, "mode": encoder.mode,
            "layers": len(encoder.blocks), "kept_layers": kept, "peft": peft, "parameters": count_parameters(model),
            "random_init": bool(getattr(args, "random_init", False))}
    model.allocator_factory = factory
    return model, tokenizer, info


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names or k.endswith(".mask")}


def load_retriever(run_dir, device=None):
    payload = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    model, tokenizer, _ = build_retriever(args)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if unexpected or [m for m in missing if m in trainable]:
        raise ValueError(f"{run_dir}: state does not fit: missing={missing[:4]} unexpected={unexpected[:4]}")
    if device is not None:
        model.to(device)
    return model.eval(), tokenizer, args


# --------------------------------------------------------------------------- chunks and embeddings
class Chunks:
    """Every chunk of a set of documents: token ids, normalised words, owner document."""
    def __init__(self, tokenizer, texts, chunk_words, max_chunks, max_tokens):
        self.doc_ids, self.offsets, raw = list(texts), [0], []
        for d in self.doc_ids:
            parts = word_chunks(texts[d], chunk_words, max_chunks)
            raw.extend(parts)
            self.offsets.append(len(raw))
        self.offsets = np.array(self.offsets)
        enc = tokenizer.encode_batch([" " + r for r in raw])
        eos = tokenizer.token_to_id("<eos>") or 1
        self.ids = [(e.ids[:max_tokens] or [eos]) for e in enc]
        self.words = [normalize(r) for r in raw]
        self.owner = np.repeat(np.arange(len(self.doc_ids)), np.diff(self.offsets))

    def __len__(self):
        return len(self.ids)

    def doc_range(self, i):
        return range(self.offsets[i], self.offsets[i + 1])


def pad(seqs, device, pad_id=0):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return ids.to(device), mask.to(device)


@torch.no_grad()
def encode(model, seqs, device, dtype, batch_tokens=65536):
    """Unit vectors [len(seqs), dim] (float16); batches of similar length, padded tokens <= batch_tokens."""
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros((len(seqs), model.out_dim), dtype=np.float16)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * len(seqs[order[end]]) <= batch_tokens:
            end += 1
        idx = order[start:end]
        ids, mask = pad([seqs[j] for j in idx], device)
        with autocast(device, dtype):
            v = model(ids, mask)
        out[idx] = v.float().cpu().numpy().astype(np.float16)
        start = end
    return out


def doc_scores(q_vec, q_chunks, p_vec, p_chunks, agg, device, top=8):
    """[n queries, n candidates] from chunk vectors."""
    qv = torch.tensor(q_vec, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pv = torch.tensor(p_vec, device=device, dtype=qv.dtype)
    n_docs = len(p_chunks.doc_ids)
    if agg == "mean":
        def means(v, ch):
            m = torch.stack([v[ch.offsets[i]:ch.offsets[i + 1]].float().mean(0) for i in range(len(ch.doc_ids))])
            return F.normalize(m, dim=-1)
        return (means(qv, q_chunks) @ means(pv, p_chunks).T).cpu().numpy()
    width = int(np.diff(p_chunks.offsets).max())
    padded = torch.zeros(n_docs, width, pv.size(1), device=device, dtype=pv.dtype)
    valid = torch.zeros(n_docs, width, dtype=torch.bool, device=device)
    for i in range(n_docs):
        a, b = p_chunks.offsets[i], p_chunks.offsets[i + 1]
        padded[i, :b - a], valid[i, :b - a] = pv[a:b], True
    flat = padded.view(-1, pv.size(1))
    out = np.zeros((len(q_chunks.doc_ids), n_docs), dtype=np.float32)
    for qi in range(len(q_chunks.doc_ids)):
        a, b = q_chunks.offsets[qi], q_chunks.offsets[qi + 1]
        s = (qv[a:b] @ flat.T).float().view(b - a, n_docs, width)
        best = s.masked_fill(~valid[None], -2.0).max(-1).values               # [query chunks, docs]
        if agg == "maxsim":
            out[qi] = best.mean(0).cpu().numpy()
        elif agg == "maxsim_top":
            out[qi] = best.topk(min(top, best.size(0)), dim=0).values.mean(0).cpu().numpy()
        else:
            out[qi] = best.max(0).values.cpu().numpy()
    return out


def split_embeddings(model, tokenizer, queries, pool, args, device, dtype):
    qc = Chunks(tokenizer, query_texts(queries, getattr(args, "query_context", None)), args.chunk_words, args.max_chunks,
                args.max_tokens)
    pc = Chunks(tokenizer, pool, args.chunk_words, args.max_chunks, args.max_tokens)
    t0 = time.time()
    qv = encode(model, qc.ids, device, dtype, args.encode_tokens)
    pv = encode(model, pc.ids, device, dtype, args.encode_tokens)
    return qc, pc, qv, pv, time.time() - t0


# --------------------------------------------------------------------------- training pairs
def mine_pairs(qc, pc, queries, bm25, args, rng, qv=None, pv=None):
    """Positive passage pairs of cited cases and hard-negative passages, see the module docstring. With
    --negatives fused, bm25 holds lexical+neural scores and (qv, pv given) the hard passages are the ones the
    current retriever finds most similar (second-round / ANCE-style negatives)."""
    vocab = Vocab()
    xq_tok = [vocab.encode(w) for w in qc.words]
    xp_tok = [vocab.encode(w) for w in pc.words]
    xp, idf = chunk_tfidf(xp_tok)
    xq, _ = chunk_tfidf(xq_tok, idf=idf)
    pos_index = {d: i for i, d in enumerate(pc.doc_ids)}
    pairs, links, empty = [], 0, 0
    for qi, q in enumerate(qc.doc_ids):
        rel = [pos_index[c] for c in (queries[q]["relevant"] or []) if c in pos_index and c != q]
        if not rel:
            continue
        qr = list(qc.doc_range(qi))
        qmat = xq[qr]
        hard = []
        if args.negatives in ("bm25", "fused"):
            order = np.argsort(-bm25[qi])
            relset = set(rel)
            hard = [j for j in order if j not in relset and pc.doc_ids[j] != q][:args.hard_docs]
        hard_rows = [r for j in hard for r in pc.doc_range(j)]
        hard_mat = xp[hard_rows] if hard_rows else None
        for c in rel:
            links += 1
            cr = list(pc.doc_range(c))
            s = (qmat @ xp[cr].T).toarray()
            if s.max() <= args.min_pair_sim:
                empty += 1
                continue
            used = set()
            for flat in np.argsort(-s, axis=None):
                a, b = divmod(int(flat), s.shape[1])
                if s[a, b] <= args.min_pair_sim or len(used) >= args.pairs_per_link:
                    break
                if a in used:
                    continue
                used.add(a)
                negs = []
                if hard_mat is not None and qv is not None:
                    sims = pv[hard_rows].astype(np.float32) @ qv[qr[a]].astype(np.float32)
                    negs = [hard_rows[k] for k in np.argsort(-sims)[:3]]
                elif hard_mat is not None:
                    sims = (qmat[a] @ hard_mat.T).toarray().ravel()
                    negs = [hard_rows[k] for k in np.argsort(-sims)[:3]]
                elif args.negatives == "random":
                    negs = [rng.randrange(len(pc)) for _ in range(3)]
                    negs = [k for k in negs if pc.doc_ids[pc.owner[k]] not in (queries[q]["relevant"] or [])]
                pairs.append({"q": qr[a], "p": cr[b], "negs": negs, "query": q, "doc": pc.doc_ids[c],
                              "sim": float(s[a, b])})
    return pairs, {"links": links, "links_without_shared_phrases": empty, "pairs": len(pairs)}, (xq, xp)


# --------------------------------------------------------------------------- list-level KD from a fused teacher
def teacher_fused_scores(run, trq, trp, lex, device, dtype):
    """The teacher run's own ranking of the training pool: its neural scores (its chunking and aggregation) fused
    with the lexical scores at its dev-chosen weight."""
    tmodel, ttok, targs = load_retriever(run, device)
    tsum = json.loads((Path(run) / "summary.json").read_text())
    qc = Chunks(ttok, query_texts(trq, getattr(targs, "query_context", None)), targs.chunk_words, targs.max_chunks,
                targs.max_tokens)
    pc = Chunks(ttok, trp, targs.chunk_words, targs.max_chunks, targs.max_tokens)
    qv = encode(tmodel, qc.ids, device, dtype, targs.encode_tokens)
    pv = encode(tmodel, pc.ids, device, dtype, targs.encode_tokens)
    neu = doc_scores(qv, qc, pv, pc, tsum["agg"], device, getattr(targs, "top_chunks", 8))
    del tmodel
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fuse(neu, lex, tsum["fusion_weight"]), {"run": str(run), "agg": tsum["agg"],
                                                    "fusion_weight": tsum["fusion_weight"]}


def list_examples(qc, pc, teacher, tfidf, args):
    """Per training query: the teacher's top-M candidates with its scores, the query passages most related to
    them and, per candidate, its passages most similar to those (what the student scores in the list loss)."""
    xq, xp = tfidf
    out = []
    for qi, q in enumerate(qc.doc_ids):
        order = [j for j in np.argsort(-teacher[qi]) if pc.doc_ids[j] != q][:args.list_size]
        if len(order) < 2:
            continue
        qr = list(qc.doc_range(qi))
        rows = [r for j in order for r in pc.doc_range(j)]
        sim = (xq[qr] @ xp[rows].T).toarray()
        qpick = [qr[i] for i in np.argsort(-sim.max(1))[:args.list_qchunks]]
        qsel = [qr.index(x) for x in qpick]
        cands, at = [], 0
        for j in order:
            n = len(pc.doc_range(j))
            block = sim[qsel, at:at + n].max(0)
            cands.append([pc.doc_range(j)[k] for k in np.argsort(-block)[:args.list_cchunks]])
            at += n
        out.append({"q": qpick, "cands": cands, "teacher": np.asarray(teacher[qi][order], dtype=np.float32)})
    return out


def list_loss(model, ex_batch, qc, pc, args, device, dtype):
    """KL(teacher || student) over each query's candidate list; student score = mean over the chosen query
    passages of the best passage of the candidate."""
    total = 0.0
    for ex in ex_batch:
        width = args.list_cchunks
        flat = [c for cs in ex["cands"] for c in (cs + [cs[-1]] * (width - len(cs)))]
        q_ids, q_mask = pad([qc.ids[i] for i in ex["q"]], device)
        c_ids, c_mask = pad([pc.ids[i] for i in flat], device)
        with autocast(device, dtype):
            a, c = model(q_ids, q_mask), model(c_ids, c_mask)
        sim = (a @ c.T).view(len(ex["q"]), len(ex["cands"]), width).max(-1).values.mean(0)
        t = torch.tensor(ex["teacher"], device=device)
        total = total + F.kl_div(F.log_softmax(sim / args.tau, -1), F.log_softmax(t / args.kd_list_temperature, -1),
                                 log_target=True, reduction="sum")
    return total / max(1, len(ex_batch))


# --------------------------------------------------------------------------- lexical scores
def lexical_scores(args, split, queries, pool):
    """Fused n-gram BM25 scores of a split (the tuned configuration from the lexical run when given)."""
    cfg = None
    if args.lexical and (Path(args.lexical) / "lexical.json").exists():
        cfg = json.loads((Path(args.lexical) / "lexical.json").read_text())
    if cfg is None:
        cfg = {"per_n": {"2": {"k1": 1.2, "b": 0.75, "max_df": 0.5}}, "weights": {"2": 1.0}}
    index = SplitIndex(queries, pool)
    return FusedLexical(cfg).scores(index), index, cfg


def fuse(neu, lex, w):
    return w * zscore_rows(lex) + (1 - w) * zscore_rows(neu) if lex is not None else neu


def best_fusion(neu, lex, qids, pids, gold):
    rows = []
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        f = f1_at_k(rank_from_scores(fuse(neu, lex, w), qids, pids), gold)["f1"]
        rows.append((float(w), f))
    return max(rows, key=lambda x: (x[1], -abs(x[0] - 0.5))), rows


# --------------------------------------------------------------------------- train + rank
def train(args):
    started = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device, dtype = device_and_amp()
    model, tokenizer, info = build_retriever(args)
    if args.init_from:                                  # continue from a finished run (same backbone + method)
        payload = torch.load(Path(args.init_from) / "model.pt", map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["state"], strict=False)
        if unexpected:
            raise SystemExit(f"--init-from {args.init_from}: different model/method ({unexpected[:3]})")
        info["init_from"] = str(args.init_from)
    model.to(device)
    log(info)
    trq, trp = load_split(args.data, "train")
    dvq, dvp = load_split(args.data, "dev")
    teq, tep = load_split(args.data, "test")
    teacher = None
    if args.teacher:
        from phase6.model import same_tokenizer
        teacher, ttok, _ = load_retriever(args.teacher, device)
        if not same_tokenizer(ttok, tokenizer):
            raise SystemExit("teacher and student must share the Phase-1 tokenizer")
        for p in teacher.parameters():
            p.requires_grad = False
        info["teacher"] = {"run": args.teacher, "architecture": teacher.encoder.architecture,
                           "parameters": sum(p.numel() for p in teacher.parameters()),
                           "layers": len(teacher.encoder.blocks)}
    dv_qc = Chunks(tokenizer, query_texts(dvq, args.query_context), args.chunk_words, args.max_chunks, args.max_tokens)
    dv_pc = Chunks(tokenizer, dvp, args.chunk_words, args.max_chunks, args.max_tokens)
    dv_gold = {q: v["relevant"] for q, v in dvq.items()}

    def dev_eval():
        """Best dev F1@7 over the aggregations (the same choice the final ranking makes)."""
        qv = encode(model, dv_qc.ids, device, dtype, args.encode_tokens)
        pv = encode(model, dv_pc.ids, device, dtype, args.encode_tokens)
        return max(f1_at_k(rank_from_scores(doc_scores(qv, dv_qc, pv, dv_pc, a, device, args.top_chunks), dv_qc.doc_ids,
                                            dv_pc.doc_ids), dv_gold)["f1"] for a in AGGS)

    history, best = [], {"f1": -1.0, "state": None}
    if args.epochs > 0 and args.method != "frozen":
        t0 = time.time()
        tr_qc = Chunks(tokenizer, query_texts(trq, args.query_context), args.chunk_words, args.max_chunks, args.max_tokens)
        tr_pc = Chunks(tokenizer, trp, args.chunk_words, args.max_chunks, args.max_tokens)
        bm25 = qv0 = pv0 = None
        if args.negatives in ("bm25", "fused") or args.kd_list:
            bm25, _, _ = lexical_scores(args, "train", trq, trp)
        lex_train = bm25
        if args.negatives == "fused":                   # hard negatives the CURRENT retriever confuses
            qv0 = encode(model, tr_qc.ids, device, dtype, args.encode_tokens)
            pv0 = encode(model, tr_pc.ids, device, dtype, args.encode_tokens)
            bm25 = zscore_rows(bm25) + zscore_rows(doc_scores(qv0, tr_qc, pv0, tr_pc, "maxsim_top", device,
                                                               args.top_chunks))
        pairs, pst, tfidf = mine_pairs(tr_qc, tr_pc, trq, bm25, args, rng, qv0, pv0)
        lists = []
        if args.kd_list:
            tscores, info["list_teacher"] = teacher_fused_scores(args.kd_list, trq, trp, lex_train, device, dtype)
            lists = list_examples(tr_qc, tr_pc, tscores, tfidf, args)
            log({"list_kd_queries": len(lists), **info["list_teacher"]})
        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs = rng.sample(pairs, args.max_pairs)
        log({"train_query_chunks": len(tr_qc), "train_pool_chunks": len(tr_pc), **pst,
             "mining_s": round(time.time() - t0)})
        rel_of = {q: set(v["relevant"] or []) for q, v in trq.items()}
        full_like = args.method in ("full", "partial") or args.random_init
        lr = args.lr_full if full_like else (args.lr_peft or LR_DEFAULT.get(args.method, 3e-4))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(pairs) // args.batch)
        total = steps_per_epoch * args.epochs
        warm = max(1, int(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.05, (total - s) / max(1, total - warm))))
        factory = getattr(model, "allocator_factory", None)
        allocator = factory(model.encoder, args, total) if factory else None
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        f0 = dev_eval()
        log({"epoch": 0, "dev_f1@7": round(f0, 4)})
        best = {"f1": f0, "state": {k: v.clone() for k, v in trainable_state(model).items()}, "epoch": 0}
        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(pairs)
            losses = []
            for s in range(0, len(pairs) - args.batch + 1, args.batch):
                batch = pairs[s:s + args.batch]
                negs = [rng.choice(p["negs"]) if p["negs"] else None for p in batch]
                cand_chunks = [p["p"] for p in batch] + [n for n in negs if n is not None]
                cand_docs = [p["doc"] for p in batch] + [tr_pc.doc_ids[tr_pc.owner[n]] for n in negs if n is not None]
                block = torch.zeros(len(batch), len(cand_chunks), dtype=torch.bool)
                for i, p in enumerate(batch):
                    for j, d in enumerate(cand_docs):
                        if j != i and (d in rel_of[p["query"]] or d == p["query"]):
                            block[i, j] = True                      # a cited case is never a negative
                block = block.to(device)
                q_ids, q_mask = pad([tr_qc.ids[p["q"]] for p in batch], device)
                c_ids, c_mask = pad([tr_pc.ids[c] for c in cand_chunks], device)
                with autocast(device, dtype):
                    a = model(q_ids, q_mask)
                    c = model(c_ids, c_mask)
                logits = (a @ c.T / args.tau).masked_fill(block, -1e4)
                target = torch.arange(len(batch), device=device)
                loss = F.cross_entropy(logits, target)
                if teacher is not None:
                    with torch.no_grad(), autocast(device, dtype):
                        ta, tc = teacher(q_ids, q_mask), teacher(c_ids, c_mask)
                    t_logits = (ta @ tc.T / args.tau).masked_fill(block, -1e4)
                    tau = args.kd_temperature
                    kl = F.kl_div(F.log_softmax(logits / tau, -1), F.log_softmax(t_logits / tau, -1),
                                  log_target=True, reduction="batchmean") * tau * tau
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kl
                    if args.kd_embed and ta.size(-1) == a.size(-1):
                        loss = loss + args.kd_embed * (F.mse_loss(a, ta) + F.mse_loss(c, tc))
                if lists:
                    loss = loss + args.kd_list_weight * list_loss(model, rng.sample(lists, min(args.list_queries,
                                                                                              len(lists))),
                                                                  tr_qc, tr_pc, args, device, dtype)
                if allocator is not None:
                    loss = loss + 0.1 * allocator.orth_penalty()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if allocator is not None:
                    allocator.step(step)
                losses.append(float(loss))
                if step % 50 == 0:
                    log({"epoch": epoch, "step": step, "of": total, "loss": round(float(np.mean(losses[-50:])), 4),
                         "minutes": round((time.time() - started) / 60, 1)})
            f = dev_eval()
            row = {"epoch": epoch, "loss": round(float(np.mean(losses or [0])), 4), "dev_f1@7": round(f, 4),
                   "minutes": round((time.time() - started) / 60, 1)}
            log(row)
            history.append(row)
            if f > best["f1"]:
                best = {"f1": f, "state": {k: v.clone() for k, v in trainable_state(model).items()}, "epoch": epoch}
        model.load_state_dict(best["state"], strict=False)
    train_minutes = round((time.time() - started) / 60, 1)
    peak = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None
    torch.save({"args": vars(args), "state": trainable_state(model)}, out / "model.pt")
    del teacher
    res = rank_run(model, tokenizer, args, out, device, dtype, (dvq, dvp), (teq, tep))
    summary = {"stage": "retriever", "method": args.method, "architecture": info["architecture"], "mode": info["mode"],
               "pool": args.pool, "negatives": args.negatives, "teacher": info.get("teacher"),
               "list_teacher": info.get("list_teacher"), "init_from": info.get("init_from"),
               "query_context": args.query_context, "history": history,
               "best_epoch": best.get("epoch"), "parameters": info["parameters"], "layers": info["layers"],
               "train_minutes": train_minutes, "peak_gpu_gib": peak, **res,
               "minutes": round((time.time() - started) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"dev F1@7: neural {res['dev']['neural']['f1']:.4f} ({res['agg']}) | fused {res['dev']['fused']['f1']:.4f} "
        f"(lexical weight {res['fusion_weight']}) | lexical {res['dev']['lexical']['f1']:.4f} -> {out}/summary.json")


def rank_run(model, tokenizer, args, out, device, dtype, dev, test, aggs=AGGS, cache=True):
    """Encode dev and test, choose the aggregation and the lexical fusion weight on dev, write both submissions."""
    res, emb = {}, {}
    for name, (q, p) in (("dev", dev), ("test", test)):
        qc, pc, qv, pv, secs = split_embeddings(model, tokenizer, q, p, args, device, dtype)
        emb[name] = (qc, pc, qv, pv)
        res[f"{name}_encode_seconds"] = round(secs, 1)
        if cache:
            np.savez_compressed(out / f"emb_{name}.npz", q=qv, p=pv)
    qc, pc, qv, pv = emb["dev"]
    gold = {d: dev[0][d]["relevant"] for d in qc.doc_ids}
    per_agg = {a: doc_scores(qv, qc, pv, pc, a, device, args.top_chunks) for a in aggs}
    f_agg = {a: f1_at_k(rank_from_scores(s, qc.doc_ids, pc.doc_ids), gold) for a, s in per_agg.items()}
    agg = max(f_agg, key=lambda a: f_agg[a]["f1"])
    lex_dev, _, cfg = lexical_scores(args, "dev", *dev)
    (w, _), curve = best_fusion(per_agg[agg], lex_dev, qc.doc_ids, pc.doc_ids, gold)
    fused = fuse(per_agg[agg], lex_dev, w)
    res["agg"], res["fusion_weight"], res["lexical_config"] = agg, w, cfg
    res["dev"] = {"neural": f1_at_k(rank_from_scores(per_agg[agg], qc.doc_ids, pc.doc_ids), gold),
                  "fused": f1_at_k(rank_from_scores(fused, qc.doc_ids, pc.doc_ids), gold),
                  "lexical": f1_at_k(rank_from_scores(lex_dev, qc.doc_ids, pc.doc_ids), gold),
                  "per_agg": {a: round(v["f1"], 4) for a, v in f_agg.items()},
                  "fusion_curve": {str(k): round(v, 4) for k, v in curve},
                  "neural_recall": recall_at(per_agg[agg], qc.doc_ids, pc.doc_ids, gold),
                  "fused_recall": recall_at(fused, qc.doc_ids, pc.doc_ids, gold)}
    np.savez_compressed(out / "scores_dev.npz", neural=per_agg[agg], lexical=lex_dev, fused=fused,
                        query_ids=np.array(qc.doc_ids), pool_ids=np.array(pc.doc_ids))
    tqc, tpc, tqv, tpv = emb["test"]
    neu_t = doc_scores(tqv, tqc, tpv, tpc, agg, device, args.top_chunks)
    lex_t, _, _ = lexical_scores(args, "test", *test)
    np.savez_compressed(out / "scores_test.npz", neural=neu_t, lexical=lex_t, fused=fuse(neu_t, lex_t, w),
                        query_ids=np.array(tqc.doc_ids), pool_ids=np.array(tpc.doc_ids))
    write_submission(rank_from_scores(fuse(neu_t, lex_t, w), tqc.doc_ids, tpc.doc_ids), test[0], test[1],
                     out / "submission")
    write_submission(rank_from_scores(neu_t, tqc.doc_ids, tpc.doc_ids), test[0], test[1], out / "submission_neural")
    res["index_bytes_test"] = int(tpv.nbytes + tqv.nbytes)
    res["chunks"] = {"dev": [len(qc), len(pc)], "test": [len(tqc), len(tpc)]}
    return res


def add_args(p):
    p.add_argument("--data", required=True, help="Phase_7_PCR directory")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--lexical", help="lexical run directory (lexical.json): fused BM25 for hard negatives and fusion")
    p.add_argument("--method", default="lorma", choices=["lorma", "lora", "rslora", "qlora", "adalora", "ia3", "prefix",
                                                        "prompt", "partial", "full", "frozen", "normfit"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--top-k", type=int, default=4, help="partial: top blocks trained")
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--adalora-init-r", type=int, default=12)
    p.add_argument("--adalora-target-r", type=int, default=8)
    p.add_argument("--prefix-length", type=int, default=16)
    p.add_argument("--prefix-hidden", type=int, default=512)
    p.add_argument("--prompt-tokens", type=int, default=16)
    p.add_argument("--mode", default="causal", choices=["causal", "bidirectional"],
                   help="decoder-only models (the encoder-decoder always uses its bidirectional encoder)")
    p.add_argument("--pool", default="mean", choices=["mean", "last", "max"])
    p.add_argument("--proj", type=int, default=0, help="trainable projection to this width (0 = none)")
    p.add_argument("--keep-layers", type=int)
    p.add_argument("--layer-pick", default="spread", choices=["first", "spread"])
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
    p.add_argument("--chunk-words", type=int, default=200)
    p.add_argument("--max-chunks", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--encode-tokens", type=int, default=65536, help="padded tokens per encoding batch")
    p.add_argument("--pairs-per-link", type=int, default=3)
    p.add_argument("--hard-docs", type=int, default=20)
    p.add_argument("--negatives", default="bm25", choices=["bm25", "fused", "random", "none"],
                   help="fused: lexical + the current model's own scores (use with --init-from: second round)")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr-peft", type=float)
    p.add_argument("--lr-full", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--kd-embed", type=float, default=0.0)
    p.add_argument("--kd-list", help="finished retriever run: list-level KD from its FUSED (neural + lexical) ranking")
    p.add_argument("--kd-list-weight", type=float, default=1.0)
    p.add_argument("--kd-list-temperature", type=float, default=1.0)
    p.add_argument("--list-size", type=int, default=16, help="teacher's top candidates per query in the list loss")
    p.add_argument("--list-queries", type=int, default=2, help="queries per step in the list loss")
    p.add_argument("--list-qchunks", type=int, default=4)
    p.add_argument("--list-cchunks", type=int, default=3)
    p.add_argument("--init-from", help="finished run of the same backbone + method to continue from")
    p.add_argument("--query-context", type=int, default=None,
                   help="represent queries by the sentences within N of each <CITATION> mark (default: whole text)")
    p.add_argument("--top-chunks", type=int, default=8, help="query passages averaged by the maxsim_top aggregation")
    p.add_argument("--min-pair-sim", type=float, default=0.0, help="minimum TF-IDF cosine of a positive passage pair")
