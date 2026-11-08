"""Hierarchical rhetorical-role tagger on a Phase-1 language model.

  sentences --(Phase-1 LM + LoRMA/LoRA/full, packed windows)--> token states
            --(pool each sentence's own tokens)--> sentence vectors
            --(+ relative-position and domain embeddings)--> document context model (BiLSTM / transformer / none)
            --(linear head, one per domain optionally)--> emissions [n_sentences, 13]
            --(linear-chain CRF over sentence labels, or softmax)--> roles

Packing: consecutive sentences are concatenated into windows of up to --window tokens, so with a causal
model every sentence representation also sees the sentences before it in the window (no padding
inside a window). --packing isolated encodes each sentence alone (the sentence-only ablation).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from phase2.model import Encoder, load_phase1
from phase2.peft import Adapter, AdapterBlock, LoRALinear, LoRMALinear, apply_method, count_parameters

from .peft import apply_technique

N_DOMAINS = 3          # CL, IT, unknown (teacher-labelled unlabeled text)
DOMAIN_INDEX = {"CL": 0, "IT": 1, None: 2}


# --------------------------------------------------------------------------- CRF over sentence labels
class ChainCRF(nn.Module):
    """Unconstrained linear-chain CRF over K labels (no BIO structure: every transition allowed)."""
    def __init__(self, k):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(k, k))
        self.start = nn.Parameter(torch.zeros(k))
        self.end = nn.Parameter(torch.zeros(k))

    def _alpha(self, em, trans, start):
        alpha = start + em[0]
        alphas = [alpha]
        for i in range(1, em.size(0)):
            alpha = torch.logsumexp(alpha.unsqueeze(1) + trans, 0) + em[i]
            alphas.append(alpha)
        return torch.stack(alphas)

    def nll(self, em, tags):
        """em [T,K] float, tags [T] -> negative log-likelihood / T."""
        em = em.float()
        t = em.size(0)
        score = self.start[tags[0]] + em[0, tags[0]] + self.end[tags[-1]]
        if t > 1:
            score = score + self.trans[tags[:-1], tags[1:]].sum() + em[torch.arange(1, t), tags[1:]].sum()
        logz = torch.logsumexp(self._alpha(em, self.trans, self.start)[-1] + self.end, 0)
        return (logz - score) / t

    def log_marginals(self, em, temperature=1.0):
        """Per-position log posterior [T,K] (forward-backward); differentiable (used for KD)."""
        em = em.float() / temperature
        trans, start, end = self.trans / temperature, self.start / temperature, self.end / temperature
        alpha = self._alpha(em, trans, start)
        beta = [end.expand_as(em[0])]
        for i in range(em.size(0) - 1, 0, -1):
            beta.append(torch.logsumexp(trans + (em[i] + beta[-1]).unsqueeze(0), 1))
        beta = torch.stack(beta[::-1])
        joint = alpha + beta
        return joint - torch.logsumexp(joint, 1, keepdim=True)

    def params(self):
        return {k: v.detach().cpu().float().numpy() for k, v in
                (("trans", self.trans), ("start", self.start), ("end", self.end))}


def viterbi_np(em, crf, offsets=None):
    """em [T,K] numpy, crf dict(trans,start,end) -> best label index list."""
    import numpy as np
    em = em if offsets is None else em + offsets
    trans, start, end = crf["trans"], crf["start"], crf["end"]
    score = start + em[0]
    back = np.empty((len(em), em.shape[1]), dtype=np.int64)
    for i in range(1, len(em)):
        total = score[:, None] + trans
        back[i] = total.argmax(0)
        score = total.max(0) + em[i]
    best = [int((score + end).argmax())]
    for i in range(len(em) - 1, 0, -1):
        best.append(int(back[i, best[-1]]))
    return best[::-1]


# --------------------------------------------------------------------------- adapters from Phase 2
@torch.no_grad()
def merge_adapters(module):
    """Fold LoRMA/LoRA wrappers into plain Linear weights (Phase-2 NER weights become a new base)."""
    merged = 0
    for name, child in list(module.named_modules()):
        for child_name, leaf in list(child.named_children()):
            if isinstance(leaf, LoRMALinear):         # W = (I + s U V) W0
                w = leaf.base.weight.float()
                new = w + leaf.scale * leaf.U.float() @ (leaf.V.float() @ w)
            elif isinstance(leaf, LoRALinear):        # W = W0 + s B A
                w = leaf.base.weight.float()
                new = w + leaf.scale * leaf.B.float() @ leaf.A.float()
            else:
                continue
            linear = leaf.base
            linear.weight.copy_(new.to(linear.weight.dtype))
            setattr(child, child_name, linear)
            merged += 1
    return merged


def phase2_encoder(model_path, project_root, gradient_checkpointing=False):
    """Encoder of a saved Phase-2 NER model (model.pt), with its adapters merged into the weights."""
    import argparse
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**payload["args"])
    lm, tokenizer, _ = load_phase1(a.checkpoint, project_root)
    encoder = Encoder(lm, mode=a.mode, keep_layers=getattr(a, "keep_layers", None),
                      prompt_tokens=a.prompt_tokens if a.method == "prompt" else 0,
                      gradient_checkpointing=gradient_checkpointing)
    apply_method(encoder, a.method, rank=a.rank, alpha=a.alpha, top_k=getattr(a, "top_k", 4),
                 bottleneck=getattr(a, "bottleneck", 64))
    state = {k[len("encoder."):]: v for k, v in payload["state"].items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected or [m for m in missing if "prompt" not in m]:
        raise ValueError(f"Phase-2 encoder weights do not fit: missing={missing[:5]} unexpected={unexpected[:5]}")
    merged = merge_adapters(encoder)
    encoder.prompt = None                  # a Phase-2 soft prompt is NER-specific; drop it
    return encoder, tokenizer, {"phase2_model": str(model_path), "phase2_method": a.method,
                                "phase2_checkpoint": a.checkpoint, "merged_linears": merged}


# --------------------------------------------------------------------------- the tagger
# coarse role groups for the auxiliary head (fine labels of one group are confused with each other most)
COARSE = {"Fact": "Fact", "Issue": "Issue", "ArgumentPetitioner": "Argument", "ArgumentRespondent": "Argument",
          "Statute": "Statute", "Dissent": "Dissent", "PrecedentReliedUpon": "Precedent",
          "PrecedentNotReliedUpon": "Precedent", "PrecedentOverruled": "Precedent", "RulingByLowerCourt": "Ruling",
          "RulingByPresentCourt": "Ruling", "RatioOfTheDecision": "Ratio", "None": "None"}


def coarse_map(labels):
    """fine label index -> coarse group index, and the group names."""
    groups = []
    for l in labels:
        g = COARSE.get(l, l)
        if g not in groups:
            groups.append(g)
    return [groups.index(COARSE.get(l, l)) for l in labels], groups


class RoleTagger(nn.Module):
    def __init__(self, encoder, labels, pooling="meanlast", context="bilstm", hidden=256, layers=1,
                 position_buckets=20, domain_heads=False, dropout=0.2, use_crf=True, two_pass=False,
                 aux_shift=False, aux_coarse=False, tf_layers=2):
        super().__init__()
        self.encoder = encoder
        self.labels = list(labels)
        k = len(labels)
        self.pooling = pooling
        self.two_pass = two_pass
        sent_dim = encoder.dim * (2 if pooling == "meanlast" else 1) * (2 if two_pass else 1)
        self.position_buckets = position_buckets
        self.pos = nn.Embedding(position_buckets, 32) if position_buckets else None
        self.dom = nn.Embedding(N_DOMAINS, 16)
        in_dim = hidden
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(sent_dim, hidden), nn.GELU())
        extra = (32 if self.pos is not None else 0) + 16
        self.context_kind = context
        if context == "bilstm":
            self.context = nn.LSTM(in_dim + extra, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                   dropout=dropout if layers > 1 else 0.0)
            out_dim = 2 * hidden
        elif context == "transformer":
            self.context_in = nn.Linear(in_dim + extra, hidden)
            layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout, batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, max(1, layers) * 2)
            out_dim = hidden
        elif context == "lstm_tf":
            # BiLSTM for order and local continuity, then self-attention over the whole document: a sentence
            # can attend to the speaker introduction or the court's verdict hundreds of sentences away
            self.context = nn.LSTM(in_dim + extra, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                   dropout=dropout if layers > 1 else 0.0)
            layer = nn.TransformerEncoderLayer(2 * hidden, 4, 4 * hidden, dropout, batch_first=True, norm_first=True)
            self.global_context = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
            out_dim = 2 * hidden
        elif context == "none":
            self.context = None
            out_dim = in_dim + extra
        else:
            raise ValueError(context)
        self.drop = nn.Dropout(dropout)
        self.domain_heads = domain_heads
        self.heads = nn.ModuleList([nn.Linear(out_dim, k) for _ in range(2 if domain_heads else 1)])
        self.crf = ChainCRF(k)
        self.use_crf = use_crf
        self.out_dim = out_dim
        # auxiliary heads (built only when used, so older checkpoints load unchanged)
        self.shift_head = nn.Sequential(nn.Linear(3 * out_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)) if aux_shift else None
        if aux_coarse:
            mapping, groups = coarse_map(self.labels)
            self.register_buffer("coarse_index", torch.tensor(mapping), persistent=False)
            self.coarse_head = nn.Linear(out_dim, len(groups))
        else:
            self.coarse_head = None

    # ---- sentence vectors
    def pool(self, piece):
        """piece [n_tokens, d]: one sentence's token states -> sentence vector."""
        if self.pooling == "mean":
            return piece.mean(0)
        if self.pooling == "last":
            return piece[-1]
        if self.pooling == "max":
            return piece.max(0).values
        return torch.cat((piece.mean(0), piece[-1]))

    def contextualize(self, vectors, positions, domain, return_hidden=False):
        """vectors [S,D] -> emissions [S,K] (and the contextual states [S,H] with return_hidden).
        positions: relative positions in [0,1) of each sentence."""
        x = self.proj(vectors.float())
        feats = [x]
        if self.pos is not None:
            bucket = (positions * self.position_buckets).long().clamp(0, self.position_buckets - 1)
            feats.append(self.pos(bucket))
        d = torch.full((x.size(0),), DOMAIN_INDEX.get(domain, 2), dtype=torch.long, device=x.device)
        feats.append(self.dom(d))
        x = torch.cat(feats, -1).unsqueeze(0)
        if self.context_kind == "bilstm":
            with torch.autocast(x.device.type, enabled=False):
                x, _ = self.context(x.float())
        elif self.context_kind == "transformer":
            x = self.context(self.context_in(x))
        elif self.context_kind == "lstm_tf":
            with torch.autocast(x.device.type, enabled=False):
                x, _ = self.context(x.float())
                x = self.global_context(x)
        x = self.drop(x.squeeze(0)).float()
        if not self.domain_heads:
            em = self.heads[0](x)
        elif domain in ("CL", "IT"):
            em = self.heads[DOMAIN_INDEX[domain]](x)
        else:
            em = 0.5 * (self.heads[0](x) + self.heads[1](x))
        return (em, x) if return_hidden else em

    def aux_loss(self, hidden, tags, prev_hidden=None, prev_tag=None, shift_weight=0.0, coarse_weight=0.0):
        """Label-shift prediction (does the role change from the previous sentence?) and coarse-role CE.
        hidden [S,H], tags [S]; prev_hidden/prev_tag: the sentence before the segment, if any."""
        loss = hidden.new_zeros(())
        if self.shift_head is not None and shift_weight:
            h, t = hidden, tags
            if prev_hidden is not None:
                h = torch.cat([prev_hidden.unsqueeze(0), hidden])
                t = torch.cat([prev_tag.view(1), tags])
            if h.size(0) > 1:
                a, b = h[:-1], h[1:]
                logit = self.shift_head(torch.cat([b, a, a * b], -1)).squeeze(-1)
                target = (t[1:] != t[:-1]).float()
                pos = target.mean().clamp(0.02, 0.98)          # shifts are ~13%: balance the two outcomes
                w = torch.where(target > 0, 0.5 / pos, 0.5 / (1 - pos))
                loss = loss + shift_weight * (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * w).mean()
        if self.coarse_head is not None and coarse_weight:
            loss = loss + coarse_weight * F.cross_entropy(self.coarse_head(hidden), self.coarse_index[tags])
        return loss

    def log_probs(self, emissions, temperature=1.0):
        """Per-sentence log posterior: CRF marginals, or softmax without CRF."""
        if self.use_crf:
            return self.crf.log_marginals(emissions, temperature)
        return F.log_softmax(emissions.float() / temperature, -1)


def build_encoder(args):
    """(encoder, tokenizer, info) from a Phase-1 checkpoint or a Phase-2 model, with the PEFT method applied."""
    if getattr(args, "init_phase2", None):
        encoder, tokenizer, info = phase2_encoder(args.init_phase2, args.project_root, args.gradient_checkpointing)
        if args.keep_layers:
            encoder.blocks = nn.ModuleList(list(encoder.blocks)[:args.keep_layers])
        if args.mode != encoder.mode and not encoder.seq2seq:
            encoder.mode = args.mode
    else:
        lm, tokenizer, _ = load_phase1(args.checkpoint, args.project_root)
        encoder = Encoder(lm, mode=args.mode, keep_layers=args.keep_layers,
                          prompt_tokens=args.prompt_tokens if args.method == "prompt" else 0,
                          gradient_checkpointing=args.gradient_checkpointing)
        del lm
        info = {"checkpoint": str(args.checkpoint)}
    for module in encoder.modules():            # adapter blocks inherited from Phase 2 stay frozen
        if isinstance(module, (Adapter, AdapterBlock)):
            for p in module.parameters():
                p.requires_grad = False
    if args.method == "prompt" and encoder.prompt is None:          # Phase-2 start: add the prompt now
        encoder.prompt = nn.Parameter(torch.randn(args.prompt_tokens, encoder.dim) * 0.02)
    prompt = encoder.prompt.size(0) if encoder.prompt is not None else 0
    if args.window + prompt > encoder.max_context:
        args.window = encoder.max_context - prompt
    info["peft"], allocator = apply_technique(encoder, args.method, args)
    encoder.allocator_factory = allocator
    info.update({"architecture": encoder.architecture, "mode": encoder.mode, "layers": len(encoder.blocks)})
    return encoder, tokenizer, info


def build_tagger(args, labels):
    encoder, tokenizer, info = build_encoder(args)
    tagger = RoleTagger(encoder, labels, pooling=args.pooling, context=args.context, hidden=args.hidden,
                        layers=args.context_layers, position_buckets=0 if args.no_position else args.position_buckets,
                        domain_heads=args.domain_heads, dropout=args.dropout, use_crf=args.loss == "crf",
                        two_pass=getattr(args, "two_pass", False), aux_shift=bool(getattr(args, "aux_shift", 0)),
                        aux_coarse=bool(getattr(args, "aux_coarse", 0)), tf_layers=getattr(args, "tf_layers", 2))
    info["parameters"] = count_parameters(tagger)
    info["encoder_parameters"] = count_parameters(encoder)
    return tagger, tokenizer, info


def trainable_state(tagger):
    """Only what training changed: adapter / unfrozen encoder weights and everything above the encoder.
    The frozen Phase-1 (or merged Phase-2) weights are rebuilt from the checkpoint at load time."""
    names = {n for n, p in tagger.named_parameters() if p.requires_grad or not n.startswith("encoder.")}
    return {k: v.detach().cpu() for k, v in tagger.state_dict().items()
            if k in names or not k.startswith("encoder.") or k.endswith(".mask")}   # .mask: AdaLoRA ranks


def save_tagger(path, tagger, args, labels, extra=None):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "labels": labels, "state": trainable_state(tagger), **(extra or {})}, path)


def load_tagger(path, device=None):
    import argparse
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    tagger, tokenizer, _ = build_tagger(args, payload["labels"])
    missing, unexpected = tagger.load_state_dict(payload["state"], strict=False)
    trainable = {n for n, p in tagger.named_parameters() if p.requires_grad}
    if unexpected or [m for m in missing if m in trainable or not m.startswith("encoder.")]:
        raise ValueError(f"{path}: state does not fit the rebuilt model: missing={missing[:5]} unexpected={unexpected[:5]}")
    if device is not None:
        tagger.to(device)
    return tagger.eval(), tokenizer, payload
