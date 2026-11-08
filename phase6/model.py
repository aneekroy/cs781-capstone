"""Phase-1 language models as summarizers (all four architectures), with every Phase-6 adaptation technique.

  decoder-only (dense_gpt, gqa_local, recurrent_hybrid):
      [soft prompt] source chunk  "\\n\\nSummary:"  target summary <eos>        loss on the target tokens only
  encoder-decoder:
      encoder([soft prompt] source chunk)  ->  decoder(<bos> target)  -> target <eos>
      FiD (--strategy fid): every chunk of the document is encoded separately (bounded encoder context), the
      encoder states are concatenated and ONE decoder pass cross-attends to the whole document and writes the whole
      summary (Izacard & Grave 2021 "fusion in decoder", a hierarchical encoder-decoder representation).

Padding. The Phase-1 code has no attention masks (Phase 1 batched equal-length prompts only). Here a key-padding
mask is applied by patching phase1.models.Attention.forward (and RecurrentBlock.forward for left-padded LSTM
prefill), the same way phase3/flash.py patches it: the encoder ignores padded source positions, cross-attention
ignores padded memory, and batched decoder-only generation uses left padding. Without a mask the patched code calls
the previous forward unchanged (FlashAttention / SDPA from phase3.flash when enabled first).

Techniques (--method): lorma (LoRMA, compulsory), lora, rslora, qlora (NF4 base + LoRA), adalora, houlsby (two
bottleneck adapters per layer: after the attention output projection and after the feed-forward down projection),
pfeiffer (one adapter per layer after the feed-forward sublayer; phase2 "adapter"), prefix (key/value prefixes in
every decoder self-attention layer), prompt (soft prompt tokens before the input), ia3, partial (top-k blocks),
full. Knowledge distillation is in train.py.
"""
import argparse
import contextlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from phase2.model import load_phase1
from phase2.peft import Adapter, LoRALinear, LoRMALinear, _replace_linears, apply_method, count_parameters
from phase3.peft import AdaLoRALinear, Prefix, apply_technique, make_allocator

SEP = "\n\nSummary:"

# --------------------------------------------------------------------------- padding masks
_MASK = {"src": None, "self": None, "shift": None, "fixed": 0}
_PREV = {}


def norm_mask(m):
    """None when there is nothing to mask (all positions real) so the fast unmasked kernels are used."""
    if m is None:
        return None
    m = m.bool()
    return None if bool(m.all()) else m


@contextlib.contextmanager
def masks(src=None, self_keys=None, shift=None, fixed=0):
    """Key masks (True = real) for the attention calls inside: src for encoder self-attention and
    cross-attention, self_keys for causal self-attention over [prefix] + inputs (+ cache).
    shift [B]: left padding per row; the first `fixed` keys (prefix-tuning keys) are not shifted. Rotary positions
    and the sliding window then use each row's own positions, as if the row had been decoded alone."""
    old = dict(_MASK)
    _MASK["src"], _MASK["self"], _MASK["shift"], _MASK["fixed"] = src, self_keys, shift, fixed
    try:
        yield
    finally:
        _MASK.update(old)


def rotary_rows(x, pos):
    """phase1.models.rotary with a position per row and token: x [B, H, L, D], pos [B, L]."""
    dim = x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = pos.float()[..., None] * inv
    cos, sin = angles.cos().to(x.dtype)[:, None], angles.sin().to(x.dtype)[:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def _attention(self, x, context=None, past=None, causal=True, use_cache=False):
    cross = context is not None
    m = _MASK["src"] if (cross or not causal) else _MASK["self"]
    if m is None:
        return _PREV["attn"](self, x, context=context, past=past, causal=causal, use_cache=use_cache)
    from phase1.models import rotary
    b, length, dim = x.shape
    shift = _MASK["shift"] if (causal and not cross) else None
    offset0 = past[0].size(-2) if (past is not None and not cross) else 0
    if shift is not None:
        qlog = torch.arange(offset0, offset0 + length, device=x.device)[None, :] - shift[:, None]   # [B, L]
    if shift is not None and self.rope:
        rot = lambda t: rotary_rows(t, qlog)
    else:
        rot = lambda t: rotary(t, offset0)
    q = self.q(x).view(b, length, self.heads, self.head_dim).transpose(1, 2)
    offset = offset0
    if cross and past is not None:
        k, v = past
    else:
        source = context if cross else x
        k = self.k(source).view(b, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(source).view(b, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.rope and not cross:
            k = rot(k)
        if past is not None and not cross:
            k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
    if self.rope and not cross:
        q = rot(q)
    cache = (k, v) if use_cache else None
    keys = k.size(-2)
    if m.size(1) != keys:
        raise RuntimeError(f"padding mask covers {m.size(1)} keys, attention has {keys}")
    if k.dtype != q.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)
    groups = self.heads // self.kv_heads
    if groups > 1:
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    allowed = m[:, None, None, :]
    if not cross:
        qi = torch.arange(offset, offset + length, device=x.device)[:, None]
        ki = torch.arange(keys, device=x.device)[None, :]
        if causal:
            rule = ki <= qi
            if self.local:
                if shift is not None:            # the window over each row's own positions
                    kpos = ki - torch.where(ki >= _MASK["fixed"], shift[:, None], torch.zeros_like(shift)[:, None])
                    rule = rule & (kpos[:, None, :] > qlog[:, :, None] - self.window)       # [B, L, K]
                    rule = rule[:, None]
                else:
                    rule = rule & (ki > qi - self.window)
            allowed = allowed & rule
        allowed = allowed | (ki == qi)           # a padded query still sees itself: no empty softmax rows
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                       dropout_p=self.dropout if self.training else 0.0)
    return self.o(y.transpose(1, 2).contiguous().view(b, length, dim)), cache


def _recurrent(self, x, memory=None, past=None, causal=True, use_cache=False):
    """LSTM layer of the hybrid with left padding: each row runs over its real positions only."""
    m = _MASK["self"]
    if m is None or x.size(1) == 1:
        return _PREV["rnn"](self, x, memory=memory, past=past, causal=causal, use_cache=use_cache)
    keep = m[:, -x.size(1):]
    h = self.norm1(x)
    y = torch.zeros_like(h)
    state0 = past.get("state") if past else None
    hs, cs = [], []
    for r in range(x.size(0)):
        idx = keep[r].nonzero().squeeze(1)
        st = None if state0 is None else (state0[0][:, r:r + 1].contiguous(), state0[1][:, r:r + 1].contiguous())
        out, (hn, cn) = self.rnn(h[r:r + 1, idx], st)
        y[r, idx] = out[0].to(y.dtype)
        hs.append(hn)
        cs.append(cn)
    x = x + self.drop(y)
    x = x + self.ff(self.norm2(x))
    return x, ({"state": (torch.cat(hs, 1), torch.cat(cs, 1))} if use_cache else None)


_attention._phase6 = True


def install_masks():
    """Patch phase1.models (idempotent). Call AFTER phase3.flash.enable so unmasked calls still use it."""
    from phase1 import models
    if getattr(models.Attention.forward, "_phase6", False):
        return
    _PREV["attn"], _PREV["rnn"] = models.Attention.forward, models.RecurrentBlock.forward
    models.Attention.forward = _attention
    models.RecurrentBlock.forward = _recurrent


def _run(block, x, past=None, self_mask=None, memory=None, src_mask=None, causal=True):
    """One block under masks (also what gradient checkpointing recomputes, so the masks are re-applied)."""
    with masks(src=src_mask, self_keys=self_mask):
        return block(x, memory=memory, past=past, causal=causal)[0]


# --------------------------------------------------------------------------- PEFT on the whole language model
HOULSBY = r"(attn|cross_attn)\.o$|ff\.down$"


class HoulsbyLinear(nn.Module):
    """A projection followed by a zero-initialised bottleneck adapter (output = W x + up(gelu(down(W x))))."""
    def __init__(self, base, bottleneck=64):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.adapter = Adapter(base.out_features, bottleneck)

    def forward(self, x):
        return self.adapter(self.base(x))


class _Stack(nn.Module):
    """One block list seen as the `encoder` object that phase2/phase3 PEFT functions expect."""
    def __init__(self, blocks, dim, norm):
        super().__init__()
        self.blocks, self.final_norm, self.dim = blocks, norm, dim
        self.prompt = None


def apply_peft(lm, args):
    seq2seq = lm.config.architecture == "encoder_decoder"
    for p in lm.parameters():
        p.requires_grad = False
    dim = lm.config.dim
    stacks = ([_Stack(lm.encoder, dim, lm.encoder_norm), _Stack(lm.decoder, dim, lm.decoder_norm)] if seq2seq
              else [_Stack(lm.blocks, dim, lm.norm)])
    m = args.method
    info = {"method": m}
    if m == "full":
        for p in lm.parameters():
            p.requires_grad = True
    elif m == "partial":                                        # top-k blocks of the (decoder) stack + its norm
        for block in list(stacks[-1].blocks)[-args.top_k:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in stacks[-1].final_norm.parameters():
            p.requires_grad = True
    elif m == "houlsby":
        info["replaced"] = sum(_replace_linears(s.blocks, lambda l: HoulsbyLinear(l, args.bottleneck), HOULSBY)
                               for s in stacks)
        for mod in lm.modules():
            if isinstance(mod, HoulsbyLinear):
                for p in mod.adapter.parameters():
                    p.requires_grad = True
    elif m in ("pfeiffer", "adapter"):
        for s in stacks:
            apply_method(s, "adapter", bottleneck=args.bottleneck)
        info["adapters"] = sum(len(s.blocks) for s in stacks)
    elif m in ("prefix", "prompt", "frozen"):
        pass                                                    # the Summarizer adds the trainable part
    else:                                                       # lora lorma rslora qlora adalora ia3 normfit
        sub = [apply_technique(s, m, args)[0] for s in stacks]
        info.update({k: sum(x.get(k, 0) for x in sub) for k in ("replaced",)})
        for k in ("init_r", "target_r", "base_bytes_4bit", "base_bytes_bf16"):
            if any(k in x for x in sub):
                info[k] = sum(x.get(k, 0) for x in sub) if k.startswith("base") else sub[0][k]
    for mod in lm.modules():                                    # adapters created above must stay trainable
        if isinstance(mod, (LoRALinear, LoRMALinear)) and m not in ("full",):
            for name, p in mod.named_parameters(recurse=False):
                p.requires_grad = True
    return info


# --------------------------------------------------------------------------- the summarizer
class Summarizer(nn.Module):
    def __init__(self, lm, tokenizer, args):
        super().__init__()
        self.lm = lm
        cfg = lm.config
        self.architecture = cfg.architecture
        self.seq2seq = cfg.architecture == "encoder_decoder"
        self.max_context = cfg.max_context
        self.gc = bool(getattr(args, "gradient_checkpointing", False))
        tid = tokenizer.token_to_id
        self.pad_id = tid("<pad>") if tid("<pad>") is not None else 0
        self.bos_id, self.eos_id, self.unk_id = tid("<bos>"), tid("<eos>"), tid("<unk>")
        if self.eos_id is None or (self.seq2seq and self.bos_id is None):
            raise ValueError("the Phase-1 tokenizer must define <eos> (and <bos> for the encoder-decoder)")
        self.sep = tokenizer.encode(SEP).ids
        self.prompt = None
        if args.method == "prompt":                             # initialised from real vocabulary embeddings
            g = torch.Generator().manual_seed(0)
            rows = torch.randint(5, lm.embedding.num_embeddings, (args.prompt_tokens,), generator=g)
            self.prompt = nn.Parameter(lm.embedding.weight.detach()[rows].clone())
        self.prefix = Prefix(self.dec_blocks, cfg.dim, args.prefix_length, args.prefix_hidden) \
            if args.method == "prefix" else None
        self.n_prompt = self.prompt.size(0) if self.prompt is not None else 0
        self.n_prefix = self.prefix.length if self.prefix is not None else 0

    @property
    def dec_blocks(self):
        return self.lm.decoder if self.seq2seq else self.lm.blocks

    def _past(self, rows):
        if self.prefix is None:
            return None
        pre = self.prefix(rows)
        return [{"self": pre[i]} if i in pre else None for i in range(len(self.dec_blocks))]

    def _with_prompt(self, x, mask=None):
        if self.prompt is None:
            return x, mask
        p = self.prompt.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        if mask is not None:
            mask = torch.cat((torch.ones(x.size(0), self.n_prompt, dtype=torch.bool, device=x.device), mask), 1)
        return torch.cat((p, x), 1), mask

    def lm_logits(self, h):
        return F.linear(h, self.lm.embedding.weight)

    # ---- decoder-only
    def decode_only(self, ids, cache=None, use_cache=False, self_mask=None, shift=None):
        """shift [B]: left-padding per row; learned absolute positions (dense) are counted from the first real
        token, so a padded batch reproduces single-prompt decoding exactly."""
        lm = self.lm
        x = lm.embedding(ids)
        if cache is None:
            x, _ = self._with_prompt(x)
            if shift is not None and self.prompt is not None:   # [prompt][pads][text] -> [pads][prompt][text]
                ar = torch.arange(x.size(1), device=x.device)[None, :]
                s, n = shift[:, None], self.n_prompt
                src = torch.where(ar < s, n + ar, torch.where(ar < s + n, ar - s, ar))
                x = x.gather(1, src[..., None].expand(-1, -1, x.size(2)))
            pos0, past = 0, self._past(x.size(0))
        else:
            pos0, past = cache["pos"], cache["layers"]
        length = x.size(1)
        if pos0 + length > self.max_context:
            raise ValueError(f"{pos0 + length} positions exceed the Phase-1 context {self.max_context}")
        if lm.position is not None:
            pos = torch.arange(pos0, pos0 + length, device=x.device)
            if shift is not None:                               # counted from each row's first real position
                pos = (pos[None, :] - shift[:, None]).clamp_min(0)
            x = x + lm.position(pos)
        if shift is not None and self_mask is None:
            shift = None                                        # nothing padded in this batch
        new = []
        for i, block in enumerate(lm.blocks):
            p = past[i] if past is not None else None
            if self.training and self.gc and not use_cache:
                x, st = checkpoint(_run, block, x, p, self_mask, use_reentrant=False), None
            else:
                with masks(self_keys=self_mask, shift=shift, fixed=self.n_prefix):
                    x, st = block(x, past=p, use_cache=use_cache)
            new.append(st)
        return lm.norm(x), ({"pos": pos0 + length, "layers": new} if use_cache else None)

    # ---- encoder-decoder
    def encode(self, src, src_mask):
        lm = self.lm
        x, src_mask = self._with_prompt(lm.embedding(src), src_mask)
        if x.size(1) > self.max_context:
            raise ValueError(f"encoder input {x.size(1)} exceeds the Phase-1 context {self.max_context}")
        x = x + lm.position(torch.arange(x.size(1), device=x.device))
        m = norm_mask(src_mask)
        for block in lm.encoder:
            if self.training and self.gc:
                x = checkpoint(_run, block, x, None, None, None, m, False, use_reentrant=False)
            else:
                with masks(src=m):
                    x, _ = block(x, causal=False)
        return lm.encoder_norm(x), src_mask

    def encode_fid(self, chunks, owners, rows):
        """chunks [Nc, S] (+mask) of all documents -> memory [rows, max_chunks * S', D] and its key mask."""
        ids, mask = chunks
        h, mask = self.encode(ids, mask)
        per = [owners.count(r) for r in range(rows)]
        width = max(per) * h.size(1)
        memory = h.new_zeros(rows, width, h.size(2))
        mm = torch.zeros(rows, width, dtype=torch.bool, device=h.device)
        at = 0
        for r in range(rows):
            n = per[r] * h.size(1)
            memory[r, :n] = h[at:at + per[r]].reshape(-1, h.size(2))
            mm[r, :n] = mask[at:at + per[r]].reshape(-1)
            at += per[r]
        return memory, mm

    def decode(self, ids, memory, mem_mask, cache=None, use_cache=False):
        lm = self.lm
        x = lm.embedding(ids)
        if cache is None:
            pos0, past = 0, self._past(x.size(0))
        else:
            pos0, past = cache["pos"], cache["layers"]
        length = x.size(1)
        if pos0 + length > self.max_context:
            raise ValueError(f"decoder length {pos0 + length} exceeds the Phase-1 context {self.max_context}")
        x = x + lm.position(torch.arange(pos0, pos0 + length, device=x.device))
        m = norm_mask(mem_mask)
        new = []
        for i, block in enumerate(lm.decoder):
            p = past[i] if past is not None else None
            if self.training and self.gc and not use_cache:
                x, st = checkpoint(_run, block, x, p, None, memory, m, True, use_reentrant=False), None
            else:
                with masks(src=m):
                    x, st = block(x, memory=memory, past=p, use_cache=use_cache)
            new.append(st)
        return lm.decoder_norm(x), ({"pos": pos0 + length, "layers": new} if use_cache else None)

    # ---- batches
    def _pad(self, seqs, device, left=False):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, s in enumerate(seqs):
            if left:
                ids[i, width - len(s):], mask[i, width - len(s):] = torch.tensor(s, dtype=torch.long), True
            else:
                ids[i, :len(s)], mask[i, :len(s)] = torch.tensor(s, dtype=torch.long), True
        return ids.to(device), mask.to(device)

    def fit_source(self, src, tgt_len):
        """Cut a source so that it and a target of tgt_len tokens fit the context (teachers with other limits)."""
        if self.seq2seq:
            room = self.max_context - self.n_prompt
            return [c[:room] for c in src] if src and isinstance(src[0], list) else src[:room]
        room = self.max_context - self.n_prompt - len(self.sep) - tgt_len - 1
        return src[:max(1, room)]

    def target_logits(self, examples, device, src_lm=0.0):
        """Teacher-forced logits [N, V] at every target token (incl. <eos>) of the examples, in example order, the
        labels [N], and (decoder-only, src_lm > 0) the summed next-token loss on the source tokens."""
        tgts = [e["tgt"] + [self.eos_id] for e in examples]
        aux = None
        if not self.seq2seq:
            srcs = [self.fit_source(e["src"], len(e["tgt"])) for e in examples]
            seqs = [s + self.sep + t for s, t in zip(srcs, tgts)]
            ids, _ = self._pad(seqs, device)
            h, _ = self.decode_only(ids[:, :-1])
            h = h[:, self.n_prompt:]
            sel = torch.zeros(ids.size(0), ids.size(1) - 1, dtype=torch.bool)
            src_sel = torch.zeros_like(sel)
            for i, (s, t) in enumerate(zip(srcs, tgts)):
                start = len(s) + len(self.sep)
                sel[i, start - 1:start - 1 + len(t)] = True
                src_sel[i, :len(s) - 1] = True
            sel, src_sel = sel.to(device), src_sel.to(device)
            labels = ids[:, 1:][sel]
            logits = self.lm_logits(h[sel])
            if src_lm:
                aux = F.cross_entropy(self.lm_logits(h[src_sel]).float(), ids[:, 1:][src_sel], reduction="sum")
            return logits, labels, aux
        if examples and isinstance(examples[0]["src"], list) and examples[0]["src"] and \
                isinstance(examples[0]["src"][0], list):                       # FiD
            flat, owners = [], []
            for r, e in enumerate(examples):
                for c in self.fit_source(e["src"], 0):
                    flat.append(c)
                    owners.append(r)
            memory, mm = self.encode_fid(self._pad(flat, device), owners, len(examples))
        else:
            memory, mm = self.encode(*self._pad([self.fit_source(e["src"], 0) for e in examples], device))
        dec_in, _ = self._pad([[self.bos_id] + t[:-1] for t in tgts], device)
        lab, lab_mask = self._pad(tgts, device)
        h, _ = self.decode(dec_in, memory, mm)
        return self.lm_logits(h[lab_mask]), lab[lab_mask], aux

    # ---- generation interface (generate.py)
    def room(self, prompt):
        if self.seq2seq:
            return self.max_context - 1
        return self.max_context - self.n_prompt - len(prompt) - len(self.sep)

    def start(self, prompts, device):
        """Prefill: logits [B, V] for the first generated token and the decoding state."""
        if not self.seq2seq:
            ids, pad = self._pad([p + self.sep for p in prompts], device, left=True)
            shift = (~pad).sum(1)
            width = self.n_prompt + ids.size(1)                 # inputs are [pads][prompt][text]
            inputs = torch.arange(width, device=device)[None, :] >= shift[:, None]
            key_mask = torch.cat((torch.ones(ids.size(0), self.n_prefix, dtype=torch.bool, device=device), inputs), 1)
            h, cache = self.decode_only(ids, use_cache=True, self_mask=norm_mask(key_mask), shift=shift)
            return self.lm_logits(h[:, -1]), {"cache": cache, "key_mask": key_mask, "shift": shift}
        if prompts and isinstance(prompts[0], list) and prompts[0] and isinstance(prompts[0][0], list):
            flat, owners = [], []
            for r, p in enumerate(prompts):
                for c in p:
                    flat.append(c)
                    owners.append(r)
            memory, mm = self.encode_fid(self._pad(flat, device), owners, len(prompts))
        else:
            memory, mm = self.encode(*self._pad(prompts, device))
        bos = torch.full((len(prompts), 1), self.bos_id, dtype=torch.long, device=device)
        h, cache = self.decode(bos, memory, mm, use_cache=True)
        return self.lm_logits(h[:, -1]), {"cache": cache, "memory": memory, "mem_mask": mm}

    def step(self, tokens, state):
        ids = tokens.view(-1, 1)
        if not self.seq2seq:
            key_mask = torch.cat((state["key_mask"], torch.ones_like(ids, dtype=torch.bool)), 1)
            h, cache = self.decode_only(ids, state["cache"], use_cache=True, self_mask=norm_mask(key_mask),
                                        shift=state["shift"])
            return self.lm_logits(h[:, -1]), {"cache": cache, "key_mask": key_mask, "shift": state["shift"]}
        h, cache = self.decode(ids, state["memory"], state["mem_mask"], state["cache"], use_cache=True)
        return self.lm_logits(h[:, -1]), {**state, "cache": cache}

    @staticmethod
    def reorder(state, idx):
        def go(obj, key=None):
            if obj is None or isinstance(obj, (int, float, str)):
                return obj
            if torch.is_tensor(obj):
                return obj.index_select(1 if key == "state" else 0, idx)
            if isinstance(obj, dict):
                return {k: go(v, k) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(go(v, key) for v in obj)
            return obj
        return go(state)


# --------------------------------------------------------------------------- build / save / load
def build_summarizer(args):
    lm, tokenizer, resolved = load_phase1(args.checkpoint, args.project_root)
    if getattr(args, "random_init", False):                    # ablation: same architecture, no Phase-1 weights
        from phase1.models import build_model
        lm = build_model(resolved["model"])
    k = getattr(args, "keep_layers", None)
    kept = None
    if k:
        def pick(blocks):
            # first: layers 0..k-1; spread: k evenly spaced layers incl. the first and the LAST (whose output the
            # tied LM head was trained to read) - the DistilBERT-style initialisation of a shallower student
            n = len(blocks)
            if getattr(args, "layer_pick", "first") == "spread" and k < n:
                idx = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)}) if k > 1 else [n - 1]
            else:
                idx = list(range(min(k, n)))
            return nn.ModuleList([blocks[i] for i in idx]), idx
        if lm.config.architecture == "encoder_decoder":
            lm.encoder, kept = pick(lm.encoder)
            lm.decoder, _ = pick(lm.decoder)
        else:
            lm.blocks, kept = pick(lm.blocks)
    info = {"checkpoint": str(args.checkpoint), "architecture": lm.config.architecture,
            "max_context": lm.config.max_context, "random_init": bool(getattr(args, "random_init", False)),
            "keep_layers": k, "kept_layer_indices": kept}
    info["peft"] = apply_peft(lm, args)
    model = Summarizer(lm, tokenizer, args)
    info["parameters"] = count_parameters(model)
    model.allocator_factory = make_allocator if args.method == "adalora" else None
    return model, tokenizer, info


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names or k.endswith(".mask")}


def save_model(path, model, args, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "state": trainable_state(model), **(extra or {})}, path)


def load_summarizer(run_dir, device=None):
    """(model, tokenizer, args, payload) of a finished phase6 run directory."""
    payload = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    model, tokenizer, _ = build_summarizer(args)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if unexpected or [m for m in missing if m in trainable]:
        raise ValueError(f"{run_dir}: state does not fit: missing={missing[:5]} unexpected={unexpected[:5]}")
    if device is not None:
        model.to(device)
    return model.eval(), tokenizer, args, payload


def same_tokenizer(a, b):
    return a.get_vocab() == b.get_vocab()


class Ensemble:
    """Several summarizers that share the tokenizer, decoded together: at every step the next-token distribution
    is the (weighted) mixture of the members' distributions. Exposes the Summarizer generation interface
    (start / step / reorder / room), so generate.py and summarize() work unchanged. Every member gets the same
    prompts (the first member's plan decides chunking and budgets)."""
    def __init__(self, models, weights=None):
        self.models = list(models)
        w = torch.tensor(weights or [1.0] * len(self.models), dtype=torch.float32)
        self.log_w = (w / w.sum()).log()
        first = self.models[0]
        if any(m.seq2seq != first.seq2seq for m in self.models):
            raise ValueError("ensemble members must all be decoder-only or all encoder-decoder")
        for name in ("seq2seq", "pad_id", "bos_id", "eos_id", "unk_id", "sep", "architecture"):
            setattr(self, name, getattr(first, name))

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def room(self, prompt):
        return min(m.room(prompt) for m in self.models)

    def _mix(self, logits):
        logp = torch.stack([F.log_softmax(z.float(), -1) for z in logits])            # [M, B, V]
        return torch.logsumexp(logp + self.log_w.to(logp.device)[:, None, None], 0)

    def start(self, prompts, device):
        outs = [m.start(prompts, device) for m in self.models]
        return self._mix([o[0] for o in outs]), [o[1] for o in outs]

    def step(self, tokens, state):
        outs = [m.step(tokens, s) for m, s in zip(self.models, state)]
        return self._mix([o[0] for o in outs]), [o[1] for o in outs]

    def reorder(self, state, idx):
        return [Summarizer.reorder(s, idx) for s in state]
