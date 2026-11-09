"""Fine-tuning techniques added for Phase 3, from scratch (no peft / bitsandbytes), on top of phase2.peft
(lora, lorma, adapter, prompt, normfit, partial, full):

  rslora   LoRA with the rank-stabilised scale alpha/sqrt(r) instead of alpha/r     (Kalajdzievski, 2023)
  qlora    frozen base weights stored in 4-bit NormalFloat (NF4, blocks of 64, double-quantised absmax),
           dequantised on the fly, LoRA adapters in bf16/fp32 on top               (Dettmers et al., 2023)
  adalora  SVD-form adapters P diag(E) Q with an importance-driven rank budget (cubic schedule from
           --adalora-init-r to --adalora-target-r) and an orthogonality penalty   (Zhang et al., 2023)
  prefix   learned key/value prefixes in every attention layer, reparameterised by an MLP (Li & Liang, 2021)
  ia3      learned vectors rescaling attention keys, values and the feed-forward activations (Liu et al., 2022)
  prompt   soft prompt tuning (phase2: learned input tokens)                      (Lester et al., 2021)
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from phase2.model import Encoder
from phase2.peft import DEFAULT_TARGETS, LoRALinear, _replace_linears, apply_method

NEW_METHODS = ("rslora", "qlora", "adalora", "prefix", "ia3")


# --------------------------------------------------------------------------- NF4 (QLoRA)
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


class NF4Linear(nn.Module):
    """A frozen Linear kept as 4-bit NF4 codes (two per byte) + double-quantised block absmax."""
    def __init__(self, linear, block=64, absmax_block=256):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight.detach().float().flatten()
        n = w.numel()
        pad = (-n) % block
        w = torch.cat((w, w.new_zeros(pad))).view(-1, block)
        absmax = w.abs().amax(1).clamp_min(1e-12)
        normed = w / absmax[:, None]
        mids = (NF4[1:] + NF4[:-1]) / 2
        codes = torch.bucketize(normed, mids).to(torch.uint8).flatten()
        if codes.numel() % 2:
            codes = torch.cat((codes, codes.new_zeros(1)))
        self.register_buffer("packed", (codes[0::2] << 4) | codes[1::2])
        # double quantisation of the fp32 absmax: 8-bit per block of 256, around the mean
        mean = absmax.mean()
        a = absmax - mean
        apad = (-a.numel()) % absmax_block
        a = torch.cat((a, a.new_zeros(apad))).view(-1, absmax_block)
        scale = a.abs().amax(1).clamp_min(1e-12) / 127
        self.register_buffer("absmax_q", torch.round(a / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("absmax_scale", scale)
        self.register_buffer("absmax_mean", mean.reshape(1))
        self.register_buffer("codebook", NF4.clone())
        self.n, self.block, self.n_blocks = n, block, absmax.numel()
        self.bias = linear.bias

    def weight_dequant(self, dtype):
        codes = torch.stack((self.packed >> 4, self.packed & 15), 1).flatten().long()
        absmax = (self.absmax_q.float() * self.absmax_scale[:, None]).flatten()[:self.n_blocks] + self.absmax_mean
        w = self.codebook[codes[:self.n_blocks * self.block]].view(self.n_blocks, self.block) * absmax[:, None]
        return w.flatten()[:self.n].view(self.out_features, self.in_features).to(dtype)

    @property
    def weight(self):                      # read-only view, used for shape/dtype queries and merges
        return self.weight_dequant(self.absmax_scale.dtype)

    def forward(self, x):
        # under autocast F.linear runs in bf16/fp16 whatever the operands' dtype
        return F.linear(x, self.weight_dequant(x.dtype), self.bias)


# --------------------------------------------------------------------------- AdaLoRA
class AdaLoRALinear(nn.Module):
    """W x + scale * B diag(E * mask) A x   (P = B, Lambda = E, Q = A in the paper)."""
    def __init__(self, base, rank=12, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.E = nn.Parameter(torch.zeros(rank))
        self.B = nn.Parameter(torch.randn(base.out_features, rank) * 0.02)
        self.register_buffer("mask", torch.ones(rank))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        delta = (self.drop(x) @ self.A.t()) * (self.E * self.mask) @ self.B.t()
        return self.base(x) + self.scale * delta

    def orth_penalty(self):
        r = self.A.size(0)
        eye = torch.eye(r, device=self.A.device)
        return ((self.A @ self.A.t() - eye) ** 2).sum() + ((self.B.t() @ self.B - eye) ** 2).sum()


class RankAllocator:
    """AdaLoRA budget: sensitivity |w * grad| smoothed (beta1) times its uncertainty (beta2); every
    delta_t steps the lowest-importance singular triplets are masked until the budget, which falls
    cubically from init_r to target_r per module between t_init and total - t_final."""
    def __init__(self, modules, init_r, target_r, total_steps, t_init=0.1, t_final=0.2, delta_t=10,
                 beta1=0.85, beta2=0.85):
        self.modules = list(modules)
        self.init_budget = init_r * len(self.modules)
        self.target_budget = target_r * len(self.modules)
        self.total = total_steps
        self.t_init, self.t_final = int(t_init * total_steps), int(t_final * total_steps)
        self.delta_t, self.beta1, self.beta2 = delta_t, beta1, beta2
        self.ipt, self.unc = {}, {}

    def budget(self, step):
        if step < self.t_init:
            return self.init_budget
        if step >= self.total - self.t_final:
            return self.target_budget
        frac = 1 - (step - self.t_init) / max(1, self.total - self.t_final - self.t_init)
        return int(self.target_budget + (self.init_budget - self.target_budget) * frac ** 3)

    @torch.no_grad()
    def _update_importance(self):
        for i, m in enumerate(self.modules):
            for name in ("A", "E", "B"):
                p = getattr(m, name)
                if p.grad is None:
                    continue
                s = (p * p.grad).abs()
                key = (i, name)
                if key not in self.ipt:
                    self.ipt[key] = torch.zeros_like(s)
                    self.unc[key] = torch.zeros_like(s)
                self.ipt[key].mul_(self.beta1).add_((1 - self.beta1) * s)
                self.unc[key].mul_(self.beta2).add_((1 - self.beta2) * (s - self.ipt[key]).abs())

    def _scores(self, i):
        get = lambda n: self.ipt[(i, n)] * self.unc[(i, n)]
        return get("E") + get("A").mean(1) + get("B").mean(0)

    @torch.no_grad()
    def step(self, step):
        self._update_importance()
        if not self.ipt or step < self.t_init:
            return
        frozen = step >= self.total - self.t_final
        if step % self.delta_t and not frozen:
            return
        if frozen and getattr(self, "_frozen", False):
            for m in self.modules:              # final phase: keep the last mask
                m.E.mul_(m.mask)
            return
        scores = [self._scores(i) for i in range(len(self.modules)) if (i, "E") in self.ipt]
        allv = torch.cat(scores)
        keep = min(self.budget(step), allv.numel())
        threshold = torch.kthvalue(allv.float(), max(1, allv.numel() - keep)).values if keep < allv.numel() else -1
        for m, s in zip(self.modules, scores):
            m.mask.copy_((s > threshold).float())
            m.E.mul_(m.mask)
        self._frozen = frozen

    def ranks(self):
        return [int(m.mask.sum()) for m in self.modules]

    def orth_penalty(self):
        return sum(m.orth_penalty() for m in self.modules) / max(1, len(self.modules))


# --------------------------------------------------------------------------- IA3
class IA3Out(nn.Module):
    """Rescale a projection's output (keys, values)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        out = self.base(x)
        return out * self.l.to(out.dtype)       # keep bf16 under autocast: attention kernels need q, k, v alike


class IA3In(nn.Module):
    """Rescale the feed-forward activation entering the down projection."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l.to(x.dtype))


# --------------------------------------------------------------------------- prefix tuning
class Prefix(nn.Module):
    """Per-layer key/value prefixes from an MLP over learned embeddings (the reparameterisation Li & Liang
    use for stable training); only attention layers get one (hybrid LSTM layers have no keys)."""
    def __init__(self, blocks, dim, length=16, hidden=512):
        super().__init__()
        self.length = length
        self.layers = []
        sizes = []
        for i, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is not None:
                self.layers.append((i, attn.kv_heads, attn.head_dim))
                sizes.append(2 * attn.kv_heads * attn.head_dim)
        self.sizes = sizes
        self.emb = nn.Parameter(torch.randn(length, dim) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, sum(sizes)))
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, batch):
        flat = self.mlp(self.emb)          # bf16 under autocast, like the keys it is concatenated with
        out = {}
        for (i, kv, hd), part in zip(self.layers, flat.split(self.sizes, -1)):
            k, v = part.view(self.length, 2, kv, hd).unbind(1)
            out[i] = (k.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1),
                      v.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1))
        return out


class PrefixEncoder(Encoder):
    """phase2.model.Encoder whose attention layers also attend to learned prefixes (passed as `past`)."""
    def forward(self, ids):
        x = self.embedding(ids)
        prompt = 0
        if self.prompt is not None:
            prompt = self.prompt.size(0)
            x = torch.cat((self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype), x), 1)
        if self.position is not None:
            x = x + self.position(torch.arange(x.size(1), device=x.device))
        causal = self.mode == "causal"
        prefixes = self.prefix(x.size(0))
        for i, block in enumerate(self.blocks):
            past = {"self": prefixes[i]} if i in prefixes else None
            if self.gradient_checkpointing and self.training:
                x = checkpoint(lambda y, b=block, p=past: b(y, past=p, causal=causal)[0], x, use_reentrant=False)
            else:
                x, _ = block(x, past=past, causal=causal)
        x = self.final_norm(x)
        return x[:, prompt:]


# --------------------------------------------------------------------------- dispatch
IA3_KV = r"attn\.(k|v)$"
IA3_FF = r"ff\.down$"


def apply_technique(encoder, method, args):
    """Like phase2.peft.apply_method, for every method (old and new). Returns (info, controller-factory)."""
    if method not in NEW_METHODS:
        return apply_method(encoder, method, rank=args.rank, alpha=args.alpha, top_k=args.top_k,
                            bottleneck=args.bottleneck), None
    for p in encoder.parameters():
        p.requires_grad = False
    info = {"method": method}
    if method == "rslora":
        def make(m):
            layer = LoRALinear(m, args.rank, args.alpha, 0.05)
            layer.scale = args.alpha / math.sqrt(args.rank)
            return layer
        info["replaced"] = _replace_linears(encoder.blocks, make, DEFAULT_TARGETS)
    elif method == "qlora":
        info["replaced"] = _replace_linears(encoder.blocks, lambda m: LoRALinear(NF4Linear(m), args.rank, args.alpha, 0.05),
                                            DEFAULT_TARGETS)
        nf4 = [m for m in encoder.modules() if isinstance(m, NF4Linear)]
        info["base_bytes_4bit"] = sum(m.packed.numel() + m.absmax_q.numel() + 4 * m.absmax_scale.numel() for m in nf4)
        info["base_bytes_bf16"] = sum(2 * m.n for m in nf4)
    elif method == "adalora":
        info["replaced"] = _replace_linears(
            encoder.blocks, lambda m: AdaLoRALinear(m, args.adalora_init_r, args.alpha, 0.05), DEFAULT_TARGETS)
        info["init_r"], info["target_r"] = args.adalora_init_r, args.adalora_target_r
    elif method == "ia3":
        info["replaced"] = _replace_linears(encoder.blocks, IA3Out, IA3_KV) + \
            _replace_linears(encoder.blocks, IA3In, IA3_FF)
    elif method == "prefix":
        encoder.prefix = Prefix(encoder.blocks, encoder.dim, args.prefix_length, args.prefix_hidden)
        encoder.__class__ = PrefixEncoder
        info["prefix_length"] = args.prefix_length
        info["replaced"] = len(encoder.prefix.layers)
    if not info.get("replaced"):
        raise ValueError(f"{method}: no layers matched")
    for module in encoder.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = module.B.requires_grad = True
        elif isinstance(module, AdaLoRALinear):
            for name in ("A", "E", "B"):
                getattr(module, name).requires_grad = True
        elif isinstance(module, (IA3Out, IA3In)):
            module.l.requires_grad = True
        elif isinstance(module, Prefix):
            for p in module.parameters():
                p.requires_grad = True
    return info, (make_allocator if method == "adalora" else None)


def make_allocator(encoder, args, total_steps):
    return RankAllocator([m for m in encoder.modules() if isinstance(m, AdaLoRALinear)], args.adalora_init_r,
                         args.adalora_target_r, total_steps)
