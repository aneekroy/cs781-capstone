"""Parameter-efficient adaptation methods, implemented from scratch (no peft library).

  lora     W x + (a/r) B A x                       (Hu et al., 2022)
  lorma    (I + (a/r) U V) W0 x  == h + (a/r) U (V h), h = W0 x
           LoRMA with additive rank inflation (LoRMA+, Bihany, Patel & Modi,
           Findings of ACL 2025). U is zero-initialized, so training starts
           exactly at the Phase-1 weights. Cost is O(r * d_out) per token
           because the d_out x d_out product is never materialized.
  adapter  Houlsby-style bottleneck adapter after every block (zero-init up-proj)
  prompt   soft prompt tuning (learned virtual tokens; see model.Encoder)
  normfit  BitFit-style: only norm gains/biases (+ head). The Phase-1 linears are
           bias-free, so classic BitFit has nothing to train there.
  partial  unfreeze the top-k blocks + final norm
  full     everything trainable
"""
import math
import re
import torch
from torch import nn

DEFAULT_TARGETS = r"(attn|cross_attn)\.(q|k|v|o)$|ff\.(up|gate|down)$"


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.t() @ self.B.t())


class LoRMALinear(nn.Module):
    """LoRMA+ : W = (I + (alpha/r) U V) W0."""
    def __init__(self, base, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        out = base.out_features
        self.V = nn.Parameter(torch.empty(rank, out))
        self.U = nn.Parameter(torch.zeros(out, rank))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.base(x)
        return h + self.scale * (self.drop(h) @ self.V.t() @ self.U.t())


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterBlock(nn.Module):
    def __init__(self, block, dim, bottleneck=64):
        super().__init__()
        self.block = block
        self.adapter = Adapter(dim, bottleneck)

    def forward(self, *args, **kwargs):
        x, state = self.block(*args, **kwargs)
        return self.adapter(x), state


def _replace_linears(module, factory, pattern):
    regex = re.compile(pattern)
    replaced = 0
    for name, child in list(module.named_modules()):
        for child_name, leaf in list(child.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(leaf, nn.Linear) and regex.search(full):
                setattr(child, child_name, factory(leaf))
                replaced += 1
    return replaced


def apply_method(encoder, method, rank=16, alpha=32, dropout=0.05, bottleneck=64, top_k=4,
                 targets=DEFAULT_TARGETS):
    """Mutates encoder (model.Encoder) in place; returns a description dict."""
    blocks = encoder.blocks
    for parameter in encoder.parameters():
        parameter.requires_grad = method == "full"
    info = {"method": method}
    if method == "lora":
        info["replaced"] = _replace_linears(blocks, lambda m: LoRALinear(m, rank, alpha, dropout), targets)
    elif method == "lorma":
        info["replaced"] = _replace_linears(blocks, lambda m: LoRMALinear(m, rank, alpha, 0.0), targets)
    elif method == "adapter":
        for i in range(len(blocks)):
            blocks[i] = AdapterBlock(blocks[i], encoder.dim, bottleneck)
    elif method == "normfit":
        for name, parameter in encoder.named_parameters():
            if "norm" in name or name.endswith(".bias"):
                parameter.requires_grad = True
    elif method == "partial":
        for block in list(blocks)[-top_k:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in encoder.final_norm.parameters():
            parameter.requires_grad = True
    elif method in {"full", "frozen", "prompt"}:
        pass
    else:
        raise ValueError(f"Unknown method {method}")
    if method in {"lora", "lorma"} and not info["replaced"]:
        raise ValueError("No linear layers matched the adaptation targets")
    # New adapter parameters must be trainable (created after the freeze loop).
    for module in encoder.modules():
        if isinstance(module, (LoRALinear, LoRMALinear)):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = True
        if isinstance(module, Adapter):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if encoder.prompt is not None:
        encoder.prompt.requires_grad = True
    return info


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / max(1, total), 3)}
