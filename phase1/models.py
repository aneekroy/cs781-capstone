"""Four genuinely different randomly initialized language-model designs.

All classes expose token representations through `hidden` and `encode`, so
later phases can add task heads. Generation uses attention KV/recurrent caches.
The local attention implementation computes query chunks against a bounded
key window: it does not construct a quadratic full-length local mask.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    architecture: str = "dense_gpt"
    vocab_size: int = 24000
    dim: int = 768
    layers: int = 12
    heads: int = 12
    kv_heads: int = 2
    ff_dim: int = 2048
    max_context: int = 2048
    encoder_layers: int = 6
    decoder_layers: int = 6
    local_window: int = 256
    query_chunk: int = 64
    global_every: int = 4
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def validate(self):
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("dim must be divisible by heads; heads must be divisible by kv_heads")
        if (self.dim // self.heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if min(self.layers, self.encoder_layers, self.decoder_layers, self.local_window,
               self.query_chunk, self.global_every, self.max_context) <= 0:
            raise ValueError("All layer/context/window counts must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def rotary(x, offset=0):
    dim = x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = torch.outer(torch.arange(offset, offset + x.size(-2), device=x.device).float(), inv)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config, kv_heads=None, rope=False, local=False):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = kv_heads or config.heads
        self.head_dim = config.dim // config.heads
        self.rope, self.local = rope, local
        self.window, self.chunk = config.local_window, config.query_chunk
        self.dropout = config.dropout
        self.q = nn.Linear(config.dim, config.dim, bias=False)
        self.k = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x, context=None, past=None, causal=True, use_cache=False):
        batch, length, dim = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        cross = context is not None
        offset = past[0].size(-2) if past is not None and not cross else 0
        if cross and past is not None:
            k, v = past
        else:
            source = x if context is None else context
            k = self.k(source).view(batch, -1, self.kv_heads, self.head_dim).transpose(1, 2)
            v = self.v(source).view(batch, -1, self.kv_heads, self.head_dim).transpose(1, 2)
            if self.rope and not cross:
                k = rotary(k, offset)
            if past is not None and not cross:
                k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
        if self.rope and not cross:
            q = rotary(q, offset)
        kwargs = dict(dropout_p=self.dropout if self.training else 0.0,
                      enable_gqa=self.kv_heads != self.heads)
        if self.local and causal:
            parts = []
            for start in range(0, length, self.chunk):
                end = min(start + self.chunk, length)
                left, right = max(0, offset + start - self.window + 1), offset + end
                qi = torch.arange(offset + start, offset + end, device=x.device)[:, None]
                ki = torch.arange(left, right, device=x.device)[None, :]
                allowed = (ki <= qi) & (ki > qi - self.window)
                parts.append(F.scaled_dot_product_attention(
                    q[:, :, start:end], k[:, :, left:right], v[:, :, left:right],
                    attn_mask=allowed, **kwargs))
            y = torch.cat(parts, -2)
        elif causal and past is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs)
        elif causal and length > 1:
            qi = torch.arange(offset, offset + length, device=x.device)[:, None]
            ki = torch.arange(k.size(-2), device=x.device)[None, :]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=ki <= qi, **kwargs)
        else:
            y = F.scaled_dot_product_attention(q, k, v, **kwargs)
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.o(y), (k, v) if use_cache else None


class FeedForward(nn.Module):
    def __init__(self, config, gated=True):
        super().__init__()
        self.gated = gated
        middle = config.ff_dim if gated else 4 * config.dim
        self.up = nn.Linear(config.dim, middle, bias=False)
        self.gate = nn.Linear(config.dim, middle, bias=False) if gated else None
        self.down = nn.Linear(middle, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        y = F.silu(self.gate(x)) * self.up(x) if self.gated else F.gelu(self.up(x))
        return self.drop(self.down(y))


class Block(nn.Module):
    def __init__(self, config, kind="dense", local=False, cross=False):
        super().__init__()
        dense = kind == "dense"
        norm = nn.LayerNorm if dense else RMSNorm
        self.norm1, self.norm2 = norm(config.dim), norm(config.dim)
        self.attn = Attention(config, config.kv_heads if kind == "gqa" else config.heads,
                              rope=kind in {"gqa", "hybrid"}, local=local)
        self.ff = FeedForward(config, gated=not dense)
        self.cross_attn = Attention(config) if cross else None
        self.cross_norm = norm(config.dim) if cross else None
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, memory=None, past=None, causal=True, use_cache=False):
        past = past or {}
        y, self_cache = self.attn(self.norm1(x), past=past.get("self"),
                                  causal=causal, use_cache=use_cache)
        x = x + self.drop(y)
        cross_cache = None
        if self.cross_attn is not None:
            y, cross_cache = self.cross_attn(self.cross_norm(x), context=memory,
                                           past=past.get("cross"), causal=False,
                                           use_cache=use_cache)
            x = x + self.drop(y)
        x = x + self.ff(self.norm2(x))
        return x, {"self": self_cache, "cross": cross_cache} if use_cache else None


class RecurrentBlock(nn.Module):
    """LSTM state propagation replaces attention in alternating hybrid layers."""
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.dim), RMSNorm(config.dim)
        self.rnn = nn.LSTM(config.dim, config.dim, batch_first=True)
        self.ff = FeedForward(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, memory=None, past=None, causal=True, use_cache=False):
        y, state = self.rnn(self.norm1(x), past.get("state") if past else None)
        x = x + self.drop(y)
        x = x + self.ff(self.norm2(x))
        return x, {"state": state} if use_cache else None


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class DecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_context, config.dim) if config.architecture == "dense_gpt" else None
        blocks = []
        for layer in range(config.layers):
            if config.architecture == "recurrent_hybrid" and layer % 2 == 0:
                blocks.append(RecurrentBlock(config))
            else:
                kind = {"dense_gpt": "dense", "gqa_local": "gqa", "recurrent_hybrid": "hybrid"}[config.architecture]
                local = kind == "gqa" and (layer + 1) % config.global_every != 0
                blocks.append(Block(config, kind=kind, local=local))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.dim) if config.architecture == "dense_gpt" else RMSNorm(config.dim)
        self.apply(initialize)
        if config.architecture == "dense_gpt":
            residual_std = 0.02 / math.sqrt(2 * config.layers)
            for block in self.blocks:
                nn.init.normal_(block.attn.o.weight, std=residual_std)
                nn.init.normal_(block.ff.down.weight, std=residual_std)

    def hidden(self, input_ids, cache=None, use_cache=False):
        offset = cache["length"] if cache else 0
        if input_ids.size(1) + offset > self.config.max_context:
            raise ValueError("Input plus cache exceeds max_context; no silent prompt truncation")
        x = self.embedding(input_ids)
        if self.position is not None:
            x = x + self.position(torch.arange(offset, offset + input_ids.size(1), device=x.device))
        new = []
        for i, block in enumerate(self.blocks):
            past = cache["layers"][i] if cache else None
            if self.training and self.config.gradient_checkpointing and not use_cache:
                x = checkpoint(lambda y, b=block: b(y)[0], x, use_reentrant=False)
                state = None
            else:
                x, state = block(x, past=past, use_cache=use_cache)
            new.append(state)
        state = {"length": offset + input_ids.size(1), "layers": new} if use_cache else None
        return self.norm(x), state

    def forward(self, input_ids, source_ids=None, memory=None, cache=None, use_cache=False):
        x, cache = self.hidden(input_ids, cache, use_cache)
        return F.linear(x, self.embedding.weight), cache


class EncoderDecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_context, config.dim)
        self.encoder = nn.ModuleList(Block(config, kind="seq2seq") for _ in range(config.encoder_layers))
        self.decoder = nn.ModuleList(Block(config, kind="seq2seq", cross=True) for _ in range(config.decoder_layers))
        self.encoder_norm, self.decoder_norm = RMSNorm(config.dim), RMSNorm(config.dim)
        self.apply(initialize)

    def encode(self, source_ids):
        if source_ids.size(1) > self.config.max_context:
            raise ValueError("Encoder input exceeds max_context")
        x = self.embedding(source_ids) + self.position(torch.arange(source_ids.size(1), device=source_ids.device))
        for block in self.encoder:
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(lambda y, b=block: b(y, causal=False)[0], x, use_reentrant=False)
            else:
                x, _ = block(x, causal=False)
        return self.encoder_norm(x)

    def hidden(self, input_ids, memory, cache=None, use_cache=False):
        offset = cache["length"] if cache else 0
        if offset + input_ids.size(1) > self.config.max_context:
            raise ValueError("Decoder input plus cache exceeds max_context")
        x = self.embedding(input_ids) + self.position(torch.arange(offset, offset + input_ids.size(1), device=input_ids.device))
        new = []
        for i, block in enumerate(self.decoder):
            if self.training and self.config.gradient_checkpointing and not use_cache:
                x = checkpoint(lambda y, m, b=block: b(y, memory=m)[0], x, memory, use_reentrant=False)
                state = None
            else:
                x, state = block(x, memory=memory, past=cache["layers"][i] if cache else None, use_cache=use_cache)
            new.append(state)
        state = {"length": offset + input_ids.size(1), "layers": new} if use_cache else None
        return self.decoder_norm(x), state

    def forward(self, input_ids, source_ids=None, memory=None, cache=None, use_cache=False):
        if memory is None:
            if source_ids is None:
                raise ValueError("Encoder-decoder requires source_ids or encoder memory")
            memory = self.encode(source_ids)
        x, cache = self.hidden(input_ids, memory, cache, use_cache)
        return F.linear(x, self.embedding.weight), cache


def build_model(config):
    if isinstance(config, dict):
        config = ModelConfig(**config)
    config.validate()
    if config.architecture == "encoder_decoder":
        return EncoderDecoderLM(config)
    if config.architecture in {"dense_gpt", "gqa_local", "recurrent_hybrid"}:
        return DecoderLM(config)
    raise ValueError(f"Unknown architecture: {config.architecture}")


def parameter_count(config):
    # Instantiate on meta: count real tied parameters without allocating 100M weights.
    with torch.device("meta"):
        model = build_model(config)
    return sum(p.numel() for p in model.parameters())
