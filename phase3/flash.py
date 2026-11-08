"""Fast attention for the Phase-1 models (used by Phase 3 / Phase 4 training and inference).

The Phase-1 Attention calls F.scaled_dot_product_attention. Two of its paths are slow and memory-hungry:
  * local (sliding-window) layers of gqa_local: queries in chunks of 64, each with an explicit boolean mask.
    An explicit mask rules out SDPA's flash kernel, and grouped-query attention (enable_gqa) rules out the
    memory-efficient kernel, so PyTorch falls back to the math kernel, which materialises every score matrix;
  * GQA layers in general (enable_gqa=True).
This module patches Attention.forward for the self-attention calls made in training and inference (no KV cache,
no cross-attention; generation keeps the original code) and sends them to:
  fa2   FlashAttention-2 (pip package flash-attn): causal, sliding window (window_size=(W-1, 0) is exactly the
        Phase-1 rule "key j visible to query i iff i-W < j <= i") and GQA natively, O(L) memory, one kernel
        for the whole sequence instead of a loop over query chunks. fp16/bf16 on CUDA (Ampere+: RTX 3090 ok).
  sdpa  no extra package: key/value heads expanded to the query heads (so no enable_gqa), then
        is_causal=True for full layers (PyTorch's own flash kernel) and the memory-efficient kernel with the
        band mask for local layers (one call, no python loop over chunks).
  auto  fa2 if flash_attn imports and the tensors are on CUDA in fp16/bf16, else sdpa.
  off   the original Phase-1 code.
The math is unchanged: same masks, same RoPE, same dropout; tests compare against the original forward.
"""
import torch
from torch.nn import functional as F

_STATE = {"backend": "off", "original": None, "fa2": None, "allow_cpu": False}   # allow_cpu: tests only


def _flash_attn():
    if _STATE["fa2"] is None:
        try:
            from flash_attn import flash_attn_func
            _STATE["fa2"] = flash_attn_func
        except Exception:                                   # not installed / no CUDA build
            _STATE["fa2"] = False
    return _STATE["fa2"]


def band_mask(length, window, device):
    """Causal sliding window: query i sees keys j with i - window < j <= i."""
    i = torch.arange(length, device=device)[:, None]
    j = torch.arange(length, device=device)[None, :]
    return (j <= i) & (j > i - window)


def attend(q, k, v, causal, window, dropout, backend):
    """q [B,H,L,D], k/v [B,Hkv,L,D] -> [B,H,L,D]. window=None: no sliding window."""
    if k.dtype != q.dtype or v.dtype != q.dtype:            # e.g. a float32 adapter scale on keys/values
        k, v = k.to(q.dtype), v.to(q.dtype)
    fa2 = _flash_attn() if backend in ("fa2", "auto") else False
    if fa2 and ((q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)) or _STATE["allow_cpu"]):
        out = fa2(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout, causal=causal,
                  window_size=(window - 1, 0) if window else (-1, -1))
        return out.transpose(1, 2)
    if backend == "fa2" and not fa2:
        raise RuntimeError("--attention fa2: flash_attn is not importable (pip install flash-attn --no-build-isolation)")
    groups = q.size(1) // k.size(1)
    if groups > 1:                                          # expand KV heads: keeps the fast SDPA kernels eligible
        k = k.repeat_interleave(groups, 1)
        v = v.repeat_interleave(groups, 1)
    if window and q.size(2) > window:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=band_mask(q.size(2), window, q.device),
                                              dropout_p=dropout)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout)


def _patched_forward(self, x, context=None, past=None, causal=True, use_cache=False):
    if context is not None or past is not None or use_cache:
        return _STATE["original"](self, x, context=context, past=past, causal=causal, use_cache=use_cache)
    from phase1.models import rotary
    batch, length, dim = x.shape
    q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
    k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
    if self.rope:
        k = rotary(k, 0)
        q = rotary(q, 0)
    window = self.window if (self.local and causal) else None
    y = attend(q, k, v, causal, window, self.dropout if self.training else 0.0, _STATE["backend"])
    y = y.transpose(1, 2).contiguous().view(batch, length, dim)
    return self.o(y), None


def enable(backend="auto"):
    """Patch phase1.models.Attention (idempotent). Returns a short description of what is used."""
    from phase1 import models
    if _STATE["original"] is None:
        _STATE["original"] = models.Attention.forward
    if backend == "off":
        models.Attention.forward = _STATE["original"]
        _STATE["backend"] = "off"
        return "attention: original Phase-1 SDPA code"
    if backend == "fa2" and not _flash_attn():
        raise SystemExit("--attention fa2 needs the flash-attn package: pip install flash-attn --no-build-isolation")
    models.Attention.forward = _patched_forward
    _STATE["backend"] = backend
    have = bool(_flash_attn())
    if backend == "sdpa" or not have:
        return "attention: SDPA with expanded KV heads (flash / memory-efficient kernels; flash-attn not used)"
    return "attention: FlashAttention-2 (causal, sliding window and GQA in the kernel)"
