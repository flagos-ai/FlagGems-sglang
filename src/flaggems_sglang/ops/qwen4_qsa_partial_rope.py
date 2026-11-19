# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""qwen4_qsa_partial_rope; SGLang layout and rounding contract.

Extracted from the validated Qwen3.8 SGLang baseline; this is not a vLLM kernel copy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_partial_rope_kernel(
    X,
    P,
    C,
    AX,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    POS2D: tl.constexpr,
    NEOX: tl.constexpr,
    B: tl.constexpr,
):
    ix = tl.program_id(0) * B + tl.arange(0, B)
    valid = ix < N * H * D
    row = ix // (H * D)
    head = ix // D % H
    col = ix % D
    rotary = col < R
    if NEOX:
        pair = col % (R // 2)
        first = col < R // 2
        partner = tl.where(first, col + R // 2, col - R // 2)
    else:
        pair = col // 2
        first = (col % 2) == 0
        partner = col ^ 1
    axis = tl.load(AX + pair, mask=valid & rotary, other=0) if POS2D else 0
    pos = tl.load(P + axis * PS0 + row * PS1, mask=valid & rotary, other=0).to(
        tl.int64
    )
    # Cache capacity is checked/grown by the caller once per model forward.
    co = (
        tl.load(C + pos * CS0 + pair * CS1, mask=valid & rotary, other=1)
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    si = (
        tl.load(
            C + pos * CS0 + (pair + R // 2) * CS1, mask=valid & rotary, other=0
        )
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    x = tl.load(
        X + row * XS0 + head * XS1 + col * XS2, mask=valid, other=0
    ).to(tl.float32)
    y = tl.load(
        X + row * XS0 + head * XS1 + partner * XS2,
        mask=valid & rotary,
        other=0,
    ).to(tl.float32)
    # Native apply_rotary_emb rounds each multiply before the add/sub.
    p = (x * co).to(X.dtype.element_ty).to(tl.float32)
    q = (y * si).to(X.dtype.element_ty).to(tl.float32)
    out = tl.where(first, p - q, p + q)
    tl.store(Y + ix, tl.where(rotary, out, x), mask=valid)


def qwen4_qsa_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    axis_map: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> torch.Tensor:
    if x.ndim != 3 or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Expected BF16/FP16 [tokens,heads,dim]")
    n, h, d = x.shape
    if rotary_dim <= 0 or rotary_dim > d or rotary_dim % 2:
        raise ValueError("Invalid rotary width")
    if positions.ndim not in (1, 2) or positions.shape[-1] != n:
        raise ValueError("Position shape mismatch")
    if positions.ndim == 2 and positions.shape[0] != 3:
        raise ValueError("Expected three MRoPE axes")
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if not n:
        return out
    _qsa_partial_rope_kernel[(triton.cdiv(n * h * d, 256),)](
        x,
        positions,
        cache,
        axis_map,
        out,
        n,
        h,
        d,
        rotary_dim,
        *x.stride(),
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.stride(-1),
        *cache.stride(),
        positions.ndim == 2,
        is_neox_style,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["qwen4_qsa_partial_rope"]
