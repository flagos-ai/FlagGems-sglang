# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of attention/mla_kv_pack_quantize_fp8."""

import torch
import triton
import triton.language as tl

__all__ = ["mla_kv_pack_quantize_fp8"]

_FP8_DTYPE_MAP = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}


@triton.jit
def _per_head_kernel(
    k_nope_ptr,
    k_pe_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    k_scale_inv,
    v_scale_inv,
    s_total,
    k_nope_stride_t,
    k_nope_stride_h,
    k_pe_stride_t,
    v_stride_t,
    v_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    qk_nope,
    qk_rope,
    v_head,
    BLOCK_S: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    """One program owns a tile of tokens for one head."""
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    row_ok = rows < s_total

    # The masks against the real widths keep this correct for widths that are
    # not powers of two; the arange extents themselves must be.
    nope_idx = tl.arange(0, BLOCK_NOPE)
    nope_ok = nope_idx < qk_nope
    rope_idx = tl.arange(0, BLOCK_ROPE)
    rope_ok = rope_idx < qk_rope
    v_idx = tl.arange(0, BLOCK_V)
    v_ok = v_idx < v_head

    nope_ptrs = (
        k_nope_ptr
        + rows[:, None] * k_nope_stride_t
        + pid_h * k_nope_stride_h
        + nope_idx[None, :]
    )
    k_nope = tl.load(
        nope_ptrs, mask=row_ok[:, None] & nope_ok[None, :], other=0.0
    )

    pe_ptrs = k_pe_ptr + rows[:, None] * k_pe_stride_t + rope_idx[None, :]
    k_pe = tl.load(pe_ptrs, mask=row_ok[:, None] & rope_ok[None, :], other=0.0)

    v_ptrs = (
        v_ptr
        + rows[:, None] * v_stride_t
        + pid_h * v_stride_h
        + v_idx[None, :]
    )
    v_tile = tl.load(v_ptrs, mask=row_ok[:, None] & v_ok[None, :], other=0.0)

    # The inverse scales are per tensor, and the product is formed in fp32 so
    # the single rounding to fp8 happens on the scaled value.
    k_nope_fp8 = (k_nope.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    k_pe_fp8 = (k_pe.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    v_fp8 = (v_tile.to(tl.float32) * v_scale_inv).to(FP8_DTYPE)

    out_base = (
        k_out_ptr + rows[:, None] * k_out_stride_t + pid_h * k_out_stride_h
    )
    tl.store(
        out_base + nope_idx[None, :],
        k_nope_fp8,
        mask=row_ok[:, None] & nope_ok[None, :],
    )
    tl.store(
        out_base + qk_nope + rope_idx[None, :],
        k_pe_fp8,
        mask=row_ok[:, None] & rope_ok[None, :],
    )

    v_out_ptrs = (
        v_out_ptr
        + rows[:, None] * v_out_stride_t
        + pid_h * v_out_stride_h
        + v_idx[None, :]
    )
    tl.store(v_out_ptrs, v_fp8, mask=row_ok[:, None] & v_ok[None, :])


@triton.jit
def _flat_kernel(
    k_nope_ptr,
    k_pe_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    k_scale_inv,
    v_scale_inv,
    s_total,
    num_heads,
    k_nope_stride_t,
    k_nope_stride_h,
    k_pe_stride_t,
    v_stride_t,
    v_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    qk_nope,
    qk_rope,
    v_head,
    BLOCK: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    """One program owns a run of flattened (token, head) pairs."""
    pid = tl.program_id(0)
    pair = pid * BLOCK + tl.arange(0, BLOCK)
    pair_ok = pair < s_total * num_heads
    # Splitting the pair index is what lets a single flat grid cover every
    # (token, head) without a second grid dimension.
    t_idx = pair // num_heads
    h_idx = pair % num_heads

    nope_idx = tl.arange(0, BLOCK_NOPE)
    nope_ok = nope_idx < qk_nope
    rope_idx = tl.arange(0, BLOCK_ROPE)
    rope_ok = rope_idx < qk_rope
    v_idx = tl.arange(0, BLOCK_V)
    v_ok = v_idx < v_head

    nope_ptrs = (
        k_nope_ptr
        + t_idx[:, None] * k_nope_stride_t
        + h_idx[:, None] * k_nope_stride_h
        + nope_idx[None, :]
    )
    k_nope = tl.load(
        nope_ptrs, mask=pair_ok[:, None] & nope_ok[None, :], other=0.0
    )

    pe_ptrs = k_pe_ptr + t_idx[:, None] * k_pe_stride_t + rope_idx[None, :]
    k_pe = tl.load(
        pe_ptrs, mask=pair_ok[:, None] & rope_ok[None, :], other=0.0
    )

    v_ptrs = (
        v_ptr
        + t_idx[:, None] * v_stride_t
        + h_idx[:, None] * v_stride_h
        + v_idx[None, :]
    )
    v_tile = tl.load(v_ptrs, mask=pair_ok[:, None] & v_ok[None, :], other=0.0)

    k_nope_fp8 = (k_nope.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    k_pe_fp8 = (k_pe.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    v_fp8 = (v_tile.to(tl.float32) * v_scale_inv).to(FP8_DTYPE)

    out_base = (
        k_out_ptr
        + t_idx[:, None] * k_out_stride_t
        + h_idx[:, None] * k_out_stride_h
    )
    tl.store(
        out_base + nope_idx[None, :],
        k_nope_fp8,
        mask=pair_ok[:, None] & nope_ok[None, :],
    )
    tl.store(
        out_base + qk_nope + rope_idx[None, :],
        k_pe_fp8,
        mask=pair_ok[:, None] & rope_ok[None, :],
    )

    v_out_ptrs = (
        v_out_ptr
        + t_idx[:, None] * v_out_stride_t
        + h_idx[:, None] * v_out_stride_h
        + v_idx[None, :]
    )
    tl.store(v_out_ptrs, v_fp8, mask=pair_ok[:, None] & v_ok[None, :])


def _pick_launch(s):
    """Pick the launch shape and tile for a token count.

    The crossover points mirror the upstream tuned table: below two tokens the
    launch cost dominates and a tiny per-head tile is cheapest; up to 16 tokens
    the per-head shape still wins; from 32 to 192 flattened pairs leave the
    mostly-idle head dimension behind; 1536 tokens is wide enough for the
    per-head shape to pay again; past that the flat shape takes over for good.
    """
    if s <= 2:
        return "per_head", 1, 1, 2
    if s <= 16:
        return "per_head", 4, 2, 3
    if s <= 32:
        return "flat", 8, 8, 2
    if s <= 192:
        return "flat", 16, 8, 3
    if s <= 1536:
        return "per_head", 16, 4, 3
    return "flat", 16, 8, 3


def mla_kv_pack_quantize_fp8(
    k_nope,
    k_pe,
    v,
    k_scale_inv=1.0,
    v_scale_inv=1.0,
    fp8_dtype=torch.float8_e4m3fn,
):
    """Concat the rotary keys onto the non-rotary ones and quantize to fp8.

    The rotary part is broadcast across every head, so the packed key is
    ``[qk_nope + qk_rope]`` wide per head while the key input carries each
    rotary vector once. Both packed keys and the values are scaled by their
    inverse scales in fp32 and then saturated-cast to ``fp8_dtype``.

    Args:
        k_nope: ``(tokens, heads, qk_nope)``, bf16 or fp16.
        k_pe: ``(tokens, 1, qk_rope)`` or ``(tokens, qk_rope)``, same dtype.
        v: ``(tokens, heads, v_head)``, same dtype.
        k_scale_inv: inverse scale applied to the packed keys.
        v_scale_inv: inverse scale applied to the values.
        fp8_dtype: ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.

    Returns:
        ``(key, value)`` with ``key`` of shape ``(tokens, heads,
        qk_nope + qk_rope)`` and ``value`` of shape ``(tokens, heads, v_head)``,
        both in ``fp8_dtype``. Fresh tensors -- no input is mutated.
    """
    if k_nope.ndim != 3:
        raise ValueError("k_nope must be 3D (tokens, heads, qk_nope)")
    if v.ndim != 3:
        raise ValueError("v must be 3D (tokens, heads, v_head)")
    if k_pe.ndim not in (2, 3):
        raise ValueError(
            "k_pe must be 2D (tokens, qk_rope) or 3D (tokens, 1, qk_rope)"
        )
    if k_nope.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"k_nope must be bf16 or fp16, got {k_nope.dtype}")
    if k_pe.dtype != k_nope.dtype:
        raise ValueError("k_pe must share k_nope's dtype")
    if v.dtype != k_nope.dtype:
        raise ValueError("v must share k_nope's dtype")
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported fp8_dtype {fp8_dtype}")

    s, num_heads, qk_nope = k_nope.shape
    qk_rope = int(k_pe.shape[-1])
    v_head = int(v.shape[-1])

    if k_pe.shape[0] != s:
        raise ValueError(f"k_pe has {k_pe.shape[0]} tokens, k_nope has {s}")
    if k_pe.ndim == 3 and k_pe.shape[1] != 1:
        raise ValueError(f"k_pe head dim must be 1, got {k_pe.shape[1]}")
    if v.shape[0] != s or v.shape[1] != num_heads:
        raise ValueError(
            f"v shape {tuple(v.shape)} mismatches k_nope {tuple(k_nope.shape)}"
        )

    # The kernels walk the last dimension with unit stride, so a strided
    # outer layout is fine but an inner-dim slice is not.
    for name, tensor in (("k_nope", k_nope), ("k_pe", k_pe), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous on the last dim")
    if k_nope.device != k_pe.device or k_nope.device != v.device:
        raise ValueError("all inputs must be on the same device")

    k_pe_2d = k_pe.squeeze(1) if k_pe.ndim == 3 else k_pe

    k_out = torch.empty(
        (s, num_heads, qk_nope + qk_rope),
        dtype=fp8_dtype,
        device=k_nope.device,
    )
    v_out = torch.empty(
        (s, num_heads, v_head), dtype=fp8_dtype, device=v.device
    )

    # A zero-length grid dimension launches nothing and would leave the
    # outputs as allocator garbage; they are already shape-correct and empty.
    if s == 0 or num_heads == 0 or v_head == 0:
        return k_out, v_out

    tl_dtype = _FP8_DTYPE_MAP[fp8_dtype]
    kind, block, num_warps, num_stages = _pick_launch(s)

    # arange needs power-of-two extents; the in-kernel masks handle the real
    # widths, so no input shape has to be rejected for being odd.
    block_nope = triton.next_power_of_2(max(qk_nope, 1))
    block_rope = triton.next_power_of_2(max(qk_rope, 1))
    block_v = triton.next_power_of_2(max(v_head, 1))

    if kind == "per_head":
        grid = (triton.cdiv(s, block), num_heads)
        _per_head_kernel[grid](
            k_nope,
            k_pe_2d,
            v,
            k_out,
            v_out,
            float(k_scale_inv),
            float(v_scale_inv),
            s,
            k_nope.stride(0),
            k_nope.stride(1),
            k_pe_2d.stride(0),
            v.stride(0),
            v.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            v_out.stride(0),
            v_out.stride(1),
            qk_nope,
            qk_rope,
            v_head,
            BLOCK_S=block,
            BLOCK_NOPE=block_nope,
            BLOCK_ROPE=block_rope,
            BLOCK_V=block_v,
            FP8_DTYPE=tl_dtype,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        total = s * num_heads
        grid = (triton.cdiv(total, block),)
        _flat_kernel[grid](
            k_nope,
            k_pe_2d,
            v,
            k_out,
            v_out,
            float(k_scale_inv),
            float(v_scale_inv),
            s,
            num_heads,
            k_nope.stride(0),
            k_nope.stride(1),
            k_pe_2d.stride(0),
            v.stride(0),
            v.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            v_out.stride(0),
            v_out.stride(1),
            qk_nope,
            qk_rope,
            v_head,
            BLOCK=block,
            BLOCK_NOPE=block_nope,
            BLOCK_ROPE=block_rope,
            BLOCK_V=block_v,
            FP8_DTYPE=tl_dtype,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return k_out, v_out
