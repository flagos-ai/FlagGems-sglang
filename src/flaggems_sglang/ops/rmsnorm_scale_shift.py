# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused RMSNorm with per-batch scale and shift.

Inputs: x [B, S, D], weight [D], scale/shift [B, 1, D].
The generic path requires positive power-of-two D divisible by 128.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _refined_rsqrt(variance, EPS: tl.constexpr):
    variance_work = variance
    if EPS < 1.1754943508222875e-38:
        subnormal = (variance > 0.0) & (variance < 1.1754943508222875e-38)
        variance_work = tl.where(subnormal, variance * 16777216.0, variance)
    inv = tl.rsqrt(variance_work)
    prod = variance_work * inv
    err = tl.fma(-inv, prod, 1.0)
    err = tl.fma(-inv, tl.fma(variance_work, inv, -prod), err)
    inv = tl.fma(0.5 * inv, err, inv)
    if EPS < 1.1754943508222875e-38:
        inv = tl.where(subnormal, inv * 4096.0, inv)
        inv = tl.where(
            (variance > 0.0) & (variance < float("inf")),
            inv,
            tl.rsqrt(variance),
        )
    else:
        inv = tl.where(variance == float("inf"), 0.0, inv)
    return inv


@triton.jit
def _scale_shift(norm, scale, shift, out_ptr, scale_ptr, IS_HIP: tl.constexpr):
    if IS_HIP:
        norm = norm.to(out_ptr.dtype.element_ty).to(tl.float32)
        gain = (scale + 1.0).to(scale_ptr.dtype.element_ty).to(tl.float32)
        product = (norm * gain).to(out_ptr.dtype.element_ty).to(tl.float32)
        return product + shift
    else:
        # Keep the BF16 rounding stages explicit, without inline PTX.
        norm = norm.to(tl.bfloat16).to(tl.float32)
        scale = scale.to(tl.bfloat16).to(tl.float32)
        shift = shift.to(tl.bfloat16).to(tl.float32)
        gain = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
        product = (norm * gain).to(tl.bfloat16).to(tl.float32)
        return (product + shift).to(tl.bfloat16)


@triton.jit
def _rmsnorm_scale_shift_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    X_STRIDES: tl.constexpr,
    W_S: tl.constexpr,
    SCALE_STRIDES: tl.constexpr,
    SHIFT_STRIDES: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_HIP: tl.constexpr,
    USE_CG: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQ_LEN
    seq = row % SEQ_LEN
    ptr = x_ptr + batch * X_STRIDES[0] + seq * X_STRIDES[1]
    q = tl.arange(0, 128)
    accum = tl.full((128,), 0, tl.float32)
    for start in tl.static_range(0, D, 128):
        v = tl.load(
            ptr + (start + q) * X_STRIDES[2],
            cache_modifier=".cg" if USE_CG else "",
        ).to(tl.float32)
        accum = accum + v * v
    even, odd = tl.split(tl.reshape(accum, (32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    acc = a0 + a1 + a2 + a3
    lane = tl.arange(0, 32)
    if IS_HIP:
        reverse = (
            (lane & 1) << 3
            | (lane & 2) << 1
            | (lane & 4) >> 1
            | (lane & 8) >> 3
            | lane & 16
        )
    else:
        reverse = (
            (lane & 1) << 4
            | (lane & 2) << 2
            | lane & 4
            | (lane & 8) >> 2
            | (lane & 16) >> 4
        )
    acc = tl.gather(acc, reverse, 0)
    variance = tl.sum(acc, 0) / D + EPS
    inv = _refined_rsqrt(variance, EPS)
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(
        ptr + cols * X_STRIDES[2],
        cols < D,
        other=0,
        cache_modifier=".cg" if USE_CG else "",
    ).to(tl.float32)
    w = tl.load(
        weight_ptr + cols * W_S,
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + batch * SCALE_STRIDES[0] + cols * SCALE_STRIDES[2],
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + batch * SHIFT_STRIDES[0] + cols * SHIFT_STRIDES[2],
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    value = _scale_shift(x * inv * w, scale, shift, out_ptr, scale_ptr, IS_HIP)
    tl.store(
        out_ptr + row * D + cols,
        value,
        cols < D,
        cache_modifier="" if IS_HIP else ".cs",
    )


@triton.jit
def _split_2048_chunks(x, BLOCK_ROWS: tl.constexpr):
    # Original split layout; tuple order is ascending 128-column chunks.
    group_0 = tl.trans(tl.reshape(x, (BLOCK_ROWS, 16, 128)), (0, 2, 1))
    (group_1, group_2) = tl.split(tl.reshape(group_0, (BLOCK_ROWS, 128, 8, 2)))
    (group_3, group_4) = tl.split(tl.reshape(group_1, (BLOCK_ROWS, 128, 4, 2)))
    (group_5, group_6) = tl.split(tl.reshape(group_3, (BLOCK_ROWS, 128, 2, 2)))
    (chunk_0, chunk_8) = tl.split(group_5)
    (chunk_4, chunk_12) = tl.split(group_6)
    (group_7, group_8) = tl.split(tl.reshape(group_4, (BLOCK_ROWS, 128, 2, 2)))
    (chunk_2, chunk_10) = tl.split(group_7)
    (chunk_6, chunk_14) = tl.split(group_8)
    (group_9, group_10) = tl.split(
        tl.reshape(group_2, (BLOCK_ROWS, 128, 4, 2))
    )
    (group_11, group_12) = tl.split(
        tl.reshape(group_9, (BLOCK_ROWS, 128, 2, 2))
    )
    (chunk_1, chunk_9) = tl.split(group_11)
    (chunk_5, chunk_13) = tl.split(group_12)
    (group_13, group_14) = tl.split(
        tl.reshape(group_10, (BLOCK_ROWS, 128, 2, 2))
    )
    (chunk_3, chunk_11) = tl.split(group_13)
    (chunk_7, chunk_15) = tl.split(group_14)
    return (
        chunk_0,
        chunk_1,
        chunk_2,
        chunk_3,
        chunk_4,
        chunk_5,
        chunk_6,
        chunk_7,
        chunk_8,
        chunk_9,
        chunk_10,
        chunk_11,
        chunk_12,
        chunk_13,
        chunk_14,
        chunk_15,
    )


@triton.jit
def _rmsnorm_scale_shift_dense_split_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    batch = tl.program_id(1)
    ptr = x_ptr + batch * (SEQ_LEN * D) + row[:, None] * D
    cols = tl.arange(0, BLOCK_D)
    mask = (row[:, None] < SEQ_LEN) & (cols[None, :] < D)
    x = tl.load(ptr + cols[None, :], mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    # Static tuple indexing keeps the original split DAG and addition order.
    chunks = _split_2048_chunks(x, BLOCK_ROWS)
    accum = tl.full((BLOCK_ROWS, 128), 0, tl.float32)
    for i in tl.static_range(16):
        v = chunks[i]
        accum = accum + v * v

    # Keep the original grouping and HIP lane permutation before the final sum.
    even, odd = tl.split(tl.reshape(accum, (BLOCK_ROWS, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    acc = a0 + a1 + a2 + a3
    lane = tl.arange(0, 32)
    reverse = (
        (lane & 1) << 3
        | (lane & 2) << 1
        | (lane & 4) >> 1
        | (lane & 8) >> 3
        | lane & 16
    )
    acc = tl.gather(
        acc, tl.broadcast_to(reverse[None, :], (BLOCK_ROWS, 32)), axis=1
    )
    variance = tl.sum(acc, axis=1) / D + EPS
    inv = _refined_rsqrt(variance, EPS)
    w = tl.load(
        weight_ptr + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + batch * D + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + batch * D + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    value = _scale_shift(
        x * inv[:, None] * w[None, :],
        scale[None, :],
        shift[None, :],
        out_ptr,
        scale_ptr,
        True,
    )
    tl.store(
        out_ptr + (batch * SEQ_LEN + row[:, None]) * D + cols[None, :],
        value,
        mask,
        cache_modifier=".cs",
    )


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    batch, seq_len, hidden_size = x.shape
    rows = batch * seq_len
    is_hip = torch.version.hip is not None
    dense = all(t.is_contiguous() for t in (x, weight, scale, shift))
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out

    if dense and is_hip and hidden_size == 2048 and rows >= 2048:
        _rmsnorm_scale_shift_dense_split_kernel[
            triton.cdiv(seq_len, 4), batch
        ](
            x,
            weight,
            scale,
            shift,
            out,
            seq_len,
            hidden_size,
            float(eps),
            hidden_size,
            4,
            num_warps=8 if rows > 16384 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        # constexpr strides specialize to the same addressing for dense inputs.
        _rmsnorm_scale_shift_kernel[rows,](
            x,
            weight,
            scale,
            shift,
            out,
            seq_len,
            hidden_size,
            tuple(x.stride()),
            weight.stride(0),
            tuple(scale.stride()),
            tuple(shift.stride()),
            float(eps),
            hidden_size,
            is_hip,
            is_hip and rows <= 16384,
            num_warps=8 if is_hip else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["rmsnorm_scale_shift"]
