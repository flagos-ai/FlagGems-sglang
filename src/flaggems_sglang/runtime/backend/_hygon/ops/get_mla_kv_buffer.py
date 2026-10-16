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

import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1


@triton.jit
def _gather_split_kernel(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    n_loc,
    KV_W: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
    INDEX64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    if NEED_MASK:
        mask_m = offs_m < n_loc
        rows = tl.load(loc_ptr + offs_m, mask=mask_m, other=0)
    else:
        rows = tl.load(loc_ptr + offs_m)
    if INDEX64:
        rows = rows.to(tl.int64)
    else:
        rows = rows.to(tl.int32)
    base = tl.max_contiguous(tl.multiple_of(rows * KV_W, KV_W), KV_W)

    nope_cols = tl.arange(0, NOPE_DIM)
    if NEED_MASK:
        nope_vals = tl.load(
            kv_ptr + base[:, None] + nope_cols[None, :],
            mask=mask_m[:, None],
            other=0,
        )
        tl.store(
            nope_ptr + offs_m[:, None] * NOPE_DIM + nope_cols[None, :],
            nope_vals,
            mask=mask_m[:, None],
        )
    else:
        nope_vals = tl.load(kv_ptr + base[:, None] + nope_cols[None, :])
        tl.store(
            nope_ptr + offs_m[:, None] * NOPE_DIM + nope_cols[None, :],
            nope_vals,
        )

    rope_cols = tl.arange(0, ROPE_DIM)
    if NEED_MASK:
        rope_vals = tl.load(
            kv_ptr + base[:, None] + NOPE_DIM + rope_cols[None, :],
            mask=mask_m[:, None],
            other=0,
        )
        tl.store(
            rope_ptr + offs_m[:, None] * ROPE_DIM + rope_cols[None, :],
            rope_vals,
            mask=mask_m[:, None],
        )
    else:
        rope_vals = tl.load(
            kv_ptr + base[:, None] + NOPE_DIM + rope_cols[None, :]
        )
        tl.store(
            rope_ptr + offs_m[:, None] * ROPE_DIM + rope_cols[None, :],
            rope_vals,
        )


@triton.jit
def _gather_combined_kernel(
    kv_ptr,
    loc_ptr,
    out_ptr,
    n_loc,
    KV_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
    INDEX64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    if NEED_MASK:
        mask_m = offs_m < n_loc
        rows = tl.load(loc_ptr + offs_m, mask=mask_m, other=0)
    else:
        rows = tl.load(loc_ptr + offs_m)
    if INDEX64:
        rows = rows.to(tl.int64)
    else:
        rows = rows.to(tl.int32)
    base = tl.max_contiguous(tl.multiple_of(rows * KV_W, KV_W), KV_W)

    cols = tl.arange(0, BLOCK_K)
    mask_k = cols[None, :] < KV_W
    in_ptrs = kv_ptr + base[:, None] + cols[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * KV_W + cols[None, :]
    if NEED_MASK:
        mask = mask_k & mask_m[:, None]
        tl.store(out_ptrs, tl.load(in_ptrs, mask=mask, other=0), mask=mask)
    else:
        tl.store(out_ptrs, tl.load(in_ptrs, mask=mask_k, other=0), mask=mask_k)


@triton.jit
def _gather_split_strided_kernel(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    n_loc,
    stride_kv_m,
    stride_kv_k,
    stride_nope_m,
    stride_rope_m,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_loc
    rows = tl.load(loc_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)

    nope_cols = tl.arange(0, BLOCK_NOPE)
    mask_nope = mask_m[:, None] & (nope_cols[None, :] < NOPE_DIM)
    nope_vals = tl.load(
        kv_ptr
        + rows[:, None] * stride_kv_m
        + nope_cols[None, :] * stride_kv_k,
        mask=mask_nope,
        other=0,
    )
    tl.store(
        nope_ptr + offs_m[:, None] * stride_nope_m + nope_cols[None, :],
        nope_vals,
        mask=mask_nope,
    )

    rope_cols = tl.arange(0, BLOCK_ROPE)
    mask_rope = mask_m[:, None] & (rope_cols[None, :] < ROPE_DIM)
    rope_vals = tl.load(
        kv_ptr
        + rows[:, None] * stride_kv_m
        + (NOPE_DIM + rope_cols[None, :]) * stride_kv_k,
        mask=mask_rope,
        other=0,
    )
    tl.store(
        rope_ptr + offs_m[:, None] * stride_rope_m + rope_cols[None, :],
        rope_vals,
        mask=mask_rope,
    )


_COMBINED_MIN_ROWS = 2048


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    n_loc = loc.shape[0]

    if n_loc == 0:
        return (
            torch.empty_like(cache_k_nope),
            torch.empty_like(cache_k_rope),
        )

    kv_width = nope_dim + rope_dim
    contiguous = (
        kv_buffer.stride(1) == 1
        and kv_buffer.stride(0) == kv_width
        and cache_k_nope.is_contiguous()
        and cache_k_rope.is_contiguous()
        and loc.stride(0) == 1
    )

    index64 = kv_buffer.shape[0] * kv_width > _INT32_MAX

    if not contiguous:
        nope = torch.empty_like(cache_k_nope)
        rope = torch.empty_like(cache_k_rope)
        block_nope = triton.next_power_of_2(nope_dim)
        block_rope = triton.next_power_of_2(rope_dim)
        _gather_split_strided_kernel[(triton.cdiv(n_loc, 4),)](
            kv_buffer,
            loc,
            nope,
            rope,
            n_loc,
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            nope.stride(0),
            rope.stride(0),
            NOPE_DIM=nope_dim,
            ROPE_DIM=rope_dim,
            BLOCK_NOPE=block_nope,
            BLOCK_ROPE=block_rope,
            BLOCK_M=4,
            num_warps=4,
        )
        return nope, rope

    combined = (
        n_loc > _COMBINED_MIN_ROWS and cache_k_nope.dtype == cache_k_rope.dtype
    )
    if combined:
        out = torch.empty(
            (n_loc, kv_width),
            dtype=cache_k_nope.dtype,
            device=kv_buffer.device,
        )
        _gather_combined_kernel[(triton.cdiv(n_loc, 8),)](
            kv_buffer,
            loc,
            out,
            n_loc,
            KV_W=kv_width,
            BLOCK_K=triton.next_power_of_2(kv_width),
            BLOCK_M=8,
            NEED_MASK=(n_loc % 8) != 0,
            INDEX64=index64,
            num_warps=1,
        )
        return out[:, :nope_dim], out[:, nope_dim:]

    nope = torch.empty_like(cache_k_nope)
    rope = torch.empty_like(cache_k_rope)
    if n_loc <= 1024:
        block_m, num_warps = 1, 2
    else:
        block_m, num_warps = 8, 1
    _gather_split_kernel[(triton.cdiv(n_loc, block_m),)](
        kv_buffer,
        loc,
        nope,
        rope,
        n_loc,
        KV_W=kv_width,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_M=block_m,
        NEED_MASK=(n_loc % block_m) != 0,
        INDEX64=index64,
        num_warps=num_warps,
    )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
