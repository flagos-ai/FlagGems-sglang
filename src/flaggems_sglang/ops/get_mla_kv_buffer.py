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

"""Triton implementation of kvcache/get_mla_kv_buffer."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS": 2}, num_warps=4, num_stages=1),
        triton.Config({"ROWS": 4}, num_warps=4, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _gather_split_rows_kernel(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    N,
    stride_kv,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    # Two adjacent row blocks per program; all four gather loads are issued
    # before the stores so both blocks' misses are in flight together.
    row0 = pid * (2 * ROWS) + tl.arange(0, ROWS)
    row1 = row0 + ROWS
    mask0 = row0 < N
    mask1 = row1 < N

    src0 = tl.load(loc_ptr + row0 * stride_loc, mask=mask0, other=0).to(
        tl.int64
    )
    src1 = tl.load(loc_ptr + row1 * stride_loc, mask=mask1, other=0).to(
        tl.int64
    )
    src_ptr0 = kv_ptr + src0[:, None] * stride_kv
    src_ptr1 = kv_ptr + src1[:, None] * stride_kv

    d_n = tl.arange(0, BLOCK_N)
    d_r = tl.arange(0, BLOCK_R)

    # Gathered rows: keep in L2 (re-read across calls; rows are ~1 KB).
    if BLOCK_N == NOPE_D:
        vals0_n = tl.load(
            src_ptr0 + d_n[None, :],
            mask=mask0[:, None],
            other=0,
            eviction_policy="evict_last",
        )
        vals1_n = tl.load(
            src_ptr1 + d_n[None, :],
            mask=mask1[:, None],
            other=0,
            eviction_policy="evict_last",
        )
    else:
        col_mask_n = d_n < NOPE_D
        vals0_n = tl.load(
            src_ptr0 + d_n[None, :],
            mask=mask0[:, None] & col_mask_n[None, :],
            other=0,
            eviction_policy="evict_last",
        )
        vals1_n = tl.load(
            src_ptr1 + d_n[None, :],
            mask=mask1[:, None] & col_mask_n[None, :],
            other=0,
            eviction_policy="evict_last",
        )

    if BLOCK_R == ROPE_D:
        vals0_r = tl.load(
            src_ptr0 + NOPE_D + d_r[None, :],
            mask=mask0[:, None],
            other=0,
            eviction_policy="evict_last",
        )
        vals1_r = tl.load(
            src_ptr1 + NOPE_D + d_r[None, :],
            mask=mask1[:, None],
            other=0,
            eviction_policy="evict_last",
        )
    else:
        col_mask_r = d_r < ROPE_D
        vals0_r = tl.load(
            src_ptr0 + NOPE_D + d_r[None, :],
            mask=mask0[:, None] & col_mask_r[None, :],
            other=0,
            eviction_policy="evict_last",
        )
        vals1_r = tl.load(
            src_ptr1 + NOPE_D + d_r[None, :],
            mask=mask1[:, None] & col_mask_r[None, :],
            other=0,
            eviction_policy="evict_last",
        )

    dst0_n = nope_ptr + row0[:, None] * stride_n
    dst1_n = nope_ptr + row1[:, None] * stride_n
    dst0_r = rope_ptr + row0[:, None] * stride_r
    dst1_r = rope_ptr + row1[:, None] * stride_r

    # Streamed-out outputs: evict first, they are never re-read here.
    if BLOCK_N == NOPE_D:
        tl.store(
            dst0_n + d_n[None, :],
            vals0_n.to(nope_ptr.dtype.element_ty),
            mask=mask0[:, None],
            eviction_policy="evict_first",
        )
        tl.store(
            dst1_n + d_n[None, :],
            vals1_n.to(nope_ptr.dtype.element_ty),
            mask=mask1[:, None],
            eviction_policy="evict_first",
        )
    else:
        col_mask_n = d_n < NOPE_D
        tl.store(
            dst0_n + d_n[None, :],
            vals0_n.to(nope_ptr.dtype.element_ty),
            mask=mask0[:, None] & col_mask_n[None, :],
            eviction_policy="evict_first",
        )
        tl.store(
            dst1_n + d_n[None, :],
            vals1_n.to(nope_ptr.dtype.element_ty),
            mask=mask1[:, None] & col_mask_n[None, :],
            eviction_policy="evict_first",
        )

    if BLOCK_R == ROPE_D:
        tl.store(
            dst0_r + d_r[None, :],
            vals0_r.to(rope_ptr.dtype.element_ty),
            mask=mask0[:, None],
            eviction_policy="evict_first",
        )
        tl.store(
            dst1_r + d_r[None, :],
            vals1_r.to(rope_ptr.dtype.element_ty),
            mask=mask1[:, None],
            eviction_policy="evict_first",
        )
    else:
        col_mask_r = d_r < ROPE_D
        tl.store(
            dst0_r + d_r[None, :],
            vals0_r.to(rope_ptr.dtype.element_ty),
            mask=mask0[:, None] & col_mask_r[None, :],
            eviction_policy="evict_first",
        )
        tl.store(
            dst1_r + d_r[None, :],
            vals1_r.to(rope_ptr.dtype.element_ty),
            mask=mask1[:, None] & col_mask_r[None, :],
            eviction_policy="evict_first",
        )


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    n_loc = loc.shape[0]

    nope = torch.empty_like(cache_k_nope)
    rope = torch.empty_like(cache_k_rope)
    if n_loc == 0:
        return nope, rope

    def grid(meta):
        return (triton.cdiv(n_loc, 2 * meta["ROWS"]),)

    _gather_split_rows_kernel[grid](
        kv_buffer,
        loc,
        nope,
        rope,
        n_loc,
        kv_buffer.stride(0),
        loc.stride(0),
        nope.stride(0),
        rope.stride(0),
        NOPE_D=nope_dim,
        ROPE_D=rope_dim,
        BLOCK_N=triton.next_power_of_2(nope_dim),
        BLOCK_R=triton.next_power_of_2(rope_dim),
    )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
