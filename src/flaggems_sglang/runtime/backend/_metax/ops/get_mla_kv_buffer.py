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

"""get_mla_kv_buffer: gather paged MLA KV cache rows and split."""

import torch
import triton
import triton.language as tl


@triton.jit
def _get_mla_kv_buffer_kernel(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    n_rows,
    NOPE: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    INT32_IDX: tl.constexpr,
):
    """Copy ``ROWS`` gathered rows: ``nope[i] = kv[loc[i], :NOPE]`` and"""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    loc = tl.load(loc_ptr + rows, mask=rmask, other=0)
    if INT32_IDX:
        loc = loc.to(tl.int32)
    row_stride: tl.constexpr = NOPE + ROPE

    # NoPE half: columns [0, NOPE)
    cols = tl.arange(0, NOPE_BLOCK)
    mask = rmask[:, None] & (cols < NOPE)[None, :]
    src = loc[:, None] * row_stride + cols[None, :]
    v = tl.load(
        kv_ptr + src, mask=mask, other=0, eviction_policy="evict_first"
    )
    v = v.to(nope_ptr.dtype.element_ty)
    tl.store(nope_ptr + rows[:, None] * NOPE + cols[None, :], v, mask=mask)

    # RoPE half: columns [NOPE, NOPE+ROPE)
    rcols = tl.arange(0, ROPE_BLOCK)
    rmask2 = rmask[:, None] & (rcols < ROPE)[None, :]
    src2 = loc[:, None] * row_stride + NOPE + rcols[None, :]
    v2 = tl.load(
        kv_ptr + src2, mask=rmask2, other=0, eviction_policy="evict_first"
    )
    v2 = v2.to(rope_ptr.dtype.element_ty)
    tl.store(rope_ptr + rows[:, None] * ROPE + rcols[None, :], v2, mask=rmask2)


def _pow2(x):
    """Smallest power of two >= x (branch-free via bit_length, no."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    n_rows = loc.shape[0]

    # Fresh contiguous outputs (the reference's .to() results are contiguous).
    nope = torch.empty(
        cache_k_nope.shape,
        dtype=cache_k_nope.dtype,
        device=cache_k_nope.device,
    )
    rope = torch.empty(
        cache_k_rope.shape,
        dtype=cache_k_rope.dtype,
        device=cache_k_rope.device,
    )
    if n_rows == 0:
        return nope, rope

    # Normalize layouts only when needed; the benchmark passes contiguous
    # tensors, so these checks are just cheap Python attribute reads.
    if loc.dim() != 1 or loc.stride(0) != 1:
        loc = loc.contiguous()
    if not kv_buffer.is_contiguous():
        kv_buffer = kv_buffer.contiguous()

    # Flat int32 addressing is safe when both the source element span and
    # the per-tensor destination spans stay below 2^31.
    int32_idx = (
        kv_buffer.numel() < (1 << 31)
        and n_rows * nope_dim < (1 << 31)
        and n_rows * rope_dim < (1 << 31)
        and nope_dim + rope_dim < (1 << 31)
    )

    # Fixed launch config: 2 rows per program, 1 warp (measured optimal at
    # the bandwidth-bound sizes, within noise at launch-bound small sizes).
    _get_mla_kv_buffer_kernel[((n_rows + 1) >> 1,)](
        kv_buffer,
        loc,
        nope,
        rope,
        n_rows,
        nope_dim,
        _pow2(nope_dim),
        rope_dim,
        _pow2(rope_dim),
        2,
        int32_idx,
        num_warps=1,
    )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
