# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton implementation of kvcache/get_mla_kv_buffer."""

import os

import torch
import triton
import triton.language as tl

# GCU launch limit (triton.backends.enflame.backend.GcuLauncher): grid.x
# must stay within 65535; beyond that the row blocks are walked by a
# persistent grid-stride kernel instead.
_GRID_X_MAX = 65535
# One program per SP on the 48-SP part: the per-program fixed cost
# (measured on L600) dominates small blocks, so the persistent form lets
# each SP amortize it across many 8-row blocks. Grids above 48 (96/144/192)
# and plain per-block grids all measured worse (E-1 third round sweep).
_PERSISTENT_GRID = 48
# The batch kernels stage the gathered rows in on-chip memory at roughly
# 25 B per lane; 8 rows of BLOCK_N + BLOCK_R lanes must stay far below
# the ~448 KB per-SP budget (u64 x 576 lanes = 36864 lanes already
# fails compilation with "dsmSize ... Insufficient memory").
_FAST_PATH_MAX_LANES = 16384
_ROWS_MANY = 4


@triton.jit
def _gather_split_row(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    row,
    stride_kv,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Per-row body with strictly 1D affine access: the historical
    # (ROWS, BLOCK) 2D tile form fails to freshly compile on the current
    # GCU toolchain (memref_ext.slice_deslice_start/slice_start segment
    # mismatch, see compiler_issues/sglang_get_mla_kv_buffer.md).
    d_n = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
    )
    d_r = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R
    )
    # loc enters as an int32 word view (host-side, zero-copy; see the host
    # comment): the scalar load reads the true 4-byte slot words. int64 loc
    # tensors are either silently stored as compact int32 words (env gate
    # off) or hold the value in the low word of each 8-byte slot (gate on);
    # a load at the declared int64 stride would walk 8-byte slots either
    # way and gather garbage slot ids -> wild gather addresses -> device
    # abort (compiler_issues/triton_i64_narrow_load.md).
    # The .to(tl.int64) below is only an i32 -> i64 widening of the value.
    src = tl.load(loc_ptr + row * stride_loc).to(tl.int64)

    if BLOCK_N == NOPE_D:
        vals_n = tl.load(kv_ptr + src * stride_kv + d_n)
    else:
        vals_n = tl.load(
            kv_ptr + src * stride_kv + d_n, mask=d_n < NOPE_D, other=0
        )
    if BLOCK_R == ROPE_D:
        vals_r = tl.load(kv_ptr + (src * stride_kv + NOPE_D) + d_r)
    else:
        vals_r = tl.load(
            kv_ptr + (src * stride_kv + NOPE_D) + d_r,
            mask=d_r < ROPE_D,
            other=0,
        )

    if BLOCK_N == NOPE_D:
        tl.store(
            nope_ptr + row * stride_n + d_n,
            vals_n.to(nope_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            nope_ptr + row * stride_n + d_n,
            vals_n.to(nope_ptr.dtype.element_ty),
            mask=d_n < NOPE_D,
        )
    if BLOCK_R == ROPE_D:
        tl.store(
            rope_ptr + row * stride_r + d_r,
            vals_r.to(rope_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            rope_ptr + row * stride_r + d_r,
            vals_r.to(rope_ptr.dtype.element_ty),
            mask=d_r < ROPE_D,
        )


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
    base = tl.program_id(0) * ROWS
    for r in tl.static_range(ROWS):
        row = base + r
        if row < N:
            _gather_split_row(
                kv_ptr,
                loc_ptr,
                nope_ptr,
                rope_ptr,
                row,
                stride_kv,
                stride_loc,
                stride_n,
                stride_r,
                NOPE_D,
                ROPE_D,
                BLOCK_N,
                BLOCK_R,
            )


@triton.jit
def _gather_split_rows_kernel_many(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    N_BLOCKS,
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
    # Persistent grid-stride form for block counts beyond the GCU grid.x
    # limit: each program walks row blocks pid, pid + P, pid + 2P, ...
    for pid in range(tl.program_id(0), N_BLOCKS, tl.num_programs(0)):
        base = pid * ROWS
        for r in tl.static_range(ROWS):
            row = base + r
            if row < N:
                _gather_split_row(
                    kv_ptr,
                    loc_ptr,
                    nope_ptr,
                    rope_ptr,
                    row,
                    stride_kv,
                    stride_loc,
                    stride_n,
                    stride_r,
                    NOPE_D,
                    ROPE_D,
                    BLOCK_N,
                    BLOCK_R,
                )


@triton.jit
def _row_slot(loc_ptr, row, N, stride_loc):
    # Tail-safe row source: clamped index reads the last valid slot and
    # the matching clamped store re-writes row N-1 with the same value,
    # so partial batches need no masks or branches (deterministic:
    # duplicates come from the same program, sequentially).
    c = tl.minimum(row, N - 1)
    s = tl.load(loc_ptr + c * stride_loc).to(tl.int64)
    return c, s


@triton.jit
def _load_nope(
    kv_ptr, s, stride_kv, d_n, NOPE_D: tl.constexpr, BLOCK_N: tl.constexpr
):
    if BLOCK_N == NOPE_D:
        return tl.load(kv_ptr + s * stride_kv + d_n)
    return tl.load(kv_ptr + s * stride_kv + d_n, mask=d_n < NOPE_D, other=0)


@triton.jit
def _load_rope(
    kv_ptr,
    s,
    stride_kv,
    d_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    if BLOCK_R == ROPE_D:
        return tl.load(kv_ptr + (s * stride_kv + NOPE_D) + d_r)
    return tl.load(
        kv_ptr + (s * stride_kv + NOPE_D) + d_r, mask=d_r < ROPE_D, other=0
    )


@triton.jit
def _store_nope(
    nope_ptr, c, v, stride_n, d_n, NOPE_D: tl.constexpr, BLOCK_N: tl.constexpr
):
    if BLOCK_N == NOPE_D:
        tl.store(
            nope_ptr + c * stride_n + d_n, v.to(nope_ptr.dtype.element_ty)
        )
    else:
        tl.store(
            nope_ptr + c * stride_n + d_n,
            v.to(nope_ptr.dtype.element_ty),
            mask=d_n < NOPE_D,
        )


@triton.jit
def _store_rope(
    rope_ptr, c, v, stride_r, d_r, ROPE_D: tl.constexpr, BLOCK_R: tl.constexpr
):
    if BLOCK_R == ROPE_D:
        tl.store(
            rope_ptr + c * stride_r + d_r, v.to(rope_ptr.dtype.element_ty)
        )
    else:
        tl.store(
            rope_ptr + c * stride_r + d_r,
            v.to(rope_ptr.dtype.element_ty),
            mask=d_r < ROPE_D,
        )


@triton.jit
def _gather_split_rows_u1(
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
):
    base = tl.program_id(0) * 1
    d_n = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
    )
    d_r = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R
    )
    c0, s0 = _row_slot(loc_ptr, base + 0, N, stride_loc)
    v0n = _load_nope(kv_ptr, s0, stride_kv, d_n, NOPE_D, BLOCK_N)
    v0r = _load_rope(kv_ptr, s0, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_nope(nope_ptr, c0, v0n, stride_n, d_n, NOPE_D, BLOCK_N)
    _store_rope(rope_ptr, c0, v0r, stride_r, d_r, ROPE_D, BLOCK_R)


@triton.jit
def _gather_split_rows_u8ps(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    N_BLOCKS,
    N,
    stride_kv,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # E-1 third round (2026-11-13): 8-row batches with the nope/rope phases
    # split (nope loads -> nope stores -> rope loads -> rope stores), launched
    # at num_warps=2, beat the previous 16-row three-segment form by ~1.9x on
    # the L600 sweep (13.4 vs 25.4 us at n=4096, in-process do_bench): nw2
    # halves the per-batch issue width so a whole 48-program wave stays
    # resident, and the 8-row phase split keeps the peak staged lanes at
    # 8 * BLOCK_N (~102 KB DSM at BLOCK_N=512). The grid stays one wave
    # (48 = #SPs): g96/g144/g192, plain-block grids, nw1/nw3/nw4 and the
    # u4/u16/u32 batchings all measured worse. Same three-segment batching
    # discipline (slot scalars first) and min-clamp tails as the historical
    # form.
    d_n = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
    )
    d_r = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R
    )
    for blk in range(tl.program_id(0), N_BLOCKS, tl.num_programs(0)):
        base = blk * 8
        c0, s0 = _row_slot(loc_ptr, base + 0, N, stride_loc)
        c1, s1 = _row_slot(loc_ptr, base + 1, N, stride_loc)
        c2, s2 = _row_slot(loc_ptr, base + 2, N, stride_loc)
        c3, s3 = _row_slot(loc_ptr, base + 3, N, stride_loc)
        c4, s4 = _row_slot(loc_ptr, base + 4, N, stride_loc)
        c5, s5 = _row_slot(loc_ptr, base + 5, N, stride_loc)
        c6, s6 = _row_slot(loc_ptr, base + 6, N, stride_loc)
        c7, s7 = _row_slot(loc_ptr, base + 7, N, stride_loc)
        v0n = _load_nope(kv_ptr, s0, stride_kv, d_n, NOPE_D, BLOCK_N)
        v1n = _load_nope(kv_ptr, s1, stride_kv, d_n, NOPE_D, BLOCK_N)
        v2n = _load_nope(kv_ptr, s2, stride_kv, d_n, NOPE_D, BLOCK_N)
        v3n = _load_nope(kv_ptr, s3, stride_kv, d_n, NOPE_D, BLOCK_N)
        v4n = _load_nope(kv_ptr, s4, stride_kv, d_n, NOPE_D, BLOCK_N)
        v5n = _load_nope(kv_ptr, s5, stride_kv, d_n, NOPE_D, BLOCK_N)
        v6n = _load_nope(kv_ptr, s6, stride_kv, d_n, NOPE_D, BLOCK_N)
        v7n = _load_nope(kv_ptr, s7, stride_kv, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c0, v0n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c1, v1n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c2, v2n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c3, v3n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c4, v4n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c5, v5n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c6, v6n, stride_n, d_n, NOPE_D, BLOCK_N)
        _store_nope(nope_ptr, c7, v7n, stride_n, d_n, NOPE_D, BLOCK_N)
        v0r = _load_rope(kv_ptr, s0, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v1r = _load_rope(kv_ptr, s1, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v2r = _load_rope(kv_ptr, s2, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v3r = _load_rope(kv_ptr, s3, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v4r = _load_rope(kv_ptr, s4, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v5r = _load_rope(kv_ptr, s5, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v6r = _load_rope(kv_ptr, s6, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        v7r = _load_rope(kv_ptr, s7, stride_kv, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c0, v0r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c1, v1r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c2, v2r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c3, v3r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c4, v4r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c5, v5r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c6, v6r, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_rope(rope_ptr, c7, v7r, stride_r, d_r, ROPE_D, BLOCK_R)


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    n_loc = loc.shape[0]

    nope = torch.empty_like(cache_k_nope)
    rope = torch.empty_like(cache_k_rope)
    if n_loc == 0:
        return nope, rope

    # Zero-copy int32 word view of int64 ``loc`` (same scheme as
    # set_mla_kv_buffer): torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64
    # keeps device int64 values as compact int32 words (element j at word j),
    # while with the gate on they sit in the low word of each 8-byte slot
    # (word 2*j). Either way a Triton load at the declared int64 stride
    # reads neighbor-joined garbage (see the kernel comment below), so slots
    # are read through ``loc.view(torch.int32)`` with the matching word
    # stride. Slot ids stay far below 2**31, so the low word is value-exact.
    # This drops the per-call ATen int64->int32 cast kernel that the env
    # gate turns into a real 8-byte copy (measured +1.3~2.8us per bench
    # call on set_mla_kv_buffer; +4.8us here at n=4096).
    loc_word_stride = loc.stride(0)
    if loc.dtype == torch.int64:
        loc = loc.contiguous().view(torch.int32)
        loc_word_stride = (
            2
            if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1"
            else 1
        )

    block_n = triton.next_power_of_2(nope_dim)
    block_r = triton.next_power_of_2(rope_dim)
    shapes = dict(
        NOPE_D=nope_dim, ROPE_D=rope_dim, BLOCK_N=block_n, BLOCK_R=block_r
    )
    tail = (
        n_loc,
        kv_buffer.stride(0),
        loc_word_stride,
        nope.stride(0),
        rope.stride(0),
    )

    if 16 * (block_n + block_r) > _FAST_PATH_MAX_LANES:
        # Wide rows: unchanged sealed fallback (per-row autotuned form);
        # the 16-row lane budget is kept conservative even though the fast
        # kernel now batches 8 rows, so wide-row shapes keep their proven
        # path and only the measured standard band switches to u8ps.
        max_blocks = triton.cdiv(n_loc, 2)
        if max_blocks > _GRID_X_MAX:
            n_blocks = triton.cdiv(n_loc, _ROWS_MANY)
            _gather_split_rows_kernel_many[(_GRID_X_MAX,)](
                kv_buffer,
                loc,
                nope,
                rope,
                n_blocks,
                *tail,
                ROWS=_ROWS_MANY,
                num_warps=4,
                **shapes,
            )
        else:

            def grid(meta):
                return (triton.cdiv(n_loc, meta["ROWS"]),)

            _gather_split_rows_kernel[grid](
                kv_buffer, loc, nope, rope, *tail, **shapes
            )
        return nope, rope

    # Explicitly unrolled batches (slot scalars -> row loads -> stores):
    # the GCU compiler does not reorder interleaved per-row load/store
    # chains, so the batching must be written out. n=1 keeps the tiny
    # dedicated kernel; everything else runs the 8-row phase-split
    # persistent form at one 48-program wave, num_warps=2 (E-1 third
    # round sweep: covers the old u4/u16/persistent bands, 13.4us at
    # n=4096 vs 25.4us for the previous 16-row nw4 form).
    if n_loc == 1:
        _gather_split_rows_u1[(n_loc,)](
            kv_buffer, loc, nope, rope, *tail, num_warps=4, **shapes
        )
        return nope, rope

    blocks8 = triton.cdiv(n_loc, 8)
    grid = min(_PERSISTENT_GRID, blocks8, _GRID_X_MAX)
    _gather_split_rows_u8ps[(grid,)](
        kv_buffer, loc, nope, rope, blocks8, *tail, num_warps=2, **shapes
    )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
