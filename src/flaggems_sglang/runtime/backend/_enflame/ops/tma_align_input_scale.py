# Copyright 2026 FlagOS Contributors
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

# GCU rewrite (2026-11-08, job j20261108170555_4bd809fe): the strided
# (non-contiguous) kernel path ran at ~345 GB/s effective (~12% of the 2D
# roofline) with the generic BLOCK_M=64/BLOCK_K=64/num_warps=4 tile. Measured
# levers (do_bench, full op path, GCU410/24SP): BLOCK_M 64->512 is dominant
# (m=16384: 25.14 -> 8.10us, -68%; m=8192: -> 6.50us, -52%; m=4096: 9.60 ->
# 5.79us, -40%), plus int32 index math (host-guarded i32-safe span) and a
# single fused store (pad lanes are already 0 via the masked load's other=0,
# so the separate zero store is folded into the value store's mask
# mm < M_PAD). Large-m throughput reaches ~2.1 TB/s, at the 2D-tile roofline.
# m <= ~512 is kernel-start-latency bound (~2.5us) and does not improve.
# num_warps=2 beats 1/4 at BLOCK_M=512; a fixed-grid persistent loop was
# slower than the plain tile grid at this tile size (loop overhead exceeds
# the wave-tail saving) and is not used.
#
# E-1 second round (2026-11-12, job j20261112105756_478bdc62): the round-1
# config shared BLOCK_M=512 across every stride class and m, which regressed
# on inputs with no unit stride on either axis (e.g. base[:, ::2] rowstr2 /
# [::3, ::2] weird views): enflame ran at 0.38-0.88x of the generic path
# there (rowstr2 m=512: 15.85 vs 6.57us; m=16384: 56.5 vs 74.7us -- generic
# was FASTER below m~3072, enflame above). Two dispatch arms fix this
# (do_bench, 3-draw min, data-cold per rep):
#   - no-unit-stride inputs (ss0 != 1 and ss1 != 1): BLOCK_M=64/nw4 for
#     m_pad <= 3072 (rowstr2 m=512: 15.85 -> 6.47us, 2.45x; m=2048:
#     16.10 -> 11.71us, 1.37x; m=3072: ~17.5 -> 15.21us), keep BLOCK_M=512
#     above (rowstr2 m=16384: 56.5us stays 1.31x faster than generic).
#   - unit-stride-on-an-axis inputs: BLOCK_M=128/nw2 for m_pad <= 2048
#     (rowslice m=2048: 5.13 -> 4.75us, -7.4%, > 2x the fresh-process noise
#     band; m <= 1024 is already launch-floor bound), keep BLOCK_M=512 above.
# BLOCK_K=64 stays optimal in both classes (BK=32 mid-m gains evaporate
# across fresh processes; BK=128 is worse everywhere). A tl.split pair-load
# kernel for ss1==2 (read contiguous (BK,BM,2) pairs, split off odd lanes)
# was measured at ~195us -- 13x WORSE than the strided form -- and is vetoed
# (3D-tile DSM materialization pathology; see perf_playbook).
#
# Compiler-bug note: the enflame backend's slice_transpose_start lowering
# (gcu-compiler-opt attribute-conversion mismatch, slice_start family) is
# sensitive to source-level codegen differences; this kernel keeps a minimal
# single-purpose body (no dead branches, no unused args, no cast-in-if) and
# is dense-gated over the m x k x stride matrix -- see
# compiler_issues/sglang_tma_align_input_scale.md.
#
# The i64 original kernel below is kept verbatim as the fallback for tensors
# whose index spans exceed int32 (>= 2^31 elements); that regime is far
# outside every measured shape and keeps the battle-tested codegen form.
#
# Contiguous inputs (the benchmark and accuracy-test path) still return
# unchanged via the is_contiguous() gate: zero device work, measured at the
# harness event floor (~2.1us) in both the generic and this copy.

import torch
import triton
import triton.language as tl

_FLAT_GRID_CAP = 65535

_BLOCK_M = 512
_BLOCK_K = 64
_NUM_WARPS = 2

# small-m dispatch arms (E-1 round 2): m_pad caps below which a smaller tile
# wins -- unit-stride inputs switch at 2048, no-unit-stride at 3072
_SMALL_M_CAP = 2048
_SMALL_M = 128
_SMALL_NUM_WARPS = 2
_EXOTIC_SMALL_M_CAP = 3072
_EXOTIC_SMALL_M = 64
_EXOTIC_SMALL_NUM_WARPS = 4


@triton.jit
def _transpose_pad_i32_kernel(
    src,
    dst,
    M,
    K,
    M_PAD,
    ss0,
    ss1,
    ds0,
    m_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # int32 index math; the host wrapper guarantees the source and
    # destination BYTE spans stay under 2**30 (verified-correct backend
    # domain), so every element offset below fits int32 with margin.
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    mm = (m_offset + pm) * BLOCK_M + tl.arange(0, BLOCK_M)
    kk = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_in = kk < K
    # value layout (BLOCK_K, BLOCK_M): rows are k, fastest axis is m.
    # Single store: lanes in [M, M_PAD) load as 0 (masked, other=0.0) and
    # are stored, which zero-fills the pad stripe; lanes >= M_PAD are
    # masked out of the store.
    v = tl.load(
        src + kk[:, None] * ss1 + mm[None, :] * ss0,
        mask=k_in[:, None] & (mm < M)[None, :],
        other=0.0,
    )
    tl.store(
        dst + kk[:, None] * ds0 + mm[None, :],
        v,
        mask=k_in[:, None] & (mm < M_PAD)[None, :],
    )


@triton.jit
def _transpose_pad_kernel(
    src,
    dst,
    M,
    K,
    M_PAD,
    ss0,
    ss1,
    ds0,
    m_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # original int64 form, kept as the i32-unsafe (>= 2^31 span) fallback
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    mm = ((m_offset + pm) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    kk = (pk * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    k_in = kk < K
    inside = mm < M
    pad = (mm >= M) & (mm < M_PAD)
    load_mask = k_in[:, None] & inside[None, :]
    v = tl.load(
        src + kk[:, None] * ss1 + mm[None, :] * ss0, mask=load_mask, other=0.0
    )
    tl.store(dst + kk[:, None] * ds0 + mm[None, :], v, mask=load_mask)
    z = tl.zeros((BLOCK_K, BLOCK_M), dtype=v.dtype)
    tl.store(
        dst + kk[:, None] * ds0 + mm[None, :],
        z,
        mask=k_in[:, None] & pad[None, :],
    )


def tma_align_input_scale(input_scale):
    # hot path: one C-level is_contiguous() call; every contiguous tensor
    # returns unchanged (the task reference is `return input_scale`, Exact
    # value equality), only genuinely strided layouts pay the kernel below
    if input_scale.is_contiguous():
        return input_scale
    return _tma_align_kernel_path(input_scale)


def _tma_align_kernel_path(input_scale):
    assert input_scale.dim() == 2
    m, k = input_scale.shape
    elem = input_scale.element_size()
    align = max(1, 16 // elem)
    m_pad = (m + align - 1) // align * align
    if m and k and input_scale.stride(0) == k and input_scale.stride(1) == 1:
        return input_scale
    buf = torch.empty(
        (k, m_pad), dtype=input_scale.dtype, device=input_scale.device
    )
    if m and k:
        ss0 = input_scale.stride(0)
        ss1 = input_scale.stride(1)
        elem = input_scale.element_size()
        # i32 fast path guard, on BYTE spans: the enflame backend's offset
        # arithmetic stays verified-correct up to ~2**30 bytes (probed:
        # 1 GiB strided span exact-correct, 4 GiB device-aborts, and the
        # int64 kernel form silently mis-reads at >= 8 GiB -- the same
        # pre-existing generic behavior, see
        # compiler_issues/sglang_tma_align_input_scale.md). Everything
        # outside the verified domain falls back to the original int64
        # kernel, byte-for-byte generic behavior.
        src_span = ((m - 1) * ss0 + (k - 1) * ss1 + 1) * elem
        if src_span < 2**30 and k * m_pad * elem < 2**30:
            # GCU fast path: wide m tiles, int32 indexing, fused pad store.
            # Tile dispatch (E-1 round 2): a 128KB gather tile is
            # pathological when NEITHER axis has unit stride (the load
            # degrades to element-wise 4B gathers), so those inputs use
            # the small 64-row tile below ~3072 rows; unit-stride inputs
            # use the 128-row tile below ~2048 rows (mid-m occupancy);
            # everything else keeps the round-1 512-row tile.
            if ss0 == 1 or ss1 == 1:
                if m_pad <= _SMALL_M_CAP:
                    block_m, num_warps = _SMALL_M, _SMALL_NUM_WARPS
                else:
                    block_m, num_warps = _BLOCK_M, _NUM_WARPS
            else:
                if m_pad <= _EXOTIC_SMALL_M_CAP:
                    block_m, num_warps = (
                        _EXOTIC_SMALL_M,
                        _EXOTIC_SMALL_NUM_WARPS,
                    )
                else:
                    block_m, num_warps = _BLOCK_M, _NUM_WARPS
            grid_k = triton.cdiv(k, _BLOCK_K)
            tiles_per_launch = max(1, _FLAT_GRID_CAP // grid_k)
            m0 = 0
            while m0 < m_pad:
                tiles = min(tiles_per_launch, triton.cdiv(m_pad - m0, block_m))
                _transpose_pad_i32_kernel[(tiles, grid_k)](
                    input_scale,
                    buf,
                    m,
                    k,
                    m_pad,
                    ss0,
                    ss1,
                    buf.stride(0),
                    m0 // block_m,
                    BLOCK_M=block_m,
                    BLOCK_K=_BLOCK_K,
                    num_warps=num_warps,
                )
                m0 += tiles * block_m
        else:
            # i32-unsafe span: original int64 kernel, original tile shape
            grid_k = triton.cdiv(k, 64)
            tiles_per_launch = max(1, _FLAT_GRID_CAP // grid_k)
            m0 = 0
            while m0 < m_pad:
                tiles = min(tiles_per_launch, triton.cdiv(m_pad - m0, 64))
                _transpose_pad_kernel[(tiles, grid_k)](
                    input_scale,
                    buf,
                    m,
                    k,
                    m_pad,
                    ss0,
                    ss1,
                    buf.stride(0),
                    m0 // 64,
                    BLOCK_M=64,
                    BLOCK_K=64,
                    num_warps=4,
                )
                m0 += tiles * 64
    return buf.t()[:m]


__all__ = ["tma_align_input_scale"]
