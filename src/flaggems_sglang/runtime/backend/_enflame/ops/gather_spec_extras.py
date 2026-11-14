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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCU adaptation of speculative/gather_spec_extras.

torch_gcu stores ATen int64 tensors as compact int32 words (element j
lives at word j; with TORCH_GCU_ENABLE_INT64_AND_UINT64=1 it lives in
the low word of each 8-byte slot) while the tensor metadata still
reports 8-byte elements. The generic kernel loads indices /
topk_index_buf / output_tokens_buf through int64 pointers at the
8-byte dtype stride, reading neighbor-joined garbage row ids and
storing int64 outputs past the compact slots - the wild gather aborts
the device (baseline ABORT rc=-6 in both accuracy and bench runs).

Here every int64 tensor is handed to the kernel as a zero-copy int32
word view with the matching word stride, so all loads/stores stay in
the int32 word domain (same pattern as the enflame
fill_accept_out_cache_loc / indexer_k_quant_and_cache fixes). Row ids
and token ids fit the signed int32 safe domain.

The launch shape is also reworked for GCU. The generic (m, h_chunks)
grid dispatches m*cdiv(hidden, 1024) programs and is dispatch-bound
already at m=512 (3584 programs); the row copy itself is a pure
gather of contiguous 14KB rows, so the structure of choice is a
persistent grid-stride loop capped at _MAX_PROGRAMS with one scalar
row-id load and one wide (masked when needed) vector copy per row.

Two dispatch arms (kernel-direct do_bench, hidden=7168 bf16, median):
- m <= _SMALL_M_MAX: one row per program iteration (per-row grid
  min(m, 48)); small m is launch-floor bound and per-row programs
  keep all SIPs busy (m=8: 4.7us vs 7.2us for the batch kernel).
- m > _SMALL_M_MAX: 4-row groups with all loads issued before the
  stores (assign_draft_cache_locs_contiguous batch pattern); each
  group then costs a single dependent memory latency for four rows
  (m=512: 20.8 -> 12.7us, m=2048: 68.6 -> 31.9us).

The host path is also latency-critical. ``triton.testing.do_bench``
(kernel mode) brackets each call with event records after a 256MB
L2-clear kernel, so measured latency is the device time plus any
host time that overflows that clear window; on a CPU-contended box
the per-call Python (five word views, four output allocations, the
JITFunction binder) can exceed the window and inflate the small-m
readings by 5-10us with the device side unchanged. The wrapper
therefore keeps the per-call Python minimal: the int64 word stride
is read from the environment once per call, the launch geometry is
memoized per (topk, hidden), and repeat launches with an identical
signature bypass the JITFunction dispatch through a cached
CompiledKernel handle (the cache key covers every binder
specialization axis: argument values, pointer alignments, dtypes,
grid and constexprs; profiling hooks, the flagtree lite-distributed
mode and any fast-path error fall back to the standard launch).
The hook gate treats an empty triton >= 3.4 HookChain as "no hook":
knobs.runtime.launch_*_hook is a non-None HookChain object on this
stack even when nothing is registered, so a plain is-None check would
silently disable the fast path and pay the full JITFunction dispatch
(~46us/call measured) on every launch.
"""

import os

import torch
import triton
import triton.language as tl

try:
    import torch_gcu

    _raw_stream_getter = torch_gcu._C._gcu_getCurrentRawStream
except Exception:  # pragma: no cover - defensive, GCU builds always have it
    torch_gcu = None
    _raw_stream_getter = None

try:
    from triton import knobs as _triton_knobs
except Exception:  # pragma: no cover - older triton without knobs
    _triton_knobs = None


def _int64_word_stride():
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen
    # int64 values as compact int32 words (element j lives at word j);
    # with the gate on, values sit in the low word of each 8-byte slot
    # (word 2*j). torch_gcu only honors the "1"/"true" spellings
    # (calibrated in compiler_issues/triton_i64_narrow_load.md).
    enabled = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    return 2 if enabled in ("1", "true") else 1


def _input_int32_words(tensor, word_stride):
    if tensor.dtype == torch.int32:
        return (tensor if tensor.is_contiguous() else tensor.contiguous()), 1
    if tensor.dtype == torch.int64:
        if tensor.is_contiguous():
            return tensor.view(torch.int32), word_stride
        return tensor.contiguous().view(torch.int32), word_stride
    raise NotImplementedError(
        "gather_spec_extras supports int32/int64 index tensors, got "
        f"{tensor.dtype}"
    )


def _output_int32_words(tensor, word_stride):
    # Outputs are freshly allocated contiguous tensors; the int32 view
    # only retypes the pointer, the kernel addresses the flat word
    # domain itself (word 2*e / 2*e+1 for the env-gated 8-byte layout).
    if tensor.dtype == torch.int32:
        return tensor, 1
    return tensor.view(torch.int32), word_stride


@triton.jit
def _gather_spec_extras_kernel(
    idx_ptr,
    p_buf,
    i_buf,
    t_buf,
    h_buf,
    p_out,
    i_out,
    t_out,
    h_out,
    m,
    topk,
    hidden,
    stride_p,
    stride_h,
    IDX_STRIDE: tl.constexpr,
    I_STRIDE: tl.constexpr,
    T_STRIDE: tl.constexpr,
    OI_STRIDE: tl.constexpr,
    OT_STRIDE: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    EVEN_TK: tl.constexpr,
    EVEN_H: tl.constexpr,
    LOOP_H: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs_t = tl.arange(0, BLOCK_TK)

    for r in range(pid, m, nprog):
        src = tl.load(idx_ptr + r * IDX_STRIDE)
        b = src.to(tl.int64)
        o = r.to(tl.int64)

        if EVEN_TK:
            pv = tl.load(p_buf + b * stride_p + offs_t)
            iv = tl.load(i_buf + b * (topk * I_STRIDE) + offs_t * I_STRIDE)
            tv = tl.load(t_buf + b * T_STRIDE)
            tl.store(p_out + o * topk + offs_t, pv)
            if OI_STRIDE == 1:
                tl.store(i_out + o * topk + offs_t, iv)
            else:
                tl.store(i_out + o * (topk * 2) + offs_t * 2, iv)
                tl.store(i_out + o * (topk * 2) + offs_t * 2 + 1, iv >> 31)
            if OT_STRIDE == 1:
                tl.store(t_out + o, tv)
            else:
                tl.store(t_out + o * 2, tv)
                tl.store(t_out + o * 2 + 1, tv >> 31)
        else:
            tmask = offs_t < topk
            pv = tl.load(p_buf + b * stride_p + offs_t, mask=tmask)
            iv = tl.load(
                i_buf + b * (topk * I_STRIDE) + offs_t * I_STRIDE, mask=tmask
            )
            tv = tl.load(t_buf + b * T_STRIDE)
            tl.store(p_out + o * topk + offs_t, pv, mask=tmask)
            if OI_STRIDE == 1:
                tl.store(i_out + o * topk + offs_t, iv, mask=tmask)
            else:
                tl.store(i_out + o * (topk * 2) + offs_t * 2, iv, mask=tmask)
                tl.store(
                    i_out + o * (topk * 2) + offs_t * 2 + 1,
                    iv >> 31,
                    mask=tmask,
                )
            if OT_STRIDE == 1:
                tl.store(t_out + o, tv)
            else:
                tl.store(t_out + o * 2, tv)
                tl.store(t_out + o * 2 + 1, tv >> 31)

        if HAS_HIDDEN:
            if LOOP_H:
                for c0 in range(0, hidden, BLOCK_H):
                    cols = c0 + tl.arange(0, BLOCK_H)
                    cmask = cols < hidden
                    hv = tl.load(
                        h_buf + b * stride_h + cols, mask=cmask, other=0
                    )
                    tl.store(h_out + o * hidden + cols, hv, mask=cmask)
            else:
                for c in tl.static_range(NB):
                    cols = c * BLOCK_H + tl.arange(0, BLOCK_H)
                    if EVEN_H:
                        hv = tl.load(h_buf + b * stride_h + cols)
                        tl.store(h_out + o * hidden + cols, hv)
                    else:
                        cmask = cols < hidden
                        hv = tl.load(
                            h_buf + b * stride_h + cols, mask=cmask, other=0
                        )
                        tl.store(h_out + o * hidden + cols, hv, mask=cmask)


@triton.jit
def _gather_spec_extras_batch_kernel(
    idx_ptr,
    p_buf,
    i_buf,
    t_buf,
    h_buf,
    p_out,
    i_out,
    t_out,
    h_out,
    m,
    topk,
    hidden,
    stride_p,
    stride_h,
    IDX_STRIDE: tl.constexpr,
    I_STRIDE: tl.constexpr,
    T_STRIDE: tl.constexpr,
    OI_STRIDE: tl.constexpr,
    OT_STRIDE: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    EVEN_TK: tl.constexpr,
    EVEN_H: tl.constexpr,
    LOOP_H: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs_t = tl.arange(0, BLOCK_TK)

    for r0 in range(pid * 4, m, nprog * 4):
        # Tail groups clamp the row id for loads and mask the stores;
        # only the single last group of the grid can be partial.
        r1 = tl.minimum(r0 + 1, m - 1)
        r2 = tl.minimum(r0 + 2, m - 1)
        r3 = tl.minimum(r0 + 3, m - 1)
        v1 = r0 + 1 < m
        v2 = r0 + 2 < m
        v3 = r0 + 3 < m
        s0 = tl.load(idx_ptr + r0 * IDX_STRIDE)
        s1 = tl.load(idx_ptr + r1 * IDX_STRIDE)
        s2 = tl.load(idx_ptr + r2 * IDX_STRIDE)
        s3 = tl.load(idx_ptr + r3 * IDX_STRIDE)
        b0 = s0.to(tl.int64)
        b1 = s1.to(tl.int64)
        b2 = s2.to(tl.int64)
        b3 = s3.to(tl.int64)
        o0 = r0.to(tl.int64)
        o1 = r1.to(tl.int64)
        o2 = r2.to(tl.int64)
        o3 = r3.to(tl.int64)

        if EVEN_TK:
            pv0 = tl.load(p_buf + b0 * stride_p + offs_t)
            pv1 = tl.load(p_buf + b1 * stride_p + offs_t)
            pv2 = tl.load(p_buf + b2 * stride_p + offs_t)
            pv3 = tl.load(p_buf + b3 * stride_p + offs_t)
            iv0 = tl.load(i_buf + b0 * (topk * I_STRIDE) + offs_t * I_STRIDE)
            iv1 = tl.load(i_buf + b1 * (topk * I_STRIDE) + offs_t * I_STRIDE)
            iv2 = tl.load(i_buf + b2 * (topk * I_STRIDE) + offs_t * I_STRIDE)
            iv3 = tl.load(i_buf + b3 * (topk * I_STRIDE) + offs_t * I_STRIDE)
            tv0 = tl.load(t_buf + b0 * T_STRIDE)
            tv1 = tl.load(t_buf + b1 * T_STRIDE)
            tv2 = tl.load(t_buf + b2 * T_STRIDE)
            tv3 = tl.load(t_buf + b3 * T_STRIDE)
            tl.store(p_out + o0 * topk + offs_t, pv0)
            tl.store(p_out + o1 * topk + offs_t, pv1, mask=(offs_t >= 0) & v1)
            tl.store(p_out + o2 * topk + offs_t, pv2, mask=(offs_t >= 0) & v2)
            tl.store(p_out + o3 * topk + offs_t, pv3, mask=(offs_t >= 0) & v3)
            if OI_STRIDE == 1:
                tl.store(i_out + o0 * topk + offs_t, iv0)
                tl.store(
                    i_out + o1 * topk + offs_t, iv1, mask=(offs_t >= 0) & v1
                )
                tl.store(
                    i_out + o2 * topk + offs_t, iv2, mask=(offs_t >= 0) & v2
                )
                tl.store(
                    i_out + o3 * topk + offs_t, iv3, mask=(offs_t >= 0) & v3
                )
            else:
                tl.store(i_out + o0 * (topk * 2) + offs_t * 2, iv0)
                tl.store(
                    i_out + o1 * (topk * 2) + offs_t * 2,
                    iv1,
                    mask=(offs_t >= 0) & v1,
                )
                tl.store(
                    i_out + o2 * (topk * 2) + offs_t * 2,
                    iv2,
                    mask=(offs_t >= 0) & v2,
                )
                tl.store(
                    i_out + o3 * (topk * 2) + offs_t * 2,
                    iv3,
                    mask=(offs_t >= 0) & v3,
                )
                tl.store(i_out + o0 * (topk * 2) + offs_t * 2 + 1, iv0 >> 31)
                tl.store(
                    i_out + o1 * (topk * 2) + offs_t * 2 + 1,
                    iv1 >> 31,
                    mask=(offs_t >= 0) & v1,
                )
                tl.store(
                    i_out + o2 * (topk * 2) + offs_t * 2 + 1,
                    iv2 >> 31,
                    mask=(offs_t >= 0) & v2,
                )
                tl.store(
                    i_out + o3 * (topk * 2) + offs_t * 2 + 1,
                    iv3 >> 31,
                    mask=(offs_t >= 0) & v3,
                )
            if OT_STRIDE == 1:
                tl.store(t_out + o0, tv0)
                tl.store(t_out + o1, tv1, mask=v1)
                tl.store(t_out + o2, tv2, mask=v2)
                tl.store(t_out + o3, tv3, mask=v3)
            else:
                tl.store(t_out + o0 * 2, tv0)
                tl.store(t_out + o1 * 2, tv1, mask=v1)
                tl.store(t_out + o2 * 2, tv2, mask=v2)
                tl.store(t_out + o3 * 2, tv3, mask=v3)
                tl.store(t_out + o0 * 2 + 1, tv0 >> 31)
                tl.store(t_out + o1 * 2 + 1, tv1 >> 31, mask=v1)
                tl.store(t_out + o2 * 2 + 1, tv2 >> 31, mask=v2)
                tl.store(t_out + o3 * 2 + 1, tv3 >> 31, mask=v3)
        else:
            tmask = offs_t < topk
            m1 = tmask & v1
            m2 = tmask & v2
            m3 = tmask & v3
            pv0 = tl.load(p_buf + b0 * stride_p + offs_t, mask=tmask)
            pv1 = tl.load(p_buf + b1 * stride_p + offs_t, mask=tmask)
            pv2 = tl.load(p_buf + b2 * stride_p + offs_t, mask=tmask)
            pv3 = tl.load(p_buf + b3 * stride_p + offs_t, mask=tmask)
            iv0 = tl.load(
                i_buf + b0 * (topk * I_STRIDE) + offs_t * I_STRIDE, mask=tmask
            )
            iv1 = tl.load(
                i_buf + b1 * (topk * I_STRIDE) + offs_t * I_STRIDE, mask=tmask
            )
            iv2 = tl.load(
                i_buf + b2 * (topk * I_STRIDE) + offs_t * I_STRIDE, mask=tmask
            )
            iv3 = tl.load(
                i_buf + b3 * (topk * I_STRIDE) + offs_t * I_STRIDE, mask=tmask
            )
            tv0 = tl.load(t_buf + b0 * T_STRIDE)
            tv1 = tl.load(t_buf + b1 * T_STRIDE)
            tv2 = tl.load(t_buf + b2 * T_STRIDE)
            tv3 = tl.load(t_buf + b3 * T_STRIDE)
            tl.store(p_out + o0 * topk + offs_t, pv0, mask=tmask)
            tl.store(p_out + o1 * topk + offs_t, pv1, mask=m1)
            tl.store(p_out + o2 * topk + offs_t, pv2, mask=m2)
            tl.store(p_out + o3 * topk + offs_t, pv3, mask=m3)
            if OI_STRIDE == 1:
                tl.store(i_out + o0 * topk + offs_t, iv0, mask=tmask)
                tl.store(i_out + o1 * topk + offs_t, iv1, mask=m1)
                tl.store(i_out + o2 * topk + offs_t, iv2, mask=m2)
                tl.store(i_out + o3 * topk + offs_t, iv3, mask=m3)
            else:
                tl.store(i_out + o0 * (topk * 2) + offs_t * 2, iv0, mask=tmask)
                tl.store(i_out + o1 * (topk * 2) + offs_t * 2, iv1, mask=m1)
                tl.store(i_out + o2 * (topk * 2) + offs_t * 2, iv2, mask=m2)
                tl.store(i_out + o3 * (topk * 2) + offs_t * 2, iv3, mask=m3)
                tl.store(
                    i_out + o0 * (topk * 2) + offs_t * 2 + 1,
                    iv0 >> 31,
                    mask=tmask,
                )
                tl.store(
                    i_out + o1 * (topk * 2) + offs_t * 2 + 1,
                    iv1 >> 31,
                    mask=m1,
                )
                tl.store(
                    i_out + o2 * (topk * 2) + offs_t * 2 + 1,
                    iv2 >> 31,
                    mask=m2,
                )
                tl.store(
                    i_out + o3 * (topk * 2) + offs_t * 2 + 1,
                    iv3 >> 31,
                    mask=m3,
                )
            if OT_STRIDE == 1:
                tl.store(t_out + o0, tv0)
                tl.store(t_out + o1, tv1, mask=v1)
                tl.store(t_out + o2, tv2, mask=v2)
                tl.store(t_out + o3, tv3, mask=v3)
            else:
                tl.store(t_out + o0 * 2, tv0)
                tl.store(t_out + o1 * 2, tv1, mask=v1)
                tl.store(t_out + o2 * 2, tv2, mask=v2)
                tl.store(t_out + o3 * 2, tv3, mask=v3)
                tl.store(t_out + o0 * 2 + 1, tv0 >> 31)
                tl.store(t_out + o1 * 2 + 1, tv1 >> 31, mask=v1)
                tl.store(t_out + o2 * 2 + 1, tv2 >> 31, mask=v2)
                tl.store(t_out + o3 * 2 + 1, tv3 >> 31, mask=v3)

        if HAS_HIDDEN:
            if LOOP_H:
                for c0 in range(0, hidden, BLOCK_H):
                    cols = c0 + tl.arange(0, BLOCK_H)
                    cmask = cols < hidden
                    g0 = tl.load(
                        h_buf + b0 * stride_h + cols, mask=cmask, other=0
                    )
                    g1 = tl.load(
                        h_buf + b1 * stride_h + cols, mask=cmask, other=0
                    )
                    g2 = tl.load(
                        h_buf + b2 * stride_h + cols, mask=cmask, other=0
                    )
                    g3 = tl.load(
                        h_buf + b3 * stride_h + cols, mask=cmask, other=0
                    )
                    tl.store(h_out + o0 * hidden + cols, g0, mask=cmask)
                    tl.store(h_out + o1 * hidden + cols, g1, mask=cmask & v1)
                    tl.store(h_out + o2 * hidden + cols, g2, mask=cmask & v2)
                    tl.store(h_out + o3 * hidden + cols, g3, mask=cmask & v3)
            else:
                for c in tl.static_range(NB):
                    cols = c * BLOCK_H + tl.arange(0, BLOCK_H)
                    if EVEN_H:
                        g0 = tl.load(h_buf + b0 * stride_h + cols)
                        g1 = tl.load(h_buf + b1 * stride_h + cols)
                        g2 = tl.load(h_buf + b2 * stride_h + cols)
                        g3 = tl.load(h_buf + b3 * stride_h + cols)
                        tl.store(h_out + o0 * hidden + cols, g0)
                        tl.store(
                            h_out + o1 * hidden + cols,
                            g1,
                            mask=(cols >= 0) & v1,
                        )
                        tl.store(
                            h_out + o2 * hidden + cols,
                            g2,
                            mask=(cols >= 0) & v2,
                        )
                        tl.store(
                            h_out + o3 * hidden + cols,
                            g3,
                            mask=(cols >= 0) & v3,
                        )
                    else:
                        cmask = cols < hidden
                        g0 = tl.load(
                            h_buf + b0 * stride_h + cols, mask=cmask, other=0
                        )
                        g1 = tl.load(
                            h_buf + b1 * stride_h + cols, mask=cmask, other=0
                        )
                        g2 = tl.load(
                            h_buf + b2 * stride_h + cols, mask=cmask, other=0
                        )
                        g3 = tl.load(
                            h_buf + b3 * stride_h + cols, mask=cmask, other=0
                        )
                        tl.store(h_out + o0 * hidden + cols, g0, mask=cmask)
                        tl.store(
                            h_out + o1 * hidden + cols, g1, mask=cmask & v1
                        )
                        tl.store(
                            h_out + o2 * hidden + cols, g2, mask=cmask & v2
                        )
                        tl.store(
                            h_out + o3 * hidden + cols, g3, mask=cmask & v3
                        )


# Persistent-grid cap. The eval device runs 48 SIPs; larger grids only
# add CTA dispatch latency for this memory-bound row gather (measured
# monotonically worse at 96/192/m for both kernels).
_MAX_PROGRAMS = 48

_NUM_WARPS = 4

# Row-batch crossover: below this the per-row kernel wins (all SIPs
# busy at one row each, launch-floor bound); above it 4-row groups win
# (measured tie at m=48, u4 wins from m=64 on).
_SMALL_M_MAX = 48

# Prefer one wide (masked when the hidden size is not a power of two)
# vector op per row over several narrow tiles: one dependent memory
# latency per row instead of one per tile (measured monotonically
# better 8192 > 4096 > 2048 > 1024 at hidden=7168).
_PREFER_WIDE = True
_WIDE_CAP = 8192


def _hidden_launch(hidden):
    if hidden <= 0:
        return 1, 1, True, False
    np2 = triton.next_power_of_2(hidden)
    if np2 == hidden:
        block = min(np2, _WIDE_CAP)
        return block, (hidden + block - 1) // block, True, False
    if _PREFER_WIDE and np2 <= _WIDE_CAP:
        return np2, 1, False, False
    if hidden % 1024 == 0:
        block = 1024
        while block < _WIDE_CAP and hidden % (block * 2) == 0:
            block *= 2
        return block, hidden // block, True, False
    return 1024, 0, False, True


# Memoized (topk, hidden) -> (BLOCK_TK, BLOCK_H, NB, EVEN_TK, EVEN_H,
# LOOP_H); pure function of the two sizes, so one dict lookup replaces
# the per-call next_power_of_2 / _hidden_launch walk.
_HIDDEN_CFG_CACHE = {}

# Repeat-launch fast path: kernel identity + binder specialization
# axes (argument values, pointer 16B alignment classes, dtypes) ->
# (CompiledKernel.run, function, packed_metadata). Misses populate it
# from a standard JITFunction launch.
_LAUNCH_CACHE = {}


def _current_raw_stream(device_index):
    if _raw_stream_getter is not None:
        return _raw_stream_getter(device_index)
    return torch.gcu.current_stream(device_index).gcu_stream


def _hooks_quiescent(value):
    # triton >= 3.4 keeps the runtime launch hooks in HookChain objects
    # that are non-None even when nothing is registered; an empty chain
    # behaves exactly like "no hook" for the direct-launch bypass, while
    # a plain non-None callable (older stacks) still means a live hook.
    if value is None:
        return True
    calls = getattr(value, "calls", None)
    return calls is not None and len(calls) == 0


def _direct_launch_allowed():
    # The flagtree lite-distributed mode appends two extra launcher
    # arguments and profiling hooks need the launch_metadata object;
    # both must keep using the standard JITFunction dispatch.
    if os.environ.get("FLAGTREE_LITE_DIST"):
        return False
    if _triton_knobs is not None:
        if not _hooks_quiescent(_triton_knobs.runtime.launch_enter_hook):
            return False
        if not _hooks_quiescent(_triton_knobs.runtime.launch_exit_hook):
            return False
    return True


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    word_stride = _int64_word_stride()

    idx_t = indices if indices.is_contiguous() else indices.contiguous()
    p_buf = (
        topk_p_buf if topk_p_buf.is_contiguous() else topk_p_buf.contiguous()
    )
    i_t = (
        topk_index_buf
        if topk_index_buf.is_contiguous()
        else topk_index_buf.contiguous()
    )
    t_t = (
        output_tokens_buf
        if output_tokens_buf.is_contiguous()
        else output_tokens_buf.contiguous()
    )

    m = idx_t.shape[0]
    topk = p_buf.shape[1]

    # Word strides are pure functions of the dtypes; the int32 word views
    # themselves are only needed on the JIT binder path below.
    idx_stride = 1 if idx_t.dtype == torch.int32 else word_stride
    i_stride = 1 if i_t.dtype == torch.int32 else word_stride
    t_stride = 1 if t_t.dtype == torch.int32 else word_stride

    out_p = torch.empty((m, topk), dtype=p_buf.dtype, device=p_buf.device)
    out_i = torch.empty((m, topk), dtype=i_t.dtype, device=i_t.device)
    out_t = torch.empty((m,), dtype=t_t.dtype, device=t_t.device)
    oi_stride = 1 if out_i.dtype == torch.int32 else word_stride
    ot_stride = 1 if out_t.dtype == torch.int32 else word_stride

    if hidden_states_buf is not None:
        h_buf = (
            hidden_states_buf
            if hidden_states_buf.is_contiguous()
            else hidden_states_buf.contiguous()
        )
        hidden = h_buf.shape[1]
        out_h = torch.empty(
            (m, hidden), dtype=h_buf.dtype, device=h_buf.device
        )
        stride_h = h_buf.stride(0)
    else:
        h_buf = p_buf
        out_h = out_p
        hidden = 0
        stride_h = 1
    has_hidden = hidden > 0
    result = (
        out_p,
        out_i,
        out_t,
        (out_h if hidden_states_buf is not None else None),
    )

    if m == 0 or topk == 0:
        return result

    cfg = _HIDDEN_CFG_CACHE.get((topk, hidden))
    if cfg is None:
        block_h, nb, even_h, loop_h = _hidden_launch(hidden)
        block_tk = triton.next_power_of_2(max(topk, 1))
        cfg = (block_tk, block_h, nb, block_tk == topk, even_h, loop_h)
        _HIDDEN_CFG_CACHE[(topk, hidden)] = cfg
    block_tk, block_h, nb, even_tk, even_h, loop_h = cfg

    stride_p = p_buf.stride(0)

    if m <= _SMALL_M_MAX:
        kernel = _gather_spec_extras_kernel
        grid = (min(m, _MAX_PROGRAMS),)
    else:
        kernel = _gather_spec_extras_batch_kernel
        grid = (min((m + 3) // 4, _MAX_PROGRAMS),)

    tail = (
        m,
        topk,
        hidden,
        stride_p,
        stride_h,
        idx_stride,
        i_stride,
        t_stride,
        oi_stride,
        ot_stride,
        has_hidden,
        even_tk,
        even_h,
        loop_h,
        block_tk,
        block_h,
        nb,
    )
    device_index = torch.gcu.current_device()
    # The key is built from the original tensors: every int32 word view
    # shares its storage, so the alignment classes are identical either
    # way. The three index dtypes are part of the key as well: an
    # unsupported dtype must never match a cached entry (it could then
    # bypass the _input_int32_words check below and silently launch on
    # the wrong word layout) and always takes the binder path, which
    # raises NotImplementedError for anything but int32/int64.
    key = (
        device_index,
        m,
        topk,
        hidden,
        stride_p,
        stride_h,
        idx_stride,
        i_stride,
        t_stride,
        oi_stride,
        ot_stride,
        has_hidden,
        p_buf.dtype,
        h_buf.dtype,
        idx_t.dtype,
        i_t.dtype,
        t_t.dtype,
        idx_t.data_ptr() % 16,
        p_buf.data_ptr() % 16,
        i_t.data_ptr() % 16,
        t_t.data_ptr() % 16,
        h_buf.data_ptr() % 16,
        out_p.data_ptr() % 16,
        out_i.data_ptr() % 16,
        out_t.data_ptr() % 16,
        out_h.data_ptr() % 16,
    )

    entry = _LAUNCH_CACHE.get(key)
    if entry is not None and _direct_launch_allowed():
        run, function, packed_metadata = entry
        try:
            # Cache hit: the compiled launcher only reads each tensor
            # argument's data_ptr (word views share the storage), so the
            # five views can be skipped and the originals passed straight
            # to the C launcher.
            run(
                grid[0], 1, 1,
                _current_raw_stream(device_index),
                function,
                packed_metadata,
                None, None, None,
                idx_t, p_buf, i_t, t_t, h_buf,
                out_p, out_i, out_t, out_h,
                *tail,
            )
            return result
        except Exception:
            _LAUNCH_CACHE.pop(key, None)

    # Binder path (first call per key, live profiling hooks, lite-distributed
    # mode or a failed direct launch): hand the kernel int32 word views.
    idx_words, _ = _input_int32_words(idx_t, word_stride)
    i_words, _ = _input_int32_words(i_t, word_stride)
    t_words, _ = _input_int32_words(t_t, word_stride)
    i_out_words, _ = _output_int32_words(out_i, word_stride)
    t_out_words, _ = _output_int32_words(out_t, word_stride)
    args = (
        idx_words,
        p_buf,
        i_words,
        t_words,
        h_buf,
        out_p,
        i_out_words,
        t_out_words,
        out_h,
        *tail,
    )
    compiled = kernel.run(
        *args, grid=grid, warmup=False, num_warps=_NUM_WARPS
    )
    if entry is None and compiled is not None:
        try:
            _LAUNCH_CACHE[key] = (
                compiled.run,
                compiled.function,
                compiled.packed_metadata,
            )
        except Exception:
            pass

    return result


__all__ = ["gather_spec_extras"]
