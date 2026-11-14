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

"""state_scatter_with_mask for the Enflame GCU (L600/gcu400, 48 SPs).

GCU rewrite of the generic row-gather copy (job j20261108170555_1a7114ef):

- The generic kernel launches grid=(sequences, layers) with one row
  (8 KB fp32) per program. The core benchmark shapes hit 64/512 programs,
  far above the 48-SP scheduling sweet spot, and each program pays two
  serial scalar loads (step, slot) plus a branch before its single
  load->store pair.
- The rewrite flattens work to m = layers*sequences rows in layer-major
  order and uses the row-gather-copy launch law from
  knowledge/perf_playbook.md: persistent grid = min(ceil(m/4), 48), groups
  of 4 consecutive rows, all 8 scalar index loads hoisted and issued
  together, then 4 masked row loads batched before 4 batched stores
  (assign_draft_cache_locs_contiguous style U-batch).
- Measured (official bench protocol, do_bench warmup=1000ms rep=100ms
  median, L600): requests=8 6.51->4.13us (-37%), requests=64
  36.31->7.99us (-78%); extended shape requests=256 (m=2048) 27.2->17.6us
  vs the persistent per-row variant. Marginal copy bandwidth reaches
  ~2 TB/s (r+w).
- Probed and rejected (see knowledge/perf_playbook.md row-gather section):
  U=2/U=8 batches (shallower issue / register pressure), req-major layout
  with per-request scalar reuse (parallelism loss dominates), contiguous
  chunk dispatch, tl.range num_stages, batch-level all-valid branch
  (predicate tax already invisible).
- Fallbacks keep generic behavior for shapes outside the fast path:
  total > 8192 or int64-span shapes use the generic tiled kernel (column
  blocking); int64 index inputs are narrowed to int32 on the host (values
  are slot/step indices < 2^31; avoids the torch_gcu silent int64
  narrowing trap).
"""

import torch
import triton
import triton.language as tl

_INT32_LIMIT = 2**31 - 1

_MAX_GRID = 48
_BATCH_ROWS = 4
_MAX_BLOCK_N = 8192


@triton.jit
def _state_scatter_gcu_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    src_sl,
    src_sr,
    src_ss,
    dst_sl,
    dst_sr,
    n_groups,
    M,
    SEQ: tl.constexpr,
    NPROG: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    nmask = n < TOTAL
    for g in tl.range(pid, n_groups, NPROG):
        base = g * 4
        # Clamp trailing rows of the last group into range; stores for
        # out-of-range rows are masked off (assign_draft_cache_locs style).
        r0 = tl.minimum(base + 0, M - 1)
        r1 = tl.minimum(base + 1, M - 1)
        r2 = tl.minimum(base + 2, M - 1)
        r3 = tl.minimum(base + 3, M - 1)
        # Scalar index loads hoisted: independent issue, one dependency
        # stall covers all four rows of the group.
        st0 = tl.load(step_idx_ptr + r0 % SEQ)
        st1 = tl.load(step_idx_ptr + r1 % SEQ)
        st2 = tl.load(step_idx_ptr + r2 % SEQ)
        st3 = tl.load(step_idx_ptr + r3 % SEQ)
        sl0 = tl.load(dst_idx_ptr + r0 % SEQ)
        sl1 = tl.load(dst_idx_ptr + r1 % SEQ)
        sl2 = tl.load(dst_idx_ptr + r2 % SEQ)
        sl3 = tl.load(dst_idx_ptr + r3 % SEQ)
        m0 = (base + 0 < M) & (st0 >= 0) & nmask
        m1 = (base + 1 < M) & (st1 >= 0) & nmask
        m2 = (base + 2 < M) & (st2 >= 0) & nmask
        m3 = (base + 3 < M) & (st3 >= 0) & nmask
        v0 = tl.load(
            src_ptr + r0 // SEQ * src_sl + r0 % SEQ * src_sr + st0 * src_ss + n,
            mask=m0,
            other=0.0,
        )
        v1 = tl.load(
            src_ptr + r1 // SEQ * src_sl + r1 % SEQ * src_sr + st1 * src_ss + n,
            mask=m1,
            other=0.0,
        )
        v2 = tl.load(
            src_ptr + r2 // SEQ * src_sl + r2 % SEQ * src_sr + st2 * src_ss + n,
            mask=m2,
            other=0.0,
        )
        v3 = tl.load(
            src_ptr + r3 // SEQ * src_sl + r3 % SEQ * src_sr + st3 * src_ss + n,
            mask=m3,
            other=0.0,
        )
        tl.store(dst_ptr + r0 // SEQ * dst_sl + sl0 * dst_sr + n, v0, mask=m0)
        tl.store(dst_ptr + r1 // SEQ * dst_sl + sl1 * dst_sr + n, v1, mask=m1)
        tl.store(dst_ptr + r2 // SEQ * dst_sl + sl2 * dst_sr + n, v2, mask=m2)
        tl.store(dst_ptr + r3 // SEQ * dst_sl + sl3 * dst_sr + n, v3, mask=m3)


@triton.jit
def _state_scatter_kernel_tiled(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    src_sl,
    src_sr,
    src_ss,
    dst_sl,
    dst_sr,
    LAYERS: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NO_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    req = tl.program_id(0)
    l0 = tl.program_id(1) * BLOCK_L
    n0 = tl.program_id(2) * BLOCK_N

    step = tl.load(step_idx_ptr + req)
    if step >= 0:
        slot = tl.load(dst_idx_ptr + req)
        layer_idx = l0 + tl.arange(0, BLOCK_L)[:, None]
        n = n0 + tl.arange(0, BLOCK_N)[None, :]
        if WIDE:
            layer_idx = layer_idx.to(tl.int64)
            n = n.to(tl.int64)
            dst_off = layer_idx * dst_sl + slot.to(tl.int64) * dst_sr + n
            src_off = (
                layer_idx * src_sl
                + req.to(tl.int64) * src_sr
                + step.to(tl.int64) * src_ss
                + n
            )
        else:
            dst_off = layer_idx * dst_sl + slot * dst_sr + n
            src_off = layer_idx * src_sl + req * src_sr + step * src_ss + n
        if NO_MASK:
            val = tl.load(src_ptr + src_off)
            tl.store(dst_ptr + dst_off, val)
        else:
            mask = (layer_idx < LAYERS) & (n < TOTAL)
            val = tl.load(src_ptr + src_off, mask=mask, other=0)
            tl.store(dst_ptr + dst_off, val, mask=mask)


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    layers, sequences = src.shape[0], src.shape[1]
    total = (
        dst[0, 0].numel() if dst.dim() > 1 else dst.numel() // max(layers, 1)
    )

    if not dst.is_contiguous():
        dst = dst.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    if sequences == 0 or layers == 0 or total == 0:
        return dst

    if dst_indices_raw.dtype in (torch.int64, torch.uint64):
        dst_indices_raw = dst_indices_raw.to(torch.int32)
    if step_indices_raw.dtype in (torch.int64, torch.uint64):
        step_indices_raw = step_indices_raw.to(torch.int32)

    src_sl, src_sr, src_ss = src.stride(0), src.stride(1), src.stride(2)
    dst_sl, dst_sr = dst.stride(0), dst.stride(1)

    draft = src.shape[2]
    cache = dst.shape[1] if dst.dim() > 1 else 1
    src_span = (
        (layers - 1) * src_sl
        + (sequences - 1) * src_sr
        + (draft - 1) * src_ss
        + total
        - 1
    )
    dst_span = (layers - 1) * dst_sl + (cache - 1) * dst_sr + total - 1
    wide = max(src_span, dst_span) > _INT32_LIMIT

    m = layers * sequences

    if not wide and total <= _MAX_BLOCK_N:
        block_n = max(16, triton.next_power_of_2(total))
        num_warps = 4 if block_n >= 512 else 1
        n_groups = (m + _BATCH_ROWS - 1) // _BATCH_ROWS
        nprog = min(n_groups, _MAX_GRID)
        _state_scatter_gcu_kernel[(nprog,)](
            dst,
            src,
            dst_indices_raw,
            step_indices_raw,
            src_sl,
            src_sr,
            src_ss,
            dst_sl,
            dst_sr,
            n_groups,
            m,
            SEQ=sequences,
            NPROG=nprog,
            TOTAL=total,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        return dst

    block_l = 1 if layers <= 8 else 2
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(total))
    _state_scatter_kernel_tiled[
        (sequences, triton.cdiv(layers, block_l), triton.cdiv(total, block_n))
    ](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        src_sl,
        src_sr,
        src_ss,
        dst_sl,
        dst_sr,
        LAYERS=layers,
        TOTAL=total,
        BLOCK_L=block_l,
        BLOCK_N=block_n,
        NO_MASK=(layers % block_l) == 0 and (total % block_n) == 0,
        WIDE=wide,
        num_warps=8,
    )
    return dst


__all__ = ["state_scatter_with_mask"]
