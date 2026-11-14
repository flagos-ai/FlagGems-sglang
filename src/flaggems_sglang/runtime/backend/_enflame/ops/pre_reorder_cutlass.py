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

import triton
import triton.language as tl

# Below this many output elements, 32-bit offsets are safe (offsets are
# ``dst * hidden_size + col < n * hidden_size``).
_INT32_MAX_ELEMS = 1 << 31

# Launch regimes (deterministic host-side heuristic, no autotune):
# - tokens <= _SMALL_T: column-parallel 2D grid (grid.y = tokens). CTA count
#   stays <= 56 at hidden%1024==0, so the ~48ns/CTA dispatch floor never bites
#   and column blocks run concurrently (a serial in-CTA hidden loop costs
#   ~3us per dependent DMA step at these sizes).
# - _SMALL_T < tokens <= _MID_T: wide-chunk 2D-tile kernel ([KP2, 2048]
#   tiles, row base hoisted). Short chains are load-chain bound; 4 wide
#   loads per token instead of 14 takes T=64 from 27.4us to 23.9us
#   (-13%, 2026-11-13).
# - tokens > _MID_T and hidden decomposes into <= _MAX_ROW_CHUNKS pow2
#   chunks within the DSM budget: whole-row per-slot kernel
#   (_pre_reorder_scatter_rowchunks) — each live slot's row is written as NC
#   contiguous 1D wide stores and the row is read with NC wide loads. At
#   hidden 7168/topk 8 that is 24 store descriptors per token instead of 112
#   ([8, 512] tile rows) and 3 loads instead of 14, which takes T=512 from
#   142.2us to 111.8us (-21%) at the ~507GB/s scattered-row write wall
#   (store_floor probes; the win is load-side exposure removal, measured
#   2026-11-13 on GCU400/L600).
# - otherwise (odd hidden or beyond the whole-row DSM budget): persistent
#   token-stride 1D grid of min(tokens, 48) programs with the hidden-block
#   loop inside the program. Dispatch cost is pinned to one wave and the
#   [num_tokens, topk] index vectors are loaded once per token and reused
#   by every hidden block (the 2D-grid form at num_tokens=512 launched 1785
#   CTAs whose ~135us empty-grid dispatch floor was 68% of the old kernel's
#   wall time).
_SMALL_T = 8
_MID_T = 128  # <=: wide-chunk 2D-tile kernel; >: whole-row per-slot kernel
_PERSIST_CTAS = (
    48  # one wave on the 48-SPC GCU (see knowledge/perf_playbook.md)
)
_MAX_ROW_CHUNKS = (
    3  # whole-row kernel widths C0/C1/C2; more chunks -> tile form
)
# Whole-row kernel DSM: all NC chunk buffers coexist, ~1.75B per hidden
# element (measured: hidden 524288 required 917632B vs the 917504B hardware
# limit, 2026-11-13). 131072 keeps the footprint ~229KB (4x headroom);
# wider hidden falls back to the [KP2, 512]-tile tokgrid kernel.
_ROWCHUNKS_MAX_HIDDEN = 131072


@triton.jit
def _pre_reorder_scatter_grid2d(
    input_ptr,  # [num_tokens, hidden] token rows to broadcast
    out_ptr,  # [n, hidden] gateup_input, written in place
    dst_ptr,  # [num_tokens, topk] int32 permutation of [0, n)
    ids_ptr,  # [num_tokens, topk] int32 expert ids
    scale_ptr,  # [1] fp32 (or any tensor when HAS_SCALE is False)
    num_tokens,
    HAS_SCALE: tl.constexpr,
    HIDDEN: tl.constexpr,  # hidden_size as compile-time constant
    NLE: tl.constexpr,  # num_local_experts as compile-time constant
    TOPK: tl.constexpr,
    KP2: tl.constexpr,  # next_power_of_2(TOPK) for the slot vector
    POW2_TOPK: tl.constexpr,  # TOPK == KP2: slot loads need no k mask
    INT64: tl.constexpr,  # use 64-bit row offsets (huge outputs only)
    BLOCK_H: tl.constexpr,
):
    EVEN_H: tl.constexpr = HIDDEN % BLOCK_H == 0  # column masks elided

    # Column tile is loop-invariant: one contiguous, BLOCK_H-aligned slab of
    # each row (tell the compiler so loads/stores vectorize into wide ops).
    hb = tl.program_id(0)
    cols = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_H), BLOCK_H)
    k = tl.arange(0, KP2)

    pid_t = tl.program_id(1)
    num_pid_t = tl.num_programs(1)

    if HAS_SCALE:
        inv_scale = 1.0 / tl.load(scale_ptr)  # loop-invariant reciprocal

    for t in range(pid_t, num_tokens, num_pid_t):
        # All topk slots of this token in one vectorized load each.
        if POW2_TOPK:
            ids_v = tl.load(ids_ptr + t * TOPK + k)
            dst_v = tl.load(dst_ptr + t * TOPK + k)
        else:
            kmask = k < TOPK
            ids_v = tl.load(ids_ptr + t * TOPK + k, mask=kmask, other=NLE)
            dst_v = tl.load(dst_ptr + t * TOPK + k, mask=kmask, other=0)
        if INT64:
            dst_v = dst_v.to(tl.int64)
        # Slots routed off this EP rank keep their original gateup row.
        live = ids_v != NLE

        # Token row, dequantized once and reused for all of the token's slots.
        if INT64:
            if EVEN_H:
                x = tl.load(
                    input_ptr + t.to(tl.int64) * HIDDEN + cols.to(tl.int64)
                )
            else:
                x = tl.load(
                    input_ptr + t.to(tl.int64) * HIDDEN + cols.to(tl.int64),
                    mask=cols < HIDDEN,
                    other=0.0,
                )
            off = dst_v[:, None] * HIDDEN + cols.to(tl.int64)[None, :]
        else:
            if EVEN_H:
                x = tl.load(input_ptr + t * HIDDEN + cols)
            else:
                x = tl.load(
                    input_ptr + t * HIDDEN + cols,
                    mask=cols < HIDDEN,
                    other=0.0,
                )
            off = dst_v[:, None] * HIDDEN + cols[None, :]
        x = x.to(tl.float32)
        if HAS_SCALE:
            x = x * inv_scale
        x = x.to(out_ptr.dtype.element_ty)

        # All-live fast path: drop the per-row store predicate entirely (the
        # masked [KP2, BLOCK_H] store lowers markedly slower; ~78% of tokens
        # take this path at the benchmark's uniform id distribution).
        # Predicate form matters: tl.max(ids_v) < NLE beats the equivalent
        # tl.sum((ids_v != NLE)) == TOPK by ~11% wall (the small-vector sum
        # reduction lowers worse). Non-power-of-2 topk keeps this
        # conservative (padding lanes load other=NLE -> always slow path).
        all_live = tl.max(ids_v) < NLE
        if EVEN_H:
            if all_live:
                tl.store(out_ptr + off, x[None, :])
            else:
                tl.store(out_ptr + off, x[None, :], mask=live[:, None])
        else:
            cmask = cols < HIDDEN
            if all_live:
                tl.store(out_ptr + off, x[None, :], mask=cmask[None, :])
            else:
                tl.store(
                    out_ptr + off,
                    x[None, :],
                    mask=live[:, None] & cmask[None, :],
                )


@triton.jit
def _pre_reorder_scatter_tokgrid(
    input_ptr,  # [num_tokens, hidden] token rows to broadcast
    out_ptr,  # [n, hidden] gateup_input, written in place
    dst_ptr,  # [num_tokens, topk] int32 permutation of [0, n)
    ids_ptr,  # [num_tokens, topk] int32 expert ids
    scale_ptr,  # [1] fp32 (or any tensor when HAS_SCALE is False)
    num_tokens,
    HAS_SCALE: tl.constexpr,
    HIDDEN: tl.constexpr,
    NLE: tl.constexpr,
    TOPK: tl.constexpr,
    KP2: tl.constexpr,
    POW2_TOPK: tl.constexpr,
    INT64: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    EVEN_H: tl.constexpr = HIDDEN % BLOCK_H == 0

    pid_t = tl.program_id(0)
    num_pid_t = tl.num_programs(0)
    k = tl.arange(0, KP2)

    if HAS_SCALE:
        inv_scale = 1.0 / tl.load(scale_ptr)

    for t in range(pid_t, num_tokens, num_pid_t):
        # Index vectors are loaded once per token and reused by every hidden
        # block below (the 2D-grid form reloaded them per column block).
        if POW2_TOPK:
            ids_v = tl.load(ids_ptr + t * TOPK + k)
            dst_v = tl.load(dst_ptr + t * TOPK + k)
        else:
            kmask = k < TOPK
            ids_v = tl.load(ids_ptr + t * TOPK + k, mask=kmask, other=NLE)
            dst_v = tl.load(dst_ptr + t * TOPK + k, mask=kmask, other=0)
        if INT64:
            dst_v = dst_v.to(tl.int64)
        live = ids_v != NLE
        # See the grid2d kernel: tl.max predicate form, ~11% faster than the
        # equivalent tl.sum reduction on this backend.
        all_live = tl.max(ids_v) < NLE

        for hb in range(0, HIDDEN, BLOCK_H):
            cols = hb + tl.arange(0, BLOCK_H)
            cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_H), BLOCK_H)
            if INT64:
                if EVEN_H:
                    x = tl.load(
                        input_ptr + t.to(tl.int64) * HIDDEN + cols.to(tl.int64)
                    )
                else:
                    x = tl.load(
                        input_ptr
                        + t.to(tl.int64) * HIDDEN
                        + cols.to(tl.int64),
                        mask=cols < HIDDEN,
                        other=0.0,
                    )
                off = dst_v[:, None] * HIDDEN + cols.to(tl.int64)[None, :]
            else:
                if EVEN_H:
                    x = tl.load(input_ptr + t * HIDDEN + cols)
                else:
                    x = tl.load(
                        input_ptr + t * HIDDEN + cols,
                        mask=cols < HIDDEN,
                        other=0.0,
                    )
                off = dst_v[:, None] * HIDDEN + cols[None, :]
            x = x.to(tl.float32)
            if HAS_SCALE:
                x = x * inv_scale
            x = x.to(out_ptr.dtype.element_ty)

            if EVEN_H:
                if all_live:
                    tl.store(out_ptr + off, x[None, :])
                else:
                    tl.store(out_ptr + off, x[None, :], mask=live[:, None])
            else:
                cmask = cols < HIDDEN
                if all_live:
                    tl.store(out_ptr + off, x[None, :], mask=cmask[None, :])
                else:
                    tl.store(
                        out_ptr + off,
                        x[None, :],
                        mask=live[:, None] & cmask[None, :],
                    )


def _row_chunks(hidden_size):
    # Binary decomposition of hidden_size into pow2 column chunks
    # (e.g. 7168 -> [4096@0, 2048@4096, 1024@6144]); each chunk is a
    # mask-free tl.arange width for the whole-row kernel.
    chunks = []
    remaining, offset = hidden_size, 0
    while remaining > 0:
        width = 1
        while width * 2 <= remaining:
            width *= 2
        chunks.append((width, offset))
        offset += width
        remaining -= width
    return chunks


@triton.jit
def _pre_reorder_scatter_rowchunks(
    input_ptr,  # [num_tokens, hidden] token rows to broadcast
    out_ptr,  # [n, hidden] gateup_input, written in place
    dst_ptr,  # [num_tokens, topk] int32 permutation of [0, n)
    ids_ptr,  # [num_tokens, topk] int32 expert ids
    scale_ptr,  # [1] fp32 (or any tensor when HAS_SCALE is False)
    num_tokens,
    HAS_SCALE: tl.constexpr,
    HIDDEN: tl.constexpr,
    NLE: tl.constexpr,
    TOPK: tl.constexpr,
    KP2: tl.constexpr,
    POW2_TOPK: tl.constexpr,
    INT64: tl.constexpr,
    C0: tl.constexpr,  # pow2 chunk widths, descending; C0 covers [0, C0)
    C1: tl.constexpr,  # covers [O1, O1+C1) when NC > 1
    C2: tl.constexpr,  # covers [O2, O2+C2) when NC > 2
    O1: tl.constexpr,
    O2: tl.constexpr,
    NC: tl.constexpr,  # 1..3 chunks; C0+C1+C2 == HIDDEN when all present
):
    # Whole-row per-slot form: each live slot's row is written with NC
    # contiguous 1D wide-segment stores (no [KP2, BLOCK] 2D-tile lowering, no
    # column masks). At hidden 7168/topk 8 that is 24 wide store descriptors
    # per token vs 112 one-KB row segments for the tile form, and the row is
    # read with 3 wide loads vs 14 narrow ones (load-chain exposure -31us at
    # T=512); writes stay at the scattered-row wall either way.
    pid_t = tl.program_id(0)
    num_pid_t = tl.num_programs(0)
    k = tl.arange(0, KP2)
    if HAS_SCALE:
        inv_scale = 1.0 / tl.load(scale_ptr)

    cols0 = tl.arange(0, C0)
    cols0 = tl.max_contiguous(tl.multiple_of(cols0, C0), C0)

    for t in range(pid_t, num_tokens, num_pid_t):
        if POW2_TOPK:
            ids_v = tl.load(ids_ptr + t * TOPK + k)
            dst_v = tl.load(dst_ptr + t * TOPK + k)
        else:
            kmask = k < TOPK
            ids_v = tl.load(ids_ptr + t * TOPK + k, mask=kmask, other=NLE)
            dst_v = tl.load(dst_ptr + t * TOPK + k, mask=kmask, other=0)
        if INT64:
            dst_v = dst_v.to(tl.int64)
        all_live = tl.max(ids_v) < NLE

        # NC chunk loads issued upfront (independent DMAs, shared by slots).
        if INT64:
            t64 = t.to(tl.int64)
            x0 = tl.load(input_ptr + t64 * HIDDEN + cols0.to(tl.int64))
        else:
            x0 = tl.load(input_ptr + t * HIDDEN + cols0)
        x0 = x0.to(tl.float32)
        if HAS_SCALE:
            x0 = x0 * inv_scale
        x0 = x0.to(out_ptr.dtype.element_ty)
        if NC > 1:
            cols1 = O1 + tl.arange(0, C1)
            cols1 = tl.max_contiguous(tl.multiple_of(cols1, C1), C1)
            if INT64:
                x1 = tl.load(input_ptr + t64 * HIDDEN + cols1.to(tl.int64))
            else:
                x1 = tl.load(input_ptr + t * HIDDEN + cols1)
            x1 = x1.to(tl.float32)
            if HAS_SCALE:
                x1 = x1 * inv_scale
            x1 = x1.to(out_ptr.dtype.element_ty)
        if NC > 2:
            cols2 = O2 + tl.arange(0, C2)
            cols2 = tl.max_contiguous(tl.multiple_of(cols2, C2), C2)
            if INT64:
                x2 = tl.load(input_ptr + t64 * HIDDEN + cols2.to(tl.int64))
            else:
                x2 = tl.load(input_ptr + t * HIDDEN + cols2)
            x2 = x2.to(tl.float32)
            if HAS_SCALE:
                x2 = x2 * inv_scale
            x2 = x2.to(out_ptr.dtype.element_ty)

        if all_live:
            for kk in tl.static_range(KP2):
                dst_kk = tl.sum(tl.where(k == kk, dst_v, 0))
                r0 = dst_kk * HIDDEN
                tl.store(out_ptr + r0 + cols0, x0)
                if NC > 1:
                    tl.store(out_ptr + r0 + cols1, x1)
                if NC > 2:
                    tl.store(out_ptr + r0 + cols2, x2)
        else:
            for kk in tl.static_range(KP2):
                id_kk = tl.sum(tl.where(k == kk, ids_v, 0))
                if id_kk != NLE:
                    dst_kk = tl.sum(tl.where(k == kk, dst_v, 0))
                    r0 = dst_kk * HIDDEN
                    tl.store(out_ptr + r0 + cols0, x0)
                    if NC > 1:
                        tl.store(out_ptr + r0 + cols1, x1)
                    if NC > 2:
                        tl.store(out_ptr + r0 + cols2, x2)


@triton.jit
def _pre_reorder_scatter_wchunk(
    input_ptr,  # [num_tokens, hidden] token rows to broadcast
    out_ptr,  # [n, hidden] gateup_input, written in place
    dst_ptr,  # [num_tokens, topk] int32 permutation of [0, n)
    ids_ptr,  # [num_tokens, topk] int32 expert ids
    scale_ptr,  # [1] fp32 (or any tensor when HAS_SCALE is False)
    num_tokens,
    HAS_SCALE: tl.constexpr,
    HIDDEN: tl.constexpr,
    NLE: tl.constexpr,
    TOPK: tl.constexpr,
    KP2: tl.constexpr,
    POW2_TOPK: tl.constexpr,
    INT64: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Wide-chunk 2D-tile form for the short-chain regime (few tokens per
    # CTA): [KP2, CHUNK] tiles with the row-base product hoisted out of the
    # chunk loop. At T=64/hidden 7168 it beats the tile-512 form (23.9us vs
    # 27.4us) and the whole-row form (27.5us) because the load chain, not
    # the store wall, binds there: 4 chunk loads per token instead of 14.
    # Crossovers measured 2026-11-13: wins up to T=128, loses from T=256.
    EVEN_C: tl.constexpr = HIDDEN % CHUNK == 0
    pid_t = tl.program_id(0)
    num_pid_t = tl.num_programs(0)
    k = tl.arange(0, KP2)
    if HAS_SCALE:
        inv_scale = 1.0 / tl.load(scale_ptr)
    for t in range(pid_t, num_tokens, num_pid_t):
        if POW2_TOPK:
            ids_v = tl.load(ids_ptr + t * TOPK + k)
            dst_v = tl.load(dst_ptr + t * TOPK + k)
        else:
            kmask = k < TOPK
            ids_v = tl.load(ids_ptr + t * TOPK + k, mask=kmask, other=NLE)
            dst_v = tl.load(dst_ptr + t * TOPK + k, mask=kmask, other=0)
        if INT64:
            dst_v = dst_v.to(tl.int64)
        live = ids_v != NLE
        all_live = tl.max(ids_v) < NLE
        row_base = dst_v * HIDDEN
        for off in range(0, HIDDEN, CHUNK):
            cols = off + tl.arange(0, CHUNK)
            cols = tl.max_contiguous(tl.multiple_of(cols, CHUNK), CHUNK)
            if INT64:
                x = tl.load(
                    input_ptr + t.to(tl.int64) * HIDDEN + cols.to(tl.int64)
                )
                o = row_base[:, None] + cols.to(tl.int64)[None, :]
            else:
                x = tl.load(input_ptr + t * HIDDEN + cols)
                o = row_base[:, None] + cols[None, :]
            x = x.to(tl.float32)
            if HAS_SCALE:
                x = x * inv_scale
            x = x.to(out_ptr.dtype.element_ty)
            if EVEN_C:
                if all_live:
                    tl.store(out_ptr + o, x[None, :])
                else:
                    tl.store(out_ptr + o, x[None, :], mask=live[:, None])
            else:
                cmask = cols < HIDDEN
                if all_live:
                    tl.store(out_ptr + o, x[None, :], mask=cmask[None, :])
                else:
                    tl.store(
                        out_ptr + o,
                        x[None, :],
                        mask=live[:, None] & cmask[None, :],
                    )


def _block_h_grid2d(hidden_size, num_tokens):
    # T<=3: 512-wide blocks put 14 CTAs on hidden 7168 instead of 7 and win
    # the single-wave latency split (4.66us vs 4.87us at T=1, 2026-11-13).
    # T>=4 keeps 1024 (8.91us vs 10.44us at T=8: past ~48 CTAs the extra
    # wave costs more than the finer split saves).
    if num_tokens <= 3 and hidden_size % 512 == 0:
        return 512
    if hidden_size % 1024 == 0:
        return 1024
    if hidden_size % 512 == 0:
        return 512
    return max(triton.next_power_of_2(hidden_size), 16)


def _block_h_tokgrid(hidden_size):
    # 512 measured fastest for the persistent form ([KP2, 512] stores) at
    # hidden 7168: 142.5us vs 152.3us (1024) and 242.7us (256) at T=512.
    return min(512, max(triton.next_power_of_2(hidden_size), 16))


def pre_reorder_cutlass(
    input,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
):
    out = gateup_input
    if topk_ids.numel() == 0:
        return out

    # The kernels index with flat row-major offsets; make that assumption
    # explicit (host-side, free for the already-contiguous benchmark cases).
    if not input.is_contiguous():
        input = input.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    if not src2dst.is_contiguous():
        src2dst = src2dst.contiguous()
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()

    int64 = out.numel() >= _INT32_MAX_ELEMS
    scale_arg = a1_scales if a1_scales is not None else input
    has_scale = a1_scales is not None

    # num_warps=1: the [KP2, BLOCK_H] broadcast store tile fails to compile
    # on triton_gcu 3.6.0 (2026-11-07 build) whenever BLOCK_H >= 256 and
    # num_warps >= 2 (memref_ext.slice_start operandSegmentSizes 6 vs 7 — see
    # compiler_issues/sglang_pre_reorder_cutlass.md); the whole-row kernel
    # only emits 1D stores (immune) but nw1 stays the fastest launch shape in
    # the >48-CTA regime, so both keep num_warps=1.
    if num_tokens <= _SMALL_T:
        block_h = _block_h_grid2d(hidden_size, num_tokens)
        grid = (triton.cdiv(hidden_size, block_h), num_tokens)
        _pre_reorder_scatter_grid2d[grid](
            input,
            out,
            src2dst,
            topk_ids,
            scale_arg,
            num_tokens,
            HAS_SCALE=has_scale,
            INT64=int64,
            HIDDEN=hidden_size,
            NLE=num_local_experts,
            TOPK=topk,
            KP2=triton.next_power_of_2(topk),
            POW2_TOPK=(topk == triton.next_power_of_2(topk)),
            BLOCK_H=block_h,
            num_warps=1,
        )
        return out

    if num_tokens <= _MID_T:
        chunk = min(2048, max(triton.next_power_of_2(hidden_size), 16))
        grid = (min(num_tokens, _PERSIST_CTAS),)
        _pre_reorder_scatter_wchunk[grid](
            input,
            out,
            src2dst,
            topk_ids,
            scale_arg,
            num_tokens,
            HAS_SCALE=has_scale,
            INT64=int64,
            HIDDEN=hidden_size,
            NLE=num_local_experts,
            TOPK=topk,
            KP2=triton.next_power_of_2(topk),
            POW2_TOPK=(topk == triton.next_power_of_2(topk)),
            CHUNK=chunk,
            num_warps=1,
        )
        return out

    chunks = _row_chunks(hidden_size)
    if len(chunks) <= _MAX_ROW_CHUNKS and hidden_size <= _ROWCHUNKS_MAX_HIDDEN:
        widths = [c for c, _ in chunks]
        offsets = [o for _, o in chunks]
        widths += [1] * (_MAX_ROW_CHUNKS - len(widths))
        offsets += [0] * (_MAX_ROW_CHUNKS - len(offsets))
        grid = (min(num_tokens, _PERSIST_CTAS),)
        _pre_reorder_scatter_rowchunks[grid](
            input,
            out,
            src2dst,
            topk_ids,
            scale_arg,
            num_tokens,
            HAS_SCALE=has_scale,
            INT64=int64,
            HIDDEN=hidden_size,
            NLE=num_local_experts,
            TOPK=topk,
            KP2=triton.next_power_of_2(topk),
            POW2_TOPK=(topk == triton.next_power_of_2(topk)),
            C0=widths[0],
            C1=widths[1],
            C2=widths[2],
            O1=offsets[1],
            O2=offsets[2],
            NC=len(chunks),
            num_warps=1,
        )
    else:
        block_h = _block_h_tokgrid(hidden_size)
        grid = (min(num_tokens, _PERSIST_CTAS),)
        _pre_reorder_scatter_tokgrid[grid](
            input,
            out,
            src2dst,
            topk_ids,
            scale_arg,
            num_tokens,
            HAS_SCALE=has_scale,
            INT64=int64,
            HIDDEN=hidden_size,
            NLE=num_local_experts,
            TOPK=topk,
            KP2=triton.next_power_of_2(topk),
            POW2_TOPK=(topk == triton.next_power_of_2(topk)),
            BLOCK_H=block_h,
            num_warps=1,
        )
    return out


__all__ = ["pre_reorder_cutlass"]
