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

"""DeepEP MoE post-reorder epilogue -- Enflame GCU specialization.

v11 (2026-11-13): E-1 second-round optimization. The topk==8 fast path
moves from the v10 BLOCK_H=1024 x NB static_range column walk to the
production post_reorder_cutlass v2 shape (same board/toolchain, probe
re-validated): all 8 slot dst/weight scalars hoisted once, then columns
walked as NF unmasked BLOCK_H=4096 chunks plus one masked tail chunk
(wide masked tail beats several narrow unmasked chunks on this device;
4096 measured best of 2048/4096/8192). num_warps=4 stays the winner for
the 4096-chunk FMA form (nw2 loses ~14% at every T; the deepgemm 1024-
tile "nw2 sweet spot" law does not transfer to this form). T >= 64
dispatch flips to a persistent 48-program grid-stride form (also nw4),
amortizing the dispatch floor and handling num_tokens > 65535; measured
cutover sits between T=32 and T=64 (T=64: 11.3 vs 12.3 us; T=512:
44.7 vs 54.4; T=2048: 153.4 vs 194.3). The v10 T<16 direct-mapped arm is
retired for topk==8: the fat-CTA row form already wins at T=1 (5.3 vs
7.0 us) because one 4096-wide CTA saturates the column loop while 7
narrow 1024 CTAs pay per-segment issue costs. The ep_gather PAIR recipe
(2 rows/program, per-j sequential accumulate) was probe-tested and
loses 2.4x at T=512 (130 vs 54 us): the 8-deep back-to-back gather
window of the FMA form beats the pair form's 2-deep window here.
Index math stays int32 (same safe domain as v10/cutlass/generic ops):
valid while down_output holds < 2^31 elements.

v10 (2026-11-08): full rewrite for the 11-07 20:22 toolchain regression
(memref_ext.slice_pad_start 8 vs 7 on BLOCK>=8192 masked row gather /
column store -- see compiler_issues/sglang_deepep_post_reorder.md).
Non-8 topk keeps the v10 direct-mapped / padded scalar-load kernels;
hidden sizes beyond the fast path (> 65536, or non-8 topk with weird
hidden) keep the masked BLOCK_H=1024 chunk fallback (masked tiles at
1024 stay outside the regression's trigger face).
"""

import triton
import triton.language as tl

# Column chunk for the topk==8 fast path (cutlass v2 shape). 4096
# measured best (2048 and 8192 both lose; wide masked tail beats
# multiple narrow unmasked chunks). Production hidden sizes are all
# <= 8192 today; hidden <= 65536 fits the NF/TAIL unrolled walk.
_BLOCK_H = 4096

# Token count at which the topk==8 fast path flips from one fat CTA
# per token (grid=(num_tokens,)) to the persistent 48-program form.
# Measured crossover (kernel-direct do_bench, H=7168, nw4): T=32 row
# 7.5 vs persist 8.0 us; T=64 row 12.3 vs persist 11.3 us.
_PERSIST_MIN_TOKENS = 64

# Fixed program count for the persistent form (dispatch-floor
# amortization; also serves num_tokens > 65535 past the grid.x cap).
_PERSIST_PROGRAMS = 48

# Largest grid.x the device accepts (vllm baseline cluster: grid.x>65535).
_MAX_GRID_X = 65535

# Hidden-size cap for the topk==8 fast path; wider hidden falls back to
# the masked 1024-chunk kernel (grid.y/compile-time reasons).
_MAX_FAST_HIDDEN = 65536

# Legacy fallback tile (non-8 topk with hidden not divisible by 1024,
# or hidden > 8192, or topk==8 hidden > 65536).
_FALLBACK_BLOCK_H = 1024

# Upper bound on concurrently resident programs for the fallback kernels.
_MAX_PROGRAMS = 96


@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden_size] expert outputs
    out_ptr,  # [num_tokens, hidden_size] merged output (written in place)
    src2dst_ptr,  # [num_tokens, topk] int32 destination rows (-1 = skip)
    w_ptr,  # [num_tokens, topk] routing weights
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    # Direct-mapped grid: program pid owns (token t, column tile hb).
    pid = tl.program_id(0)
    t = pid // NB
    hb = pid % NB
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    tbase_s = t * stride_s2d_t
    tbase_w = t * stride_w_t
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        # Scalar slot loads: no vector extraction chain, and all slot
        # loads are independent so the gathers behind them issue at
        # full memory-level parallelism.
        dst_i = tl.load(src2dst_ptr + tbase_s + i)
        w_i = tl.load(w_ptr + tbase_w + i)
        dst_c = tl.maximum(dst_i, 0)
        w_f = tl.where(dst_i >= 0, w_i * scale, 0.0)
        row = tl.load(down_ptr + dst_c * stride_down_m + offs)
        acc += row.to(tl.float32) * w_f
    tl.store(
        out_ptr + t * stride_out_m + offs,
        acc.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _post_reorder_kernel_wide(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    hidden_size,
    num_tokens,
    topk,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fallback for hidden_size > _FALLBACK_MAX_HIDDEN, hidden_size %
    _FALLBACK_BLOCK_H != 0, or topk == 8 with hidden_size >
    _MAX_FAST_HIDDEN: walk the columns in runtime-masked 1024 chunks.
    Masked tiles at 1024 stay outside the 11-07 20:22 regression's
    trigger face (only >=8192-wide masked tiles fail).

    Non-power-of-two topk reaches this kernel via TOPK = next_pow2(topk);
    padded slots re-read slot topk-1 (clamped vector read, so the
    src2dst/weight gathers never run past the [num_tokens, topk] rows)
    and are nullified by a zero weight -- the same pattern as
    _post_reorder_kernel_padded. Without the clamp, the last token's
    padded slot read past the end of src2dst/topk_weights and aborted
    the device (probe: T=300, H=999, topk=3)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    kis = tl.arange(0, TOPK)
    kic = tl.minimum(kis, topk - 1)
    for t in range(pid, num_tokens, nprog):
        dstv = tl.load(src2dst_ptr + t * stride_s2d_t + kic)
        wv = tl.where(
            dstv >= 0, tl.load(w_ptr + t * stride_w_t + kic) * scale, 0.0
        )
        wv = tl.where(kis < topk, wv, 0.0)
        dstc = tl.maximum(dstv, 0)
        for c0 in range(0, hidden_size, BLOCK_H):
            cols = c0 + tl.arange(0, BLOCK_H)
            cmask = cols < hidden_size
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for i in tl.static_range(TOPK):
                di = tl.sum(tl.where(kis == i, dstc, 0))
                wi = tl.sum(tl.where(kis == i, wv, 0.0))
                row = tl.load(
                    down_ptr + di * stride_down_m + cols,
                    mask=cmask,
                    other=0.0,
                )
                acc += row.to(tl.float32) * wi
            tl.store(
                out_ptr + t * stride_out_m + cols,
                acc.to(out_ptr.dtype.element_ty),
                mask=cmask,
            )


@triton.jit
def _post_reorder_kernel_row8(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NF: tl.constexpr,
    TAIL: tl.constexpr,
):
    """topk==8 fast path (T < _PERSIST_MIN_TOKENS): one fat program per
    token, hoisted slot scalars, NF unmasked BLOCK_H column chunks plus
    one masked tail chunk (post_reorder_cutlass v2 shape; probe
    re-validated on this stack, do_bench w1000/r100, H=7168: T=1/8/16/32
    = 5.3/5.8/5.8/7.5 us vs v10 7.0/12.3/16.0/9.6 us)."""
    t = tl.program_id(0)
    tbase_s = t * stride_s2d_t
    tbase_w = t * stride_w_t
    dst0 = tl.load(src2dst_ptr + tbase_s + 0)
    w0 = tl.load(w_ptr + tbase_w + 0)
    dst1 = tl.load(src2dst_ptr + tbase_s + 1)
    w1 = tl.load(w_ptr + tbase_w + 1)
    dst2 = tl.load(src2dst_ptr + tbase_s + 2)
    w2 = tl.load(w_ptr + tbase_w + 2)
    dst3 = tl.load(src2dst_ptr + tbase_s + 3)
    w3 = tl.load(w_ptr + tbase_w + 3)
    dst4 = tl.load(src2dst_ptr + tbase_s + 4)
    w4 = tl.load(w_ptr + tbase_w + 4)
    dst5 = tl.load(src2dst_ptr + tbase_s + 5)
    w5 = tl.load(w_ptr + tbase_w + 5)
    dst6 = tl.load(src2dst_ptr + tbase_s + 6)
    w6 = tl.load(w_ptr + tbase_w + 6)
    dst7 = tl.load(src2dst_ptr + tbase_s + 7)
    w7 = tl.load(w_ptr + tbase_w + 7)
    d0 = tl.maximum(dst0, 0) * stride_down_m
    d1 = tl.maximum(dst1, 0) * stride_down_m
    d2 = tl.maximum(dst2, 0) * stride_down_m
    d3 = tl.maximum(dst3, 0) * stride_down_m
    d4 = tl.maximum(dst4, 0) * stride_down_m
    d5 = tl.maximum(dst5, 0) * stride_down_m
    d6 = tl.maximum(dst6, 0) * stride_down_m
    d7 = tl.maximum(dst7, 0) * stride_down_m
    wv0 = tl.where(dst0 >= 0, w0 * scale, 0.0)
    wv1 = tl.where(dst1 >= 0, w1 * scale, 0.0)
    wv2 = tl.where(dst2 >= 0, w2 * scale, 0.0)
    wv3 = tl.where(dst3 >= 0, w3 * scale, 0.0)
    wv4 = tl.where(dst4 >= 0, w4 * scale, 0.0)
    wv5 = tl.where(dst5 >= 0, w5 * scale, 0.0)
    wv6 = tl.where(dst6 >= 0, w6 * scale, 0.0)
    wv7 = tl.where(dst7 >= 0, w7 * scale, 0.0)
    obase = t * stride_out_m
    for j in tl.static_range(NF):
        offs = j * BLOCK_H + tl.arange(0, BLOCK_H)
        r0 = tl.load(down_ptr + d0 + offs)
        r1 = tl.load(down_ptr + d1 + offs)
        r2 = tl.load(down_ptr + d2 + offs)
        r3 = tl.load(down_ptr + d3 + offs)
        r4 = tl.load(down_ptr + d4 + offs)
        r5 = tl.load(down_ptr + d5 + offs)
        r6 = tl.load(down_ptr + d6 + offs)
        r7 = tl.load(down_ptr + d7 + offs)
        acc = (
            (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
            + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
        ) + (
            (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
            + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
        )
        tl.store(
            out_ptr + obase + offs,
            acc.to(out_ptr.dtype.element_ty),
        )
    if TAIL:
        offs = NF * BLOCK_H + tl.arange(0, BLOCK_H)
        m = offs < H
        r0 = tl.load(down_ptr + d0 + offs, m, other=0)
        r1 = tl.load(down_ptr + d1 + offs, m, other=0)
        r2 = tl.load(down_ptr + d2 + offs, m, other=0)
        r3 = tl.load(down_ptr + d3 + offs, m, other=0)
        r4 = tl.load(down_ptr + d4 + offs, m, other=0)
        r5 = tl.load(down_ptr + d5 + offs, m, other=0)
        r6 = tl.load(down_ptr + d6 + offs, m, other=0)
        r7 = tl.load(down_ptr + d7 + offs, m, other=0)
        acc = (
            (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
            + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
        ) + (
            (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
            + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
        )
        tl.store(
            out_ptr + obase + offs,
            acc.to(out_ptr.dtype.element_ty),
            m,
        )


@triton.jit
def _post_reorder_kernel_row_persist(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    scale,
    num_tokens,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NF: tl.constexpr,
    TAIL: tl.constexpr,
):
    """topk==8 fast path (T >= _PERSIST_MIN_TOKENS or T > _MAX_GRID_X):
    the row8 body under a fixed 48-program grid-stride token loop.
    Amortizes the per-CTA dispatch floor and removes the grid.x cap.
    Measured vs the grid=(T,) row8 form (do_bench w1000/r100, H=7168,
    num_warps=4): T=64 11.3 vs 12.3 us, T=512 44.7 vs 54.4 us, T=2048
    153.4 vs 194.3 us."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, num_tokens, nprog):
        tbase_s = t * stride_s2d_t
        tbase_w = t * stride_w_t
        dst0 = tl.load(src2dst_ptr + tbase_s + 0)
        w0 = tl.load(w_ptr + tbase_w + 0)
        dst1 = tl.load(src2dst_ptr + tbase_s + 1)
        w1 = tl.load(w_ptr + tbase_w + 1)
        dst2 = tl.load(src2dst_ptr + tbase_s + 2)
        w2 = tl.load(w_ptr + tbase_w + 2)
        dst3 = tl.load(src2dst_ptr + tbase_s + 3)
        w3 = tl.load(w_ptr + tbase_w + 3)
        dst4 = tl.load(src2dst_ptr + tbase_s + 4)
        w4 = tl.load(w_ptr + tbase_w + 4)
        dst5 = tl.load(src2dst_ptr + tbase_s + 5)
        w5 = tl.load(w_ptr + tbase_w + 5)
        dst6 = tl.load(src2dst_ptr + tbase_s + 6)
        w6 = tl.load(w_ptr + tbase_w + 6)
        dst7 = tl.load(src2dst_ptr + tbase_s + 7)
        w7 = tl.load(w_ptr + tbase_w + 7)
        d0 = tl.maximum(dst0, 0) * stride_down_m
        d1 = tl.maximum(dst1, 0) * stride_down_m
        d2 = tl.maximum(dst2, 0) * stride_down_m
        d3 = tl.maximum(dst3, 0) * stride_down_m
        d4 = tl.maximum(dst4, 0) * stride_down_m
        d5 = tl.maximum(dst5, 0) * stride_down_m
        d6 = tl.maximum(dst6, 0) * stride_down_m
        d7 = tl.maximum(dst7, 0) * stride_down_m
        wv0 = tl.where(dst0 >= 0, w0 * scale, 0.0)
        wv1 = tl.where(dst1 >= 0, w1 * scale, 0.0)
        wv2 = tl.where(dst2 >= 0, w2 * scale, 0.0)
        wv3 = tl.where(dst3 >= 0, w3 * scale, 0.0)
        wv4 = tl.where(dst4 >= 0, w4 * scale, 0.0)
        wv5 = tl.where(dst5 >= 0, w5 * scale, 0.0)
        wv6 = tl.where(dst6 >= 0, w6 * scale, 0.0)
        wv7 = tl.where(dst7 >= 0, w7 * scale, 0.0)
        obase = t * stride_out_m
        for j in tl.static_range(NF):
            offs = j * BLOCK_H + tl.arange(0, BLOCK_H)
            r0 = tl.load(down_ptr + d0 + offs)
            r1 = tl.load(down_ptr + d1 + offs)
            r2 = tl.load(down_ptr + d2 + offs)
            r3 = tl.load(down_ptr + d3 + offs)
            r4 = tl.load(down_ptr + d4 + offs)
            r5 = tl.load(down_ptr + d5 + offs)
            r6 = tl.load(down_ptr + d6 + offs)
            r7 = tl.load(down_ptr + d7 + offs)
            acc = (
                (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
                + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
            ) + (
                (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
                + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
            )
            tl.store(
                out_ptr + obase + offs,
                acc.to(out_ptr.dtype.element_ty),
            )
        if TAIL:
            offs = NF * BLOCK_H + tl.arange(0, BLOCK_H)
            m = offs < H
            r0 = tl.load(down_ptr + d0 + offs, m, other=0)
            r1 = tl.load(down_ptr + d1 + offs, m, other=0)
            r2 = tl.load(down_ptr + d2 + offs, m, other=0)
            r3 = tl.load(down_ptr + d3 + offs, m, other=0)
            r4 = tl.load(down_ptr + d4 + offs, m, other=0)
            r5 = tl.load(down_ptr + d5 + offs, m, other=0)
            r6 = tl.load(down_ptr + d6 + offs, m, other=0)
            r7 = tl.load(down_ptr + d7 + offs, m, other=0)
            acc = (
                (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
                + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
            ) + (
                (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
                + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
            )
            tl.store(
                out_ptr + obase + offs,
                acc.to(out_ptr.dtype.element_ty),
                m,
            )



# Legacy fallback domain (non-8 topk): hidden wider than this (or not a
# multiple of the fallback tile) takes the masked chunk-loop kernel.
_FALLBACK_MAX_HIDDEN = 8192


def deepep_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    num_tokens = output.shape[0]
    if num_tokens == 0 or hidden_size == 0 or topk == 0:
        return output

    topk_pad = max(triton.next_power_of_2(topk), 1)
    scale = float(routed_scaling_factor)
    strides = (
        down_output.stride(0),
        output.stride(0),
        src2dst.stride(0),
        topk_weights.stride(0),
    )

    if topk == 8 and hidden_size <= _MAX_FAST_HIDDEN:
        # Fast path (cutlass-v2 shape): hoisted slot scalars + NF
        # unmasked 4096-wide column chunks + one masked tail chunk.
        # T >= 64 flips to the persistent 48-program form (dispatch
        # floor amortization; also serves num_tokens > _MAX_GRID_X).
        block = min(_BLOCK_H, triton.next_power_of_2(hidden_size))
        nf = hidden_size // block
        tail = 1 if hidden_size % block else 0
        if num_tokens >= _PERSIST_MIN_TOKENS or num_tokens > _MAX_GRID_X:
            _post_reorder_kernel_row_persist[(_PERSIST_PROGRAMS,)](
                down_output,
                output,
                src2dst,
                topk_weights,
                scale,
                num_tokens,
                *strides,
                H=hidden_size,
                BLOCK_H=block,
                NF=nf,
                TAIL=tail,
                num_warps=4,
                num_stages=1,
            )
        else:
            _post_reorder_kernel_row8[(num_tokens,)](
                down_output,
                output,
                src2dst,
                topk_weights,
                scale,
                *strides,
                H=hidden_size,
                BLOCK_H=block,
                NF=nf,
                TAIL=tail,
                num_warps=4,
                num_stages=1,
            )
        return output

    if (
        hidden_size > _FALLBACK_MAX_HIDDEN
        or hidden_size % _FALLBACK_BLOCK_H != 0
    ):
        # Fallback: hidden not divisible by the legacy column tile (or
        # wider than the legacy cap) -- masked 1024 chunks.
        grid = (min(num_tokens, _MAX_PROGRAMS),)
        _post_reorder_kernel_wide[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden_size,
            num_tokens,
            topk,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=_FALLBACK_BLOCK_H,
            num_warps=1,
            num_stages=1,
        )
        return output

    nb = hidden_size // _FALLBACK_BLOCK_H

    # Direct-mapped grid: one program per (token, column tile).
    grid = (num_tokens * nb,)
    if topk_pad == topk:
        _post_reorder_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=_FALLBACK_BLOCK_H,
            NB=nb,
            num_warps=1,
            num_stages=1,
        )
    else:
        # Rare path: pad topk to the next power of two; padded slots
        # re-read slot topk-1 and are nullified by a zero weight.
        _post_reorder_kernel_padded[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            topk,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=_FALLBACK_BLOCK_H,
            NB=nb,
            num_warps=1,
            num_stages=1,
        )
    return output


@triton.jit
def _post_reorder_kernel_padded(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    topk,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    """Fallback for topk values that are not powers of two.

    Same direct-mapped body as _post_reorder_kernel, but lane i >= topk
    re-rows slot topk-1 (clamped scalar read) and is nullified by a
    zero weight, so every scalar load stays unmasked.
    """
    pid = tl.program_id(0)
    t = pid // NB
    hb = pid % NB
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    tbase_s = t * stride_s2d_t
    tbase_w = t * stride_w_t
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        ic = tl.minimum(i, topk - 1)
        dst_i = tl.load(src2dst_ptr + tbase_s + ic)
        w_i = tl.load(w_ptr + tbase_w + ic)
        dst_c = tl.maximum(dst_i, 0)
        ok = (i < topk) & (dst_i >= 0)
        w_f = tl.where(ok, w_i * scale, 0.0)
        row = tl.load(down_ptr + dst_c * stride_down_m + offs)
        acc += row.to(tl.float32) * w_f
    tl.store(
        out_ptr + t * stride_out_m + offs,
        acc.to(out_ptr.dtype.element_ty),
    )

__all__ = ["deepep_post_reorder"]
