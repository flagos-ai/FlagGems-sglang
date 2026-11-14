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

"""MoE cutlass post-reorder epilogue -- Enflame GCU specialization.

v2 (2026-11-08): full rewrite. The v1 kernel (BLOCK=8192 masked row
gather, num_warps=1) no longer first-compiles on the 2026-11-07 20:22
toolchain stack: its wide masked loads lower to memref_ext.slice_pad_start
and the process-global flaggems dialect shim rewrites those IR lines into
a 7-vs-8 operand mismatch with the external gcu-compiler-opt tool (see
compiler_issues/sglang_post_reorder_cutlass.md). The recorded v1 bench
numbers (15.90/16.04/33.76/180.85 us at T=1/8/64/512, H=7168, topk=8)
were pre-upgrade cache hits; fresh compiles fail, so v2 is both a fix and
an optimization.

v2 shape (probe-validated on the patched pipeline, kernel-direct
do_bench warmup=1000ms/rep=100ms, GCU400/L600):
- One program per token (grid=(num_tokens,)), all 8 slot dst/eid/weight
  scalars hoisted once as independent scalar loads (no vector-lane
  extraction chains).
- Columns walked as full BLOCK=4096 chunks that are fully UNMASKED
  (H // BLOCK_H chunks), plus one masked 4096 tail chunk when
  H % BLOCK_H != 0. Unmasked full chunks beat all-masked by ~12% at
  T=512 and beat a fully-unmasked piecewise split (4096+2048+1024) --
  one wide masked tail load is cheaper than several narrow unmasked
  loads on this device.
- K=8 fast path (production/bench shape): 6.48/6.68/15.43/76.52/285.99
  us at T=1/8/64/512/2048 (H=7168, bf16) vs the v1 baseline
  15.90/16.04/33.76/180.85 us -- 2.2x-2.45x.
- Every other topk (1..8 generic, up to 16) takes a generic masked-chunk
  row kernel with a strided persistent token loop (also used for
  num_tokens > 65535 to respect the grid.x limit). topk > 16 and
  non-unit column strides fall back to the legacy strided kernel.
- Index math stays int32 (same safe domain as v1 and the generic ops):
  valid while down_output holds < 2^31 elements.

v3 (2026-11-13): second-round RSI pass over the v2 fast path. Probes
(PAIR recipe from the ep_gather/moe_fused_mul_sum 3-scalar-chain gather
family, num_warps sweep, column-split 2D grid, persistent grid) measured
on GCU400/L600, kernel-direct do_bench w1000/r100:
- PAIR rejected at every T and every num_warps (pair1d +27%, pair2d +31%
  at T=512): this kernel's 8-gather burst already saturates the in-flight
  window, so pairing only doubles live bytes (the known dual-chunk
  pathology) or serializes the burst behind a k-loop.
- num_warps stays 4 (nw2 +12%, nw1 +23% at T=512) -- the deepgemm
  nw2 sweet spot belongs to its 1024-wide chunk form, not 4096 bursts.
- Dispatch floor 2.66us + 52.7ns/CTA: at T=512 dispatch is ~29.6us,
  39% of the 75us kernel. Two new arms buy it back:
  * T <= 16: _reorder_row8_colsplit, one (token, chunk) per program
    (halves the serial per-CTA work) -- 5.70/6.13/6.44us at T=1/8/12
    vs 6.57/6.82/7.11 (-13%/-10%/-9%); loses from T>=24 (grid.y CTAs
    cost more than the halved work saves).
  * T >= 64: _reorder_row8_persist, 48 persistent programs walk tokens
    strided with the unchanged per-token body -- 13.77/57.4/208us at
    T=64/512/2048 vs 14.95/75.1/281 (-7.7%/-23.5%/-25.4%).
  * 17 <= T < 64 keeps the v2 grid=(T,) form (strided-loop scaffolding
    costs +0.3-0.4us below one wave; same (32,64] cutover as the
    deepep_post_reorder instance of this family).
- Everything else (generic topk arm, legacy strided fallback, index
  math, per-token arithmetic) is unchanged; the new arms reuse the same
  _row8_slots/_row8_fma helpers, so per-token results are bit-identical
  to v2.
"""

import triton
import triton.language as tl

# Column chunk width for the fast path. 4096 measured best (2048 and
# 8192 both lose; see the probe matrix in the job log).
_BLOCK_H = 4096

# Persistent-program count for the large-T arm: 48 SPUs on GCU400, same
# family precedent as post_reorder_deepgemm's persist-48.
_PERSIST_NPROG = 48

# Token count from which the persistent arm wins: measured v2 ahead at
# T=32/48 (+0.3-0.4us loop scaffolding), persist48 ahead at T=64 (-1.2us
# dispatch saving); same (32,64] cutover window as the deepep_post_reorder
# instance of this family.
_PERSIST_MIN_T = 64

# Token count below which the column-split arm wins: measured -13% at
# T=1, -10% at T=8, -9% at T=12, tie at T=16, loss from T=24 (extra
# grid.y programs cost more dispatch than the halved work saves).
_COLSPLIT_MAX_T = 16

# Largest grid.x the device accepts (vllm baseline cluster: grid.x>65535).
_MAX_GRID_X = 65535

# Largest topk unrolled by the generic kernel; beyond this the legacy
# strided fallback serves the (rare) request.
_MAX_TOPK_UNROLL = 16


@triton.jit
def _row8_slots(Idx, E, W, tbi, tbe, tbw, NLE, ROWS, sd, scale):
    v = tl.load(Idx + tbi + 0).to(tl.int32)
    d0 = tl.where(v < 0, v + ROWS, v) * sd
    w0 = tl.where(
        tl.load(E + tbe + 0).to(tl.int32) != NLE,
        tl.load(W + tbw + 0).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 1).to(tl.int32)
    d1 = tl.where(v < 0, v + ROWS, v) * sd
    w1 = tl.where(
        tl.load(E + tbe + 1).to(tl.int32) != NLE,
        tl.load(W + tbw + 1).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 2).to(tl.int32)
    d2 = tl.where(v < 0, v + ROWS, v) * sd
    w2 = tl.where(
        tl.load(E + tbe + 2).to(tl.int32) != NLE,
        tl.load(W + tbw + 2).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 3).to(tl.int32)
    d3 = tl.where(v < 0, v + ROWS, v) * sd
    w3 = tl.where(
        tl.load(E + tbe + 3).to(tl.int32) != NLE,
        tl.load(W + tbw + 3).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 4).to(tl.int32)
    d4 = tl.where(v < 0, v + ROWS, v) * sd
    w4 = tl.where(
        tl.load(E + tbe + 4).to(tl.int32) != NLE,
        tl.load(W + tbw + 4).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 5).to(tl.int32)
    d5 = tl.where(v < 0, v + ROWS, v) * sd
    w5 = tl.where(
        tl.load(E + tbe + 5).to(tl.int32) != NLE,
        tl.load(W + tbw + 5).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 6).to(tl.int32)
    d6 = tl.where(v < 0, v + ROWS, v) * sd
    w6 = tl.where(
        tl.load(E + tbe + 6).to(tl.int32) != NLE,
        tl.load(W + tbw + 6).to(tl.float32) * scale,
        0.0,
    )
    v = tl.load(Idx + tbi + 7).to(tl.int32)
    d7 = tl.where(v < 0, v + ROWS, v) * sd
    w7 = tl.where(
        tl.load(E + tbe + 7).to(tl.int32) != NLE,
        tl.load(W + tbw + 7).to(tl.float32) * scale,
        0.0,
    )
    return (d0, d1, d2, d3, d4, d5, d6, d7, w0, w1, w2, w3, w4, w5, w6, w7)


@triton.jit
def _row8_fma(
    D,
    O,
    obase,
    offs,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
):
    r0 = tl.load(D + d0 + offs)
    r1 = tl.load(D + d1 + offs)
    r2 = tl.load(D + d2 + offs)
    r3 = tl.load(D + d3 + offs)
    r4 = tl.load(D + d4 + offs)
    r5 = tl.load(D + d5 + offs)
    r6 = tl.load(D + d6 + offs)
    r7 = tl.load(D + d7 + offs)
    acc = (
        (r0.to(tl.float32) * w0 + r1.to(tl.float32) * w1)
        + (r2.to(tl.float32) * w2 + r3.to(tl.float32) * w3)
    ) + (
        (r4.to(tl.float32) * w4 + r5.to(tl.float32) * w5)
        + (r6.to(tl.float32) * w6 + r7.to(tl.float32) * w7)
    )
    tl.store(O + obase + offs, acc.to(O.dtype.element_ty))


@triton.jit
def _row8_fma_m(
    D,
    O,
    obase,
    offs,
    m,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
):
    r0 = tl.load(D + d0 + offs, m, other=0)
    r1 = tl.load(D + d1 + offs, m, other=0)
    r2 = tl.load(D + d2 + offs, m, other=0)
    r3 = tl.load(D + d3 + offs, m, other=0)
    r4 = tl.load(D + d4 + offs, m, other=0)
    r5 = tl.load(D + d5 + offs, m, other=0)
    r6 = tl.load(D + d6 + offs, m, other=0)
    r7 = tl.load(D + d7 + offs, m, other=0)
    acc = (
        (r0.to(tl.float32) * w0 + r1.to(tl.float32) * w1)
        + (r2.to(tl.float32) * w2 + r3.to(tl.float32) * w3)
    ) + (
        (r4.to(tl.float32) * w4 + r5.to(tl.float32) * w5)
        + (r6.to(tl.float32) * w6 + r7.to(tl.float32) * w7)
    )
    tl.store(O + obase + offs, acc.to(O.dtype.element_ty), m)


@triton.jit
def _reorder_row8(
    D,
    O,
    I,
    E,
    W,
    scale,
    sd,
    so,
    si,
    se,
    sw,
    NLE,
    ROWS,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NF: tl.constexpr,
    TAIL: tl.constexpr,
):
    """topk==8 fast path: one program per token, hoisted slot scalars,
    NF unmasked BLOCK_H column chunks + one masked tail chunk."""
    t = tl.program_id(0)
    (d0, d1, d2, d3, d4, d5, d6, d7, w0, w1, w2, w3, w4, w5, w6, w7) = (
        _row8_slots(I, E, W, t * si, t * se, t * sw, NLE, ROWS, sd, scale)
    )
    obase = t * so
    for j in tl.static_range(NF):
        offs = j * BLOCK_H + tl.arange(0, BLOCK_H)
        _row8_fma(
            D,
            O,
            obase,
            offs,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
        )
    if TAIL:
        offs = NF * BLOCK_H + tl.arange(0, BLOCK_H)
        m = offs < H
        _row8_fma_m(
            D,
            O,
            obase,
            offs,
            m,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
        )


@triton.jit
def _reorder_row8_persist(
    D,
    O,
    I,
    E,
    W,
    scale,
    sd,
    so,
    si,
    se,
    sw,
    NLE,
    ROWS,
    TT,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NF: tl.constexpr,
    TAIL: tl.constexpr,
):
    """topk==8 large-T arm: _PERSIST_NPROG persistent programs walk tokens
    strided; per-token body identical to _reorder_row8 (amortizes the
    52.7ns/CTA dispatch floor once T exceeds one device wave)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, TT, nprog):
        (d0, d1, d2, d3, d4, d5, d6, d7, w0, w1, w2, w3, w4, w5, w6, w7) = (
            _row8_slots(I, E, W, t * si, t * se, t * sw, NLE, ROWS, sd, scale)
        )
        obase = t * so
        for j in tl.static_range(NF):
            offs = j * BLOCK_H + tl.arange(0, BLOCK_H)
            _row8_fma(
                D,
                O,
                obase,
                offs,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                w0,
                w1,
                w2,
                w3,
                w4,
                w5,
                w6,
                w7,
            )
        if TAIL:
            offs = NF * BLOCK_H + tl.arange(0, BLOCK_H)
            m = offs < H
            _row8_fma_m(
                D,
                O,
                obase,
                offs,
                m,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                w0,
                w1,
                w2,
                w3,
                w4,
                w5,
                w6,
                w7,
            )


@triton.jit
def _reorder_row8_colsplit(
    D,
    O,
    I,
    E,
    W,
    scale,
    sd,
    so,
    si,
    se,
    sw,
    NLE,
    ROWS,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NC1: tl.constexpr,
    TAIL: tl.constexpr,
):
    """topk==8 tiny-T arm: one (token, chunk) per program, halving the serial
    per-CTA work when too few tokens exist to fill the device; the tail
    chunk takes the masked helper via a per-program branch."""
    t = tl.program_id(0)
    (d0, d1, d2, d3, d4, d5, d6, d7, w0, w1, w2, w3, w4, w5, w6, w7) = (
        _row8_slots(I, E, W, t * si, t * se, t * sw, NLE, ROWS, sd, scale)
    )
    obase = t * so
    partial = 0
    if TAIL:
        partial = tl.program_id(1) == NC1
    if partial:
        offs = NC1 * BLOCK_H + tl.arange(0, BLOCK_H)
        m = offs < H
        _row8_fma_m(
            D,
            O,
            obase,
            offs,
            m,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
        )
    else:
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        _row8_fma(
            D,
            O,
            obase,
            offs,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
        )


@triton.jit
def _reorder_generic(
    D,
    O,
    I,
    E,
    W,
    scale,
    sd,
    so,
    si,
    se,
    sw,
    T,
    H,
    NLE,
    ROWS,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Every topk except the 8-slot fast path: persistent strided token
    loop (grid respects the 65535 grid.x cap), masked BLOCK_H chunks."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, T, nprog):
        tbi = t * si
        tbe = t * se
        tbw = t * sw
        obase = t * so
        for c0 in range(0, H, BLOCK_H):
            offs = c0 + tl.arange(0, BLOCK_H)
            m = offs < H
            acc = tl.zeros([BLOCK_H], tl.float32)
            for i in tl.static_range(K):
                dv = tl.load(I + tbi + i).to(tl.int32)
                ev = tl.load(E + tbe + i).to(tl.int32)
                wv = tl.load(W + tbw + i).to(tl.float32)
                wv = tl.where(ev != NLE, wv * scale, 0.0)
                row = tl.where(dv < 0, dv + ROWS, dv)
                v = tl.load(D + row * sd + offs, m, other=0)
                acc += v.to(tl.float32) * wv
            tl.store(O + obase + offs, acc.to(O.dtype.element_ty), m)


@triton.jit
def _serial(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    NLE: tl.constexpr,
    ROWS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    B: tl.constexpr,
):
    """Legacy strided fallback (arbitrary column strides, topk > 16)."""
    t = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    for slot in tl.static_range(K):
        row = tl.load(I + t * I0 + slot * I1).to(tl.int32)
        row = tl.where(row < 0, row + ROWS, row)
        eid = tl.load(E + t * E0 + slot * E1)
        w = tl.load(W + t * W0 + slot * W1).to(tl.float32)
        w = tl.where(eid != NLE, w, 0.0)
        v = tl.load(D + row * D0 + col * D1, col < H, other=0).to(tl.float32)
        acc += v * w
    tl.store(O + t * O0 + col * O1, acc * S, col < H)


def _block_for(hidden_size):
    return min(_BLOCK_H, triton.next_power_of_2(hidden_size))


def post_reorder_cutlass(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    out = output.new_empty((num_tokens, hidden_size))
    if num_tokens == 0 or hidden_size == 0:
        return out

    unit_stride = (
        down_output.stride(1) == 1
        and output.stride(1) == 1
        and src2dst.stride(1) == 1
        and topk_ids.stride(1) == 1
        and topk_weights.stride(1) == 1
    )
    if not unit_stride or topk > _MAX_TOPK_UNROLL:
        return _fallback(
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            num_tokens,
            hidden_size,
            routed_scaling_factor,
        )

    scale = float(routed_scaling_factor)
    b = _block_for(hidden_size)
    if topk == 8 and num_tokens <= _MAX_GRID_X and hidden_size <= 65536:
        nf = hidden_size // b
        tail = 1 if hidden_size % b else 0
        strides = (
            down_output.stride(0),
            out.stride(0),
            src2dst.stride(0),
            topk_ids.stride(0),
            topk_weights.stride(0),
            num_local_experts,
            down_output.shape[0],
        )
        if num_tokens <= _COLSPLIT_MAX_T:
            nc = nf + tail
            _reorder_row8_colsplit[(num_tokens, nc)](
                down_output,
                out,
                src2dst,
                topk_ids,
                topk_weights,
                scale,
                *strides,
                hidden_size,
                BLOCK_H=b,
                NC1=nc - 1,
                TAIL=tail,
                num_warps=4,
                num_stages=1,
            )
        elif num_tokens < _PERSIST_MIN_T:
            _reorder_row8[(num_tokens,)](
                down_output,
                out,
                src2dst,
                topk_ids,
                topk_weights,
                scale,
                *strides,
                hidden_size,
                BLOCK_H=b,
                NF=nf,
                TAIL=tail,
                num_warps=4,
                num_stages=1,
            )
        else:
            _reorder_row8_persist[(_PERSIST_NPROG,)](
                down_output,
                out,
                src2dst,
                topk_ids,
                topk_weights,
                scale,
                *strides,
                num_tokens,
                hidden_size,
                BLOCK_H=b,
                NF=nf,
                TAIL=tail,
                num_warps=4,
                num_stages=1,
            )
        return out

    _reorder_generic[(min(num_tokens, _MAX_GRID_X),)](
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        scale,
        down_output.stride(0),
        out.stride(0),
        src2dst.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        num_tokens,
        hidden_size,
        num_local_experts,
        down_output.shape[0],
        K=topk,
        BLOCK_H=b,
        num_warps=4,
        num_stages=1,
    )
    return out


def _fallback(
    down_output,
    out,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    if num_tokens == 0 or hidden_size == 0:
        return out
    b = 2048
    while b > 128 and num_tokens * ((hidden_size + b - 1) // b) < 1024:
        b //= 2
    _serial[(num_tokens, (hidden_size + b - 1) // b)](
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        hidden_size,
        topk,
        float(routed_scaling_factor),
        num_local_experts,
        down_output.shape[0],
        *down_output.stride(),
        *out.stride(),
        *src2dst.stride(),
        *topk_ids.stride(),
        *topk_weights.stride(),
        b,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["post_reorder_cutlass"]
