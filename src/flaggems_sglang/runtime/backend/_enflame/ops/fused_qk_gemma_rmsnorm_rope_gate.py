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

# GCU notes: the upstream 2D (heads x dim) tile kernels hit the
# memref_ext.slice_pad_start 8-vs-7 miscompile family on the 2026-11-07
# toolchain (see compiler_issues/sglang_qk_head_layernorm.md), so this
# copy is rewritten as strictly-1D affine row streams with a persistent
# grid-stride loop, mirroring the delivered qk_head_layernorm rewrite.
#
# The row-batched arm gathers pos/cos/sin per (row, col) element; the
# delivered sibling qk_gemma_rmsnorm_rope measured that data-dependent
# per-row gather at ~85% of kernel time at t=512, and its token-aligned
# arms (scalar pos + affine [HALF] cos/sin shared across the token's
# head blocks) removed it.  This op mirrors those token arms (flat
# (token, head-block) groups for small t; one fused Q+K group per token
# for large t) with the gate segment copy fused into each Q group.

import torch
import triton
import triton.language as tl

# Rows per static_range group: dispatch cost amortization vs tail waste.
_ROWS = 8
# Persistent grid cap (multiples of the 48 SPCs).
_GRID_TARGET = 48
_WARPS = 1
# Token-arm dispatch line: at >= this many tokens the fused per-token
# groups (shared pos/cos/sin for Q+K, no loop-carried small loads) beat
# the flat (token, head-block) groups; below it the flat form fills more
# SPCs.
_TOKEN_FUSED_MIN_T = 48
# num_warps for the token arms (swept independently of the row-stream /
# batch arms: family precedent has the loop row-block arm sweet spot at
# nw2 while the 1D batch arms stay flat at nw1).
_WARPS_TOKEN = 1


@triton.jit
def _gemma_rope_row(
    X,
    C,
    P,
    Y,
    G,
    row,
    w1,
    w2,
    wt,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # One (token, head) row, strictly 1D affine accesses.
    token = row // H
    head = row % H
    xbase = token * XS0 + head * D * (1 + GATE) * XS1
    out = row * D
    HALF: tl.constexpr = RD // 2
    p = tl.arange(0, BP)
    EVENP: tl.constexpr = BP == HALF
    if EVENP:
        x1 = tl.load(X + xbase + p * XS1, cache_modifier=".cg").to(tl.float32)
        x2 = tl.load(
            X + xbase + (HALF + p) * XS1, cache_modifier=".cg"
        ).to(tl.float32)
        squares = tl.sum(x1 * x1 + x2 * x2, 0)
    else:
        pm = p < HALF
        x1 = tl.load(
            X + xbase + p * XS1, pm, 0, cache_modifier=".cg"
        ).to(tl.float32)
        x2 = tl.load(
            X + xbase + (HALF + p) * XS1, pm, 0, cache_modifier=".cg"
        ).to(tl.float32)
        squares = tl.sum(x1 * x1 + x2 * x2, 0)
    if RD < D:
        TAIL: tl.constexpr = D - RD
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            xt = tl.load(X + xbase + t * XS1, cache_modifier=".cg").to(
                tl.float32
            )
            squares += tl.sum(xt * xt, 0)
        else:
            tm = t < D
            xt = tl.load(
                X + xbase + t * XS1, tm, 0, cache_modifier=".cg"
            ).to(tl.float32)
            squares += tl.sum(xt * xt, 0)
    inv = tl.rsqrt(squares / D + EPS)
    if RD > 0:
        if EVENP:
            n1 = (
                (x1 * inv * (w1 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            )
            n2 = (
                (x2 * inv * (w2 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            )
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + p * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + p) * CS1).to(tl.float32)
            tl.store(Y + out + p, n1 * cos - n2 * sin)
            tl.store(Y + out + HALF + p, n2 * cos + n1 * sin)
        else:
            pm = p < HALF
            n1 = (
                (x1 * inv * (w1 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            )
            n2 = (
                (x2 * inv * (w2 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            )
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + p * CS1, pm, 0).to(tl.float32)
            sin = tl.load(
                C + pos * CS0 + (HALF + p) * CS1, pm, 0
            ).to(tl.float32)
            tl.store(Y + out + p, n1 * cos - n2 * sin, pm)
            tl.store(Y + out + HALF + p, n2 * cos + n1 * sin, pm)
    if RD < D:
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            nt = (xt * inv * (wt + 1.0)).to(Y.dtype.element_ty)
            tl.store(Y + out + t, nt)
        else:
            tm = t < D
            nt = (xt * inv * (wt + 1.0)).to(Y.dtype.element_ty)
            tl.store(Y + out + t, nt, tm)
    if GATE:
        d = tl.arange(0, BD)
        if BD == D:
            gate = tl.load(
                X + xbase + (D + d) * XS1, cache_modifier=".cg"
            )
            tl.store(G + out + d, gate)
        else:
            dm = d < D
            gate = tl.load(
                X + xbase + (D + d) * XS1, dm, 0, cache_modifier=".cg"
            )
            tl.store(G + out + d, gate, dm)


@triton.jit
def _fused_qk_gemma_rmsnorm_rope_gate_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    NQ_ROWS,
    NK_ROWS,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Loop-invariant weights: loaded once per program instead of per row.
    HALF: tl.constexpr = RD // 2
    p = tl.arange(0, BP)
    EVENP: tl.constexpr = BP == HALF
    if EVENP:
        qw1 = tl.load(QW + p * QWS).to(tl.float32)
        qw2 = tl.load(QW + (HALF + p) * QWS).to(tl.float32)
        kw1 = tl.load(KW + p * KWS).to(tl.float32)
        kw2 = tl.load(KW + (HALF + p) * KWS).to(tl.float32)
    else:
        pm = p < HALF
        qw1 = tl.load(QW + p * QWS, pm, 0).to(tl.float32)
        qw2 = tl.load(QW + (HALF + p) * QWS, pm, 0).to(tl.float32)
        kw1 = tl.load(KW + p * KWS, pm, 0).to(tl.float32)
        kw2 = tl.load(KW + (HALF + p) * KWS, pm, 0).to(tl.float32)
    qwt = 0.0
    kwt = 0.0
    if RD < D:
        TAIL: tl.constexpr = D - RD
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            qwt = tl.load(QW + t * QWS).to(tl.float32)
            kwt = tl.load(KW + t * KWS).to(tl.float32)
        else:
            tm = t < D
            qwt = tl.load(QW + t * QWS, tm, 0).to(tl.float32)
            kwt = tl.load(KW + t * KWS, tm, 0).to(tl.float32)
    # Persistent grid-stride over flat rows: [0, NQ_ROWS) are Q rows,
    # [NQ_ROWS, NQ_ROWS + NK_ROWS) are K rows.
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    total = NQ_ROWS + NK_ROWS
    for grp in tl.range(pid, tl.cdiv(total, ROWS), npid):
        base = grp * ROWS
        for r in tl.static_range(ROWS):
            row = base + r
            if row < NQ_ROWS:
                _gemma_rope_row(
                    Q, C, P, QO, GO, row, qw1, qw2, qwt,
                    NQ, D, RD, EPS, QS0, QS1, CS0, CS1, PS,
                    GATE, BP, BT, BD,
                )
            elif row < total:
                _gemma_rope_row(
                    K, C, P, KO, GO, row - NQ_ROWS, kw1, kw2, kwt,
                    NK, D, RD, EPS, KS0, KS1, CS0, CS1, PS,
                    False, BP, BT, BD,
                )


@triton.jit
def _gate_q_group(
    Q,
    QW,
    C,
    P,
    QO,
    GO,
    row0,
    R: tl.constexpr,
    D: tl.constexpr,
    NQ: tl.constexpr,
    QWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
):
    # R contiguous Q rows ([norm D | gate D] each), all-1D gather access,
    # batched (R, HALF) reduction tree in registers.
    RW: tl.constexpr = 2 * D
    j = tl.arange(0, R * HALF)
    r = j // HALF
    p = j % HALF
    base = (row0 + r) * RW
    x1 = tl.load(Q + base + p, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(Q + base + HALF + p, cache_modifier=".cg").to(tl.float32)
    x1r = tl.reshape(x1, (R, HALF))
    x2r = tl.reshape(x2, (R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 1)
    inv = tl.rsqrt(squares / D + EPS)
    pw = tl.arange(0, HALF)
    w1 = tl.load(QW + pw * QWS).to(tl.float32)
    w2 = tl.load(QW + (HALF + pw) * QWS).to(tl.float32)
    n1 = (
        (x1r * inv[:, None] * (w1[None, :] + 1.0))
        .to(QO.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, None] * (w2[None, :] + 1.0))
        .to(QO.dtype.element_ty)
        .to(tl.float32)
    )
    pos = tl.load(P + ((row0 + r) // NQ) * PS)
    cos = tl.load(C + pos * CS0 + p * CS1).to(tl.float32)
    sin = tl.load(C + pos * CS0 + (HALF + p) * CS1).to(tl.float32)
    cosr = tl.reshape(cos, (R, HALF))
    sinr = tl.reshape(sin, (R, HALF))
    o1 = n1 * cosr - n2 * sinr
    o2 = n2 * cosr + n1 * sinr
    obase = (row0 + r) * D
    tl.store(QO + obase + p, tl.reshape(o1, (R * HALF,)), cache_modifier=".cg")
    tl.store(
        QO + obase + HALF + p, tl.reshape(o2, (R * HALF,)), cache_modifier=".cg"
    )
    jg = tl.arange(0, R * D)
    rg = jg // D
    cg = jg % D
    gate = tl.load(Q + (row0 + rg) * RW + D + cg, cache_modifier=".cg")
    tl.store(GO + (row0 + rg) * D + cg, gate, cache_modifier=".cg")


@triton.jit
def _norm_group(
    X,
    W,
    C,
    P,
    Y,
    row0,
    R: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
):
    # R contiguous rows of width D (K rows, or gate-less Q rows).
    j = tl.arange(0, R * HALF)
    r = j // HALF
    p = j % HALF
    base = (row0 + r) * D
    x1 = tl.load(X + base + p, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(X + base + HALF + p, cache_modifier=".cg").to(tl.float32)
    x1r = tl.reshape(x1, (R, HALF))
    x2r = tl.reshape(x2, (R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 1)
    inv = tl.rsqrt(squares / D + EPS)
    pw = tl.arange(0, HALF)
    w1 = tl.load(W + pw * WS).to(tl.float32)
    w2 = tl.load(W + (HALF + pw) * WS).to(tl.float32)
    n1 = (
        (x1r * inv[:, None] * (w1[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, None] * (w2[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    pos = tl.load(P + ((row0 + r) // NH) * PS)
    cos = tl.load(C + pos * CS0 + p * CS1).to(tl.float32)
    sin = tl.load(C + pos * CS0 + (HALF + p) * CS1).to(tl.float32)
    cosr = tl.reshape(cos, (R, HALF))
    sinr = tl.reshape(sin, (R, HALF))
    o1 = n1 * cosr - n2 * sinr
    o2 = n2 * cosr + n1 * sinr
    obase = (row0 + r) * D
    tl.store(Y + obase + p, tl.reshape(o1, (R * HALF,)), cache_modifier=".cg")
    tl.store(
        Y + obase + HALF + p, tl.reshape(o2, (R * HALF,)), cache_modifier=".cg"
    )


@triton.jit
def _fused_qk_gemma_batch_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    NQ_ROWS,
    NK_ROWS,
    Q_GROUPS,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
):
    # Dispatch: group axis 0 = [Q groups | K groups]; host guarantees
    # RQ | NQ_ROWS and RK | NK_ROWS so every group is full (no masks).
    HALF: tl.constexpr = D // 2
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    n_groups = Q_GROUPS + tl.cdiv(NK_ROWS, RK)
    for grp in tl.range(pid, n_groups, npid):
        if grp < Q_GROUPS:
            row0 = grp * RQ
            if GATE:
                _gate_q_group(
                    Q, QW, C, P, QO, GO, row0,
                    RQ, D, NQ, QWS, CS0, CS1, PS, HALF, EPS,
                )
            else:
                _norm_group(
                    Q, QW, C, P, QO, row0,
                    RQ, D, NQ, QWS, CS0, CS1, PS, HALF, EPS,
                )
        else:
            row0 = (grp - Q_GROUPS) * RK
            _norm_group(
                K, KW, C, P, KO, row0,
                RK, D, NK, KWS, CS0, CS1, PS, HALF, EPS,
            )


@triton.jit
def _gate_rope_group(
    Q,
    QO,
    GO,
    token,
    h0,
    w1,
    w2,
    cos,
    sin,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    GATE: tl.constexpr,
):
    # R heads of one token: 1D div/mod x loads with HALF-lane contiguous
    # segments, register reshape for the batched axis=1 tree, flat stores;
    # the gate segment ([head][D | gate D] layout) is copied when GATE.
    j = tl.arange(0, R * HALF)
    h = j // HALF
    p = j % HALF
    RW: tl.constexpr = D * (1 + GATE)
    base = token * XS0 + (h0 + h) * RW + p
    x1 = tl.load(Q + base, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(Q + base + HALF, cache_modifier=".cg").to(tl.float32)
    x1r = tl.reshape(x1, (R, HALF))
    x2r = tl.reshape(x2, (R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 1)
    inv = tl.rsqrt(squares / D + EPS)
    n1 = (
        (x1r * inv[:, None] * (w1[None, :] + 1.0))
        .to(QO.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, None] * (w2[None, :] + 1.0))
        .to(QO.dtype.element_ty)
        .to(tl.float32)
    )
    o1 = n1 * cos[None, :] - n2 * sin[None, :]
    o2 = n2 * cos[None, :] + n1 * sin[None, :]
    out = (token * H + h0 + h) * D + p
    tl.store(QO + out, tl.reshape(o1, (R * HALF,)), cache_modifier=".cg")
    tl.store(
        QO + out + HALF, tl.reshape(o2, (R * HALF,)), cache_modifier=".cg"
    )
    if GATE:
        jg = tl.arange(0, R * D)
        hg = jg // D
        cg = jg % D
        gsrc = token * XS0 + (h0 + hg) * RW + D + cg
        gate = tl.load(Q + gsrc, cache_modifier=".cg")
        gout = (token * H + h0 + hg) * D + cg
        tl.store(GO + gout, gate, cache_modifier=".cg")


@triton.jit
def _fused_qk_gemma_gate_token_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    T,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    KS0: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
):
    # Flat (token, head-block) groups: [0, T*QHB) are Q groups, the rest
    # are K groups; host guarantees RQ | NQ and RK | NK so every group is
    # full. Weights are loop-invariant; pos/cos/sin are one scalar plus
    # two affine [HALF] loads per group.
    HALF: tl.constexpr = D // 2
    pw = tl.arange(0, HALF)
    qw1 = tl.load(QW + pw * QWS).to(tl.float32)
    qw2 = tl.load(QW + (HALF + pw) * QWS).to(tl.float32)
    kw1 = tl.load(KW + pw * KWS).to(tl.float32)
    kw2 = tl.load(KW + (HALF + pw) * KWS).to(tl.float32)
    QHB: tl.constexpr = NQ // RQ
    KHB: tl.constexpr = NK // RK
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    q_groups = T * QHB
    n_groups = q_groups + T * KHB
    for grp in tl.range(pid, n_groups, npid):
        if grp < q_groups:
            token = grp // QHB
            h0 = (grp % QHB) * RQ
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            _gate_rope_group(
                Q, QO, GO, token, h0, qw1, qw2, cos, sin,
                RQ, NQ, D, HALF, EPS, QS0, GATE,
            )
        else:
            g2 = grp - q_groups
            token = g2 // KHB
            h0 = (g2 % KHB) * RK
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            _gate_rope_group(
                K, KO, GO, token, h0, kw1, kw2, cos, sin,
                RK, NK, D, HALF, EPS, KS0, False,
            )


@triton.jit
def _fused_qk_gemma_gate_tokf_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    T,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    KS0: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
):
    # One fused Q+K group per token: pos/cos/sin are loaded once per
    # token and shared by the Q and K head-blocks (static_range over the
    # head-blocks keeps everything in one straight-line body).
    HALF: tl.constexpr = D // 2
    pw = tl.arange(0, HALF)
    qw1 = tl.load(QW + pw * QWS).to(tl.float32)
    qw2 = tl.load(QW + (HALF + pw) * QWS).to(tl.float32)
    kw1 = tl.load(KW + pw * KWS).to(tl.float32)
    kw2 = tl.load(KW + (HALF + pw) * KWS).to(tl.float32)
    QHB: tl.constexpr = NQ // RQ
    KHB: tl.constexpr = NK // RK
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    for token in tl.range(pid, T, npid):
        pos = tl.load(P + token * PS)
        cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
        sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
        for hb in tl.static_range(QHB):
            _gate_rope_group(
                Q, QO, GO, token, hb * RQ, qw1, qw2, cos, sin,
                RQ, NQ, D, HALF, EPS, QS0, GATE,
            )
        for hb in tl.static_range(KHB):
            _gate_rope_group(
                K, KO, GO, token, hb * RK, kw1, kw2, cos, sin,
                RK, NK, D, HALF, EPS, KS0, False,
            )


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    t = q_gate.shape[0]
    q_out = torch.empty(
        (t, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (t, num_kv_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    gate_out = (
        torch.empty(
            (t, num_q_heads, head_dim),
            dtype=q_gate.dtype,
            device=q_gate.device,
        )
        if has_gate
        else None
    )
    nq_rows = t * num_q_heads
    nk_rows = t * num_kv_heads
    total_rows = nq_rows + nk_rows
    half = rotary_dim // 2
    # Token-arm conditions (mirrors the delivered qk_gemma_rmsnorm_rope
    # rewrite): neox-free full-RoPE pairing, power-of-two head dim / half,
    # bf16, contiguous inputs, enough rows to amortize, and head counts
    # divisible by the group sizes (full groups). fp16 falls back to the
    # row-stream arm: the register reshape path has a recorded
    # silent-wrong-value precedent on fp16.
    use_token = (
        rotary_dim == head_dim
        and head_dim > 1
        and (head_dim & (head_dim - 1)) == 0
        and half > 0
        and (half & (half - 1)) == 0
        and q_gate.dtype == torch.bfloat16
        and q_gate.is_contiguous()
        and k.is_contiguous()
        and total_rows >= 128
    )
    if use_token:
        if t >= _TOKEN_FUSED_MIN_T:
            rq = min(32, num_q_heads & (-num_q_heads))
            rk = min(8, num_kv_heads & (-num_kv_heads))
        else:
            rq = min(8, num_q_heads & (-num_q_heads))
            rk = min(8, num_kv_heads & (-num_kv_heads))
        use_token = rq >= 4 and rk >= 4
    if use_token:
        if t >= _TOKEN_FUSED_MIN_T:
            grid = (min(t, _GRID_TARGET),)
            _fused_qk_gemma_gate_tokf_kernel[grid](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                t,
                NQ=num_q_heads,
                NK=num_kv_heads,
                D=head_dim,
                EPS=eps,
                QS0=q_gate.stride(0),
                KS0=k.stride(0),
                QWS=q_weight.stride(0),
                KWS=k_weight.stride(0),
                CS0=cos_sin_cache.stride(0),
                CS1=cos_sin_cache.stride(1),
                PS=positions.stride(0),
                GATE=has_gate,
                RQ=rq,
                RK=rk,
                num_warps=_WARPS_TOKEN,
                num_stages=1,
                enable_fp_fusion=False,
            )
        else:
            n_groups = t * (num_q_heads // rq) + t * (num_kv_heads // rk)
            grid = (min(n_groups, _GRID_TARGET),)
            _fused_qk_gemma_gate_token_kernel[grid](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                t,
                NQ=num_q_heads,
                NK=num_kv_heads,
                D=head_dim,
                EPS=eps,
                QS0=q_gate.stride(0),
                KS0=k.stride(0),
                QWS=q_weight.stride(0),
                KWS=k_weight.stride(0),
                CS0=cos_sin_cache.stride(0),
                CS1=cos_sin_cache.stride(1),
                PS=positions.stride(0),
                GATE=has_gate,
                RQ=rq,
                RK=rk,
                num_warps=_WARPS_TOKEN,
                num_stages=1,
                enable_fp_fusion=False,
            )
        return (q_out, k_out, gate_out)
    # Batch arm conditions: RoPE covers the full head dim, power-of-two
    # head dim, bf16, contiguous inputs (flat-row layout), enough rows to
    # amortize, and row counts divisible by the group sizes (tail-free).
    # fp16 falls back to the row-stream arm: the (R, W) register reshape
    # has a recorded silent-wrong-value precedent on fp16.
    use_batch = (
        rotary_dim == head_dim
        and head_dim > 0
        and (head_dim & (head_dim - 1)) == 0
        and q_gate.dtype == torch.bfloat16
        and q_gate.is_contiguous()
        and k.is_contiguous()
        and total_rows >= 128
    )
    if use_batch:
        rq = min(
            8 if nq_rows <= 256 else (32 if nq_rows <= 4096 else 64),
            nq_rows & (-nq_rows),
        )
        rk = min(8 if nk_rows <= 256 else 32, nk_rows & (-nk_rows))
        use_batch = rq >= 4 and rk >= 4
    if use_batch:
        q_groups = nq_rows // rq
        k_groups = nk_rows // rk
        grid = (min(q_groups + k_groups, _GRID_TARGET),)
        _fused_qk_gemma_batch_kernel[grid](
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q_out,
            k_out,
            gate_out if has_gate else q_out,
            nq_rows,
            nk_rows,
            q_groups,
            NQ=num_q_heads,
            NK=num_kv_heads,
            D=head_dim,
            EPS=eps,
            QWS=q_weight.stride(0),
            KWS=k_weight.stride(0),
            CS0=cos_sin_cache.stride(0),
            CS1=cos_sin_cache.stride(1),
            PS=positions.stride(0),
            GATE=has_gate,
            RQ=rq,
            RK=rk,
            num_warps=_WARPS,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return (q_out, k_out, gate_out)
    rows = max(1, min(_ROWS, (total_rows + _GRID_TARGET - 1) // _GRID_TARGET))
    n_groups = triton.cdiv(nq_rows + nk_rows, rows)
    grid = (min(n_groups, _GRID_TARGET),)
    _fused_qk_gemma_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_out,
        k_out,
        gate_out if has_gate else q_out,
        nq_rows,
        nk_rows,
        NQ=num_q_heads,
        NK=num_kv_heads,
        D=head_dim,
        RD=rotary_dim,
        EPS=eps,
        QS0=q_gate.stride(0),
        QS1=q_gate.stride(1),
        KS0=k.stride(0),
        KS1=k.stride(1),
        QWS=q_weight.stride(0),
        KWS=k_weight.stride(0),
        CS0=cos_sin_cache.stride(0),
        CS1=cos_sin_cache.stride(1),
        PS=positions.stride(0),
        GATE=has_gate,
        BP=triton.next_power_of_2(max(1, rotary_dim // 2)),
        BT=triton.next_power_of_2(max(1, head_dim - rotary_dim)),
        BD=triton.next_power_of_2(head_dim),
        ROWS=rows,
        num_warps=_WARPS,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (q_out, k_out, gate_out)


__all__ = ["fused_qk_gemma_rmsnorm_rope_gate"]
