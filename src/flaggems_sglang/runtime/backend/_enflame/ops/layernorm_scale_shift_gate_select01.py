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

import torch
import triton
import triton.language as tl


@triton.jit
def _crow(
    X,
    W,
    Bias,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    row,
    b,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HW: tl.constexpr,
    HB: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    m = c < C
    v = tl.load(X + row * C + c, m, other=0).to(tl.float32)
    mean = tl.sum(v, 0) / C
    centered = tl.where(m, v - mean, 0.0)
    var = tl.sum(centered * centered, 0) / C
    z = centered * tl.rsqrt(var + EPS)
    if HW:
        z = z * tl.load(W + c, m, other=0).to(tl.float32)
    if HB:
        z = z + tl.load(Bias + c, m, other=0).to(tl.float32)
    sel = tl.load(I + row) != 0
    if sel:
        scale = tl.load(S1 + b * C + c, m, other=0).to(tl.float32)
        shift = tl.load(T1 + b * C + c, m, other=0).to(tl.float32)
        gate = tl.load(G1 + b * C + c, m, other=0).to(tl.float32)
    else:
        scale = tl.load(S0 + b * C + c, m, other=0).to(tl.float32)
        shift = tl.load(T0 + b * C + c, m, other=0).to(tl.float32)
        gate = tl.load(G0 + b * C + c, m, other=0).to(tl.float32)
    tl.store(Y + row * C + c, z * (1.0 + scale) + shift, m)
    tl.store(G + row * C + c, gate, m)


@triton.jit
def _plain(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    LC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    _crow(
        X,
        X,
        X,
        S0,
        T0,
        G0,
        S1,
        T1,
        G1,
        I,
        Y,
        G,
        row,
        row // (LC >> 16),
        LC & 65535,
        EPS,
        BLOCK,
        False,
        False,
    )


@triton.jit
def _plain_many(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    M: tl.constexpr,
    LC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        _crow(
            X,
            X,
            X,
            S0,
            T0,
            G0,
            S1,
            T1,
            G1,
            I,
            Y,
            G,
            row,
            row // (LC >> 16),
            LC & 65535,
            EPS,
            BLOCK,
            False,
            False,
        )


@triton.jit
def _affine(
    X,
    W,
    Bias,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    M: tl.constexpr,
    LC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HW: tl.constexpr,
    HB: tl.constexpr,
):
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        _crow(
            X,
            W,
            Bias,
            S0,
            T0,
            G0,
            S1,
            T1,
            G1,
            I,
            Y,
            G,
            row,
            row // (LC >> 16),
            LC & 65535,
            EPS,
            BLOCK,
            HW,
            HB,
        )


@triton.jit
def _norm(
    X,
    W,
    Bias,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    M: tl.constexpr,
    L: tl.constexpr,
    C: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    WS: tl.constexpr,
    BS: tl.constexpr,
    HW: tl.constexpr,
    HB: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    m = c < C
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        b = row // L
        token_idx = row % L
        v = tl.load(X + b * XS0 + token_idx * XS1 + c * XS2, m, other=0).to(
            tl.float32
        )
        mean = tl.sum(v, 0) / C
        centered = tl.where(m, v - mean, 0.0)
        var = tl.sum(centered * centered, 0) / C
        z = centered * tl.rsqrt(var + EPS)
        if HW:
            z = z * tl.load(W + c * WS, m, other=0).to(tl.float32)
        if HB:
            z = z + tl.load(Bias + c * BS, m, other=0).to(tl.float32)
        sel = tl.load(I + b * IS0 + token_idx * IS1) != 0
        if sel:
            scale = tl.load(S1 + b * D0 + c * D1, m, other=0).to(tl.float32)
            shift = tl.load(T1 + b * E0 + c * E1, m, other=0).to(tl.float32)
            gate = tl.load(G1 + b * F0 + c * F1, m, other=0).to(tl.float32)
        else:
            scale = tl.load(S0 + b * A0 + c * A1, m, other=0).to(tl.float32)
            shift = tl.load(T0 + b * B0 + c * B1, m, other=0).to(tl.float32)
            gate = tl.load(G0 + b * C0 + c * C1, m, other=0).to(tl.float32)
        tl.store(Y + row * C + c, z * (1.0 + scale) + shift, m)
        tl.store(G + row * C + c, gate, m)


@triton.jit
def n_kernel(
    X,
    W,
    BIAS,
    S0,
    H0,
    G0,
    S1,
    H1,
    G1,
    IDX,
    Y,
    GO,
    L: tl.constexpr,
    C: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    W0: tl.constexpr,
    B0: tl.constexpr,
    S00: tl.constexpr,
    S01: tl.constexpr,
    H00: tl.constexpr,
    H01: tl.constexpr,
    G00: tl.constexpr,
    G01: tl.constexpr,
    S10: tl.constexpr,
    S11: tl.constexpr,
    H10: tl.constexpr,
    H11: tl.constexpr,
    G10: tl.constexpr,
    G11: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // L
    token_idx = row % L
    col = tl.arange(0, BLOCK)
    mask = col < C
    x = tl.load(X + b * X0 + token_idx * X1 + col * X2, mask, 0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    z = tl.where(mask, x - mean, 0.0)
    var = tl.sum(z * z, 0) / C
    n = z * tl.rsqrt(var + EPS)
    if HAS_W:
        n = n * tl.load(W + col * W0, mask, 0).to(tl.float32)
    if HAS_B:
        n = n + tl.load(BIAS + col * B0, mask, 0).to(tl.float32)
    pick = tl.load(IDX + b * I0 + token_idx * I1) != 0
    s0 = tl.load(S0 + b * S00 + col * S01, mask, 0).to(tl.float32)
    h0 = tl.load(H0 + b * H00 + col * H01, mask, 0).to(tl.float32)
    s1 = tl.load(S1 + b * S10 + col * S11, mask, 0).to(tl.float32)
    h1 = tl.load(H1 + b * H10 + col * H11, mask, 0).to(tl.float32)
    scale = tl.where(pick, s1, s0)
    shift = tl.where(pick, h1, h0)
    y = n * (1.0 + scale) + shift
    out = row * C + col
    tl.store(Y + out, y, mask)


@triton.jit
def g_kernel(
    G0,
    G1,
    IDX,
    GO,
    L: tl.constexpr,
    C: tl.constexpr,
    G00: tl.constexpr,
    G01: tl.constexpr,
    G10: tl.constexpr,
    G11: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1)
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = at < L * C
    row = at // C
    col = at % C
    pick = tl.load(IDX + b * I0 + row * I1, mask, 0) != 0
    left = tl.load(G0 + b * G00 + col * G01, mask, 0).to(tl.float32)
    right = tl.load(G1 + b * G10 + col * G11, mask, 0).to(tl.float32)
    value = tl.where(pick, right, left)
    tl.store(GO + b * L * C + at, value, mask)


def _separate(
    x,
    weight,
    bias,
    scale0,
    shift0,
    gate0,
    scale1,
    shift1,
    gate1,
    index,
    eps,
    warps,
    gblock,
):
    b, l, c = x.shape
    y = torch.empty((b, l, c), device=x.device, dtype=x.dtype)
    go = torch.empty((b, l, c), device=x.device, dtype=x.dtype)
    block = triton.next_power_of_2(c)
    w = x if weight is None else weight
    bi = x if bias is None else bias
    args = (
        x,
        w,
        bi,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        y,
        go,
        l,
        c,
        *x.stride(),
        w.stride(-1),
        bi.stride(-1),
        *scale0.stride(),
        *shift0.stride(),
        *gate0.stride(),
        *scale1.stride(),
        *shift1.stride(),
        *gate1.stride(),
        *index.stride(),
        eps,
        block,
        weight is not None,
        bias is not None,
    )
    n_kernel[(b * l, 1, 1)](*args, num_warps=warps, enable_fp_fusion=False)
    gargs = (
        gate0,
        gate1,
        index,
        go,
        l,
        c,
        *gate0.stride(),
        *gate1.stride(),
        *index.stride(),
        gblock,
    )
    g_kernel[(triton.cdiv(l * c, gblock), b, 1)](
        *gargs, num_warps=4, enable_fp_fusion=False
    )
    return (y, go)


@triton.jit
def _dual(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    LC: tl.constexpr,
    EPS: tl.constexpr,
    MAIN: tl.constexpr,
    TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    cc = LC & 65535
    b = row // (LC >> 16)
    c0 = tl.arange(0, MAIN)
    c1 = MAIN + tl.arange(0, TAIL)
    m1 = c1 < cc
    sel = tl.load(I + row) != 0
    if tl.program_id(1) == 0:
        v0 = tl.load(X + row * cc + c0).to(tl.float32)
        v1 = tl.load(X + row * cc + c1, m1, other=0).to(tl.float32)
        mean = (tl.sum(v0, 0) + tl.sum(v1, 0)) / cc
        z0 = v0 - mean
        z1 = tl.where(m1, v1 - mean, 0.0)
        inv = tl.rsqrt((tl.sum(z0 * z0, 0) + tl.sum(z1 * z1, 0)) / cc + EPS)
        n0 = z0 * inv
        n1 = z1 * inv
        if sel:
            tl.store(
                Y + row * cc + c0,
                n0 * (1.0 + tl.load(S1 + b * cc + c0).to(tl.float32))
                + tl.load(T1 + b * cc + c0).to(tl.float32),
            )
            tl.store(
                Y + row * cc + c1,
                n1
                * (1.0 + tl.load(S1 + b * cc + c1, m1, other=0).to(tl.float32))
                + tl.load(T1 + b * cc + c1, m1, other=0).to(tl.float32),
                m1,
            )
        else:
            tl.store(
                Y + row * cc + c0,
                n0 * (1.0 + tl.load(S0 + b * cc + c0).to(tl.float32))
                + tl.load(T0 + b * cc + c0).to(tl.float32),
            )
            tl.store(
                Y + row * cc + c1,
                n1
                * (1.0 + tl.load(S0 + b * cc + c1, m1, other=0).to(tl.float32))
                + tl.load(T0 + b * cc + c1, m1, other=0).to(tl.float32),
                m1,
            )
    else:
        if sel:
            tl.store(
                G + row * cc + c0, tl.load(G1 + b * cc + c0).to(tl.float32)
            )
            tl.store(
                G + row * cc + c1,
                tl.load(G1 + b * cc + c1, m1, other=0).to(tl.float32),
                m1,
            )
        else:
            tl.store(
                G + row * cc + c0, tl.load(G0 + b * cc + c0).to(tl.float32)
            )
            tl.store(
                G + row * cc + c1,
                tl.load(G0 + b * cc + c1, m1, other=0).to(tl.float32),
                m1,
            )


def _contiguous(
    x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index
):
    return (
        x.is_contiguous()
        and index.is_contiguous()
        and scale0.is_contiguous()
        and shift0.is_contiguous()
        and gate0.is_contiguous()
        and scale1.is_contiguous()
        and shift1.is_contiguous()
        and gate1.is_contiguous()
        and (weight is None or weight.is_contiguous())
        and (bias is None or bias.is_contiguous())
    )


@triton.jit
def _dual_tabs(S0, T0, G0, S1, T1, G1, b, cc, c0, c1, m1):
    s1a = tl.load(S1 + b * cc + c0).to(tl.float32)
    t1a = tl.load(T1 + b * cc + c0).to(tl.float32)
    g1a = tl.load(G1 + b * cc + c0).to(tl.float32)
    s1b = tl.load(S1 + b * cc + c1, m1, other=0).to(tl.float32)
    t1b = tl.load(T1 + b * cc + c1, m1, other=0).to(tl.float32)
    g1b = tl.load(G1 + b * cc + c1, m1, other=0).to(tl.float32)
    s0a = tl.load(S0 + b * cc + c0).to(tl.float32)
    t0a = tl.load(T0 + b * cc + c0).to(tl.float32)
    g0a = tl.load(G0 + b * cc + c0).to(tl.float32)
    s0b = tl.load(S0 + b * cc + c1, m1, other=0).to(tl.float32)
    t0b = tl.load(T0 + b * cc + c1, m1, other=0).to(tl.float32)
    g0b = tl.load(G0 + b * cc + c1, m1, other=0).to(tl.float32)
    return s1a, t1a, g1a, s1b, t1b, g1b, s0a, t0a, g0a, s0b, t0b, g0b


@triton.jit
def _dual_row(
    X, Sel, Y, G, row, cc, c0, c1, m1, v0, v1, tabs, EPS: tl.constexpr
):
    (s1a, t1a, g1a, s1b, t1b, g1b, s0a, t0a, g0a, s0b, t0b, g0b) = tabs
    sel = tl.load(Sel + row) != 0
    mean = (tl.sum(v0, 0) + tl.sum(v1, 0)) / cc
    z0 = v0 - mean
    z1 = tl.where(m1, v1 - mean, 0.0)
    inv = tl.rsqrt((tl.sum(z0 * z0, 0) + tl.sum(z1 * z1, 0)) / cc + EPS)
    n0 = z0 * inv
    n1 = z1 * inv
    sv0 = tl.where(sel, s1a, s0a)
    tv0 = tl.where(sel, t1a, t0a)
    gv0 = tl.where(sel, g1a, g0a)
    sv1 = tl.where(sel, s1b, s0b)
    tv1 = tl.where(sel, t1b, t0b)
    gv1 = tl.where(sel, g1b, g0b)
    tl.store(Y + row * cc + c0, n0 * (1.0 + sv0) + tv0)
    tl.store(Y + row * cc + c1, n1 * (1.0 + sv1) + tv1, m1)
    tl.store(G + row * cc + c0, gv0)
    tl.store(G + row * cc + c1, gv1, m1)


@triton.jit
def _dual_fast(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    L: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    MAIN: tl.constexpr,
    TAIL: tl.constexpr,
    NB: tl.constexpr,
):
    cc = C
    b = tl.program_id(0) // NB
    tb = tl.program_id(0) % NB
    c0 = tl.arange(0, MAIN)
    c1 = MAIN + tl.arange(0, TAIL)
    m1 = c1 < cc
    tabs = _dual_tabs(S0, T0, G0, S1, T1, G1, b, cc, c0, c1, m1)
    base = b * L + tb * 8
    v0_0 = tl.load(X + (base + 0) * cc + c0).to(tl.float32)
    v1_0 = tl.load(X + (base + 0) * cc + c1, m1, other=0).to(tl.float32)
    v0_1 = tl.load(X + (base + 1) * cc + c0).to(tl.float32)
    v1_1 = tl.load(X + (base + 1) * cc + c1, m1, other=0).to(tl.float32)
    v0_2 = tl.load(X + (base + 2) * cc + c0).to(tl.float32)
    v1_2 = tl.load(X + (base + 2) * cc + c1, m1, other=0).to(tl.float32)
    v0_3 = tl.load(X + (base + 3) * cc + c0).to(tl.float32)
    v1_3 = tl.load(X + (base + 3) * cc + c1, m1, other=0).to(tl.float32)
    v0_4 = tl.load(X + (base + 4) * cc + c0).to(tl.float32)
    v1_4 = tl.load(X + (base + 4) * cc + c1, m1, other=0).to(tl.float32)
    v0_5 = tl.load(X + (base + 5) * cc + c0).to(tl.float32)
    v1_5 = tl.load(X + (base + 5) * cc + c1, m1, other=0).to(tl.float32)
    v0_6 = tl.load(X + (base + 6) * cc + c0).to(tl.float32)
    v1_6 = tl.load(X + (base + 6) * cc + c1, m1, other=0).to(tl.float32)
    v0_7 = tl.load(X + (base + 7) * cc + c0).to(tl.float32)
    v1_7 = tl.load(X + (base + 7) * cc + c1, m1, other=0).to(tl.float32)
    _dual_row(X, I, Y, G, base + 0, cc, c0, c1, m1, v0_0, v1_0, tabs, EPS)
    _dual_row(X, I, Y, G, base + 1, cc, c0, c1, m1, v0_1, v1_1, tabs, EPS)
    _dual_row(X, I, Y, G, base + 2, cc, c0, c1, m1, v0_2, v1_2, tabs, EPS)
    _dual_row(X, I, Y, G, base + 3, cc, c0, c1, m1, v0_3, v1_3, tabs, EPS)
    _dual_row(X, I, Y, G, base + 4, cc, c0, c1, m1, v0_4, v1_4, tabs, EPS)
    _dual_row(X, I, Y, G, base + 5, cc, c0, c1, m1, v0_5, v1_5, tabs, EPS)
    _dual_row(X, I, Y, G, base + 6, cc, c0, c1, m1, v0_6, v1_6, tabs, EPS)
    _dual_row(X, I, Y, G, base + 7, cc, c0, c1, m1, v0_7, v1_7, tabs, EPS)


@triton.jit
def _dual_guard(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    L: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    MAIN: tl.constexpr,
    TAIL: tl.constexpr,
    U: tl.constexpr,
    NB: tl.constexpr,
):
    cc = C
    b = tl.program_id(0) // NB
    tb = tl.program_id(0) % NB
    c0 = tl.arange(0, MAIN)
    c1 = MAIN + tl.arange(0, TAIL)
    m1 = c1 < cc
    tabs = _dual_tabs(S0, T0, G0, S1, T1, G1, b, cc, c0, c1, m1)
    for u in tl.static_range(U):
        t = tb * U + u
        if t < L:
            row = b * L + t
            v0 = tl.load(X + row * cc + c0).to(tl.float32)
            v1 = tl.load(X + row * cc + c1, m1, other=0).to(tl.float32)
            _dual_row(X, I, Y, G, row, cc, c0, c1, m1, v0, v1, tabs, EPS)


@triton.jit
def _dual_hyb(
    X,
    S0,
    T0,
    G0,
    S1,
    T1,
    G1,
    I,
    Y,
    G,
    L: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    MAIN: tl.constexpr,
    TAIL: tl.constexpr,
    L8: tl.constexpr,
    NB: tl.constexpr,
):
    cc = C
    b = tl.program_id(0) // NB
    tb = tl.program_id(0) % NB
    c0 = tl.arange(0, MAIN)
    c1 = MAIN + tl.arange(0, TAIL)
    m1 = c1 < cc
    tabs = _dual_tabs(S0, T0, G0, S1, T1, G1, b, cc, c0, c1, m1)
    if tb < L8:
        base = b * L + tb * 8
        v0_0 = tl.load(X + (base + 0) * cc + c0).to(tl.float32)
        v1_0 = tl.load(X + (base + 0) * cc + c1, m1, other=0).to(tl.float32)
        v0_1 = tl.load(X + (base + 1) * cc + c0).to(tl.float32)
        v1_1 = tl.load(X + (base + 1) * cc + c1, m1, other=0).to(tl.float32)
        v0_2 = tl.load(X + (base + 2) * cc + c0).to(tl.float32)
        v1_2 = tl.load(X + (base + 2) * cc + c1, m1, other=0).to(tl.float32)
        v0_3 = tl.load(X + (base + 3) * cc + c0).to(tl.float32)
        v1_3 = tl.load(X + (base + 3) * cc + c1, m1, other=0).to(tl.float32)
        v0_4 = tl.load(X + (base + 4) * cc + c0).to(tl.float32)
        v1_4 = tl.load(X + (base + 4) * cc + c1, m1, other=0).to(tl.float32)
        v0_5 = tl.load(X + (base + 5) * cc + c0).to(tl.float32)
        v1_5 = tl.load(X + (base + 5) * cc + c1, m1, other=0).to(tl.float32)
        v0_6 = tl.load(X + (base + 6) * cc + c0).to(tl.float32)
        v1_6 = tl.load(X + (base + 6) * cc + c1, m1, other=0).to(tl.float32)
        v0_7 = tl.load(X + (base + 7) * cc + c0).to(tl.float32)
        v1_7 = tl.load(X + (base + 7) * cc + c1, m1, other=0).to(tl.float32)
        _dual_row(X, I, Y, G, base + 0, cc, c0, c1, m1, v0_0, v1_0, tabs, EPS)
        _dual_row(X, I, Y, G, base + 1, cc, c0, c1, m1, v0_1, v1_1, tabs, EPS)
        _dual_row(X, I, Y, G, base + 2, cc, c0, c1, m1, v0_2, v1_2, tabs, EPS)
        _dual_row(X, I, Y, G, base + 3, cc, c0, c1, m1, v0_3, v1_3, tabs, EPS)
        _dual_row(X, I, Y, G, base + 4, cc, c0, c1, m1, v0_4, v1_4, tabs, EPS)
        _dual_row(X, I, Y, G, base + 5, cc, c0, c1, m1, v0_5, v1_5, tabs, EPS)
        _dual_row(X, I, Y, G, base + 6, cc, c0, c1, m1, v0_6, v1_6, tabs, EPS)
        _dual_row(X, I, Y, G, base + 7, cc, c0, c1, m1, v0_7, v1_7, tabs, EPS)
    else:
        for u in tl.static_range(8):
            t = L8 * 8 + u
            if t < L:
                row = b * L + t
                v0 = tl.load(X + row * cc + c0).to(tl.float32)
                v1 = tl.load(X + row * cc + c1, m1, other=0).to(tl.float32)
                _dual_row(X, I, Y, G, row, cc, c0, c1, m1, v0, v1, tabs, EPS)


def layernorm_scale_shift_gate_select01(
    x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps
):
    b, l, c = x.shape
    rows = b * l
    if _contiguous(
        x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index
    ):
        out = torch.empty_like(x)
        gate_out = torch.empty_like(x)
        if rows == 0 or c == 0:
            return out, gate_out
        block = triton.next_power_of_2(c)
        warps = 4 if block <= 2048 else 8
        lc = l * 65536 + c
        plain = (
            x,
            scale0,
            shift0,
            gate0,
            scale1,
            shift1,
            gate1,
            index,
            out,
            gate_out,
        )
        if (
            2048 < c
            and c != block
            and weight is None
            and bias is None
            and rows <= 65535
        ):
            if c <= 4096:
                if l % 8 == 0 and rows >= 64:
                    _dual_fast[(b * (l // 8),)](
                        *plain,
                        l,
                        c,
                        float(eps),
                        2048,
                        triton.next_power_of_2(c - 2048),
                        l // 8,
                        num_warps=1,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                elif b * (l // 8) >= 48:
                    l8 = l // 8
                    nb = l8 + (1 if l % 8 else 0)
                    _dual_hyb[(b * nb,)](
                        *plain,
                        l,
                        c,
                        float(eps),
                        2048,
                        triton.next_power_of_2(c - 2048),
                        l8,
                        nb,
                        num_warps=1,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                else:
                    u = 2 if rows <= 96 else (4 if rows <= 384 else 16)
                    nb = (l + u - 1) // u
                    _dual_guard[(b * nb,)](
                        *plain,
                        l,
                        c,
                        float(eps),
                        2048,
                        triton.next_power_of_2(c - 2048),
                        u,
                        nb,
                        num_warps=1,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                return out, gate_out
            _dual[(rows, 2)](
                *plain,
                lc,
                float(eps),
                2048,
                triton.next_power_of_2(c - 2048),
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return out, gate_out
        if weight is None and bias is None:
            if rows <= 65535:
                _plain[(rows,)](
                    *plain,
                    lc,
                    float(eps),
                    block,
                    num_warps=warps,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            else:
                _plain_many[(65535,)](
                    *plain,
                    rows,
                    lc,
                    float(eps),
                    block,
                    num_warps=warps,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            return out, gate_out
        _affine[(min(rows, 65535),)](
            x,
            weight if weight is not None else x,
            bias if bias is not None else x,
            scale0,
            shift0,
            gate0,
            scale1,
            shift1,
            gate1,
            index,
            out,
            gate_out,
            rows,
            lc,
            float(eps),
            block,
            weight is not None,
            bias is not None,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return out, gate_out
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    gate_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out, gate_out
    block = triton.next_power_of_2(c)
    warps = 4 if block <= 2048 else 8
    _norm[(min(rows, 65535),)](
        x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        out,
        gate_out,
        rows,
        l,
        c,
        *x.stride(),
        *scale0.stride(),
        *shift0.stride(),
        *gate0.stride(),
        *scale1.stride(),
        *shift1.stride(),
        *gate1.stride(),
        *index.stride(),
        weight.stride(0) if weight is not None else 0,
        bias.stride(0) if bias is not None else 0,
        weight is not None,
        bias is not None,
        float(eps),
        block,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out, gate_out


__all__ = ["layernorm_scale_shift_gate_select01"]
