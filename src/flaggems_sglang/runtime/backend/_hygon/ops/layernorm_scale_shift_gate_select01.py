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
def _csplit(
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
    b = row // (LC >> 16)
    cc = LC & 65535
    c0 = tl.arange(0, MAIN)
    c1 = MAIN + tl.arange(0, TAIL)
    m1 = c1 < cc
    v0 = tl.load(X + row * cc + c0).to(tl.float32)
    v1 = tl.load(X + row * cc + c1, m1, other=0).to(tl.float32)
    mean = (tl.sum(v0, 0) + tl.sum(v1, 0)) / cc
    z0 = v0 - mean
    z1 = tl.where(m1, v1 - mean, 0.0)
    var = (tl.sum(z0 * z0, 0) + tl.sum(z1 * z1, 0)) / cc
    inv = tl.rsqrt(var + EPS)
    sel = tl.load(I + row) != 0
    n0 = z0 * inv
    n1 = z1 * inv
    if sel:
        tl.store(
            Y + row * cc + c0,
            n0 * (1.0 + tl.load(S1 + b * cc + c0).to(tl.float32))
            + tl.load(T1 + b * cc + c0).to(tl.float32),
        )
        tl.store(G + row * cc + c0, tl.load(G1 + b * cc + c0).to(tl.float32))
        tl.store(
            Y + row * cc + c1,
            n1 * (1.0 + tl.load(S1 + b * cc + c1, m1, other=0).to(tl.float32))
            + tl.load(T1 + b * cc + c1, m1, other=0).to(tl.float32),
            m1,
        )
        tl.store(
            G + row * cc + c1,
            tl.load(G1 + b * cc + c1, m1, other=0).to(tl.float32),
            m1,
        )
    else:
        tl.store(
            Y + row * cc + c0,
            n0 * (1.0 + tl.load(S0 + b * cc + c0).to(tl.float32))
            + tl.load(T0 + b * cc + c0).to(tl.float32),
        )
        tl.store(G + row * cc + c0, tl.load(G0 + b * cc + c0).to(tl.float32))
        tl.store(
            Y + row * cc + c1,
            n1 * (1.0 + tl.load(S0 + b * cc + c1, m1, other=0).to(tl.float32))
            + tl.load(T0 + b * cc + c1, m1, other=0).to(tl.float32),
            m1,
        )
        tl.store(
            G + row * cc + c1,
            tl.load(G0 + b * cc + c1, m1, other=0).to(tl.float32),
            m1,
        )


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
        warps = 4 if block <= 8192 else 8
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
            _dual[(rows, 2)](
                *plain,
                lc,
                float(eps),
                2048,
                triton.next_power_of_2(c - 2048),
                num_warps=1,
                num_stages=1,
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
    warps = 4 if block <= 8192 else 8
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
