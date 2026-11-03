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
def _creuse(
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
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    b = tl.program_id(0) // GROUPS
    group = tl.program_id(0) % GROUPS
    c = tl.arange(0, BLOCK)
    m = c < C
    scale0 = 1.0 + tl.load(S0 + b * C + c, m, other=0).to(tl.float32)
    shift0 = tl.load(T0 + b * C + c, m, other=0).to(tl.float32)
    gate0 = tl.load(G0 + b * C + c, m, other=0)
    scale1 = 1.0 + tl.load(S1 + b * C + c, m, other=0).to(tl.float32)
    shift1 = tl.load(T1 + b * C + c, m, other=0).to(tl.float32)
    gate1 = tl.load(G1 + b * C + c, m, other=0)
    for token_idx in range(group, L, GROUPS):
        row = b * L + token_idx
        v = tl.load(X + row * C + c, m, other=0).to(tl.float32)
        mean = tl.sum(v, 0) / C
        centered = tl.where(m, v - mean, 0.0)
        var = tl.sum(centered * centered, 0) / C
        z = centered * tl.rsqrt(var + EPS)
        sel = tl.load(I + row) != 0
        gain = tl.where(sel, scale1, scale0)
        shift = tl.where(sel, shift1, shift0)
        gate = tl.where(sel, gate1, gate0).to(tl.float32)
        tl.store(Y + row * C + c, z * gain + shift, m)
        tl.store(G + row * C + c, gate, m)


@triton.jit
def _stats(
    X,
    base,
    C: tl.constexpr,
    XS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK).to(tl.int64)
    v = tl.load(X + base + c * XS, c < C, other=0).to(tl.float32)
    mean = tl.sum(v, 0) / C
    centered = tl.where(c < C, v - mean, 0.0)
    var = tl.sum(centered * centered, 0) / C
    return mean, tl.rsqrt(var + EPS)


@triton.jit
def _bounded(
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
    lane = tl.arange(0, 256).to(tl.int64)
    for row in range(
        tl.program_id(0).to(tl.int64), M, tl.num_programs(0).to(tl.int64)
    ):
        b = row // L
        token_idx = row % L
        base = b * XS0 + token_idx * XS1
        mean, inv = _stats(X, base, C, XS2, EPS, BLOCK)
        sel = tl.load(I + b * IS0 + token_idx * IS1) != 0
        for tile in range(tl.cdiv(C, 256)):
            c = tile * 256 + lane
            v = tl.load(X + base + c * XS2, c < C, other=0).to(tl.float32)
            z = (v - mean) * inv
            if HW:
                z = z * tl.load(W + c * WS, c < C, other=0).to(tl.float32)
            if HB:
                z = z + tl.load(Bias + c * BS, c < C, other=0).to(tl.float32)
            if sel:
                scale = tl.load(S1 + b * D0 + c * D1, c < C, other=0).to(
                    tl.float32
                )
                shift = tl.load(T1 + b * E0 + c * E1, c < C, other=0).to(
                    tl.float32
                )
                gate = tl.load(G1 + b * F0 + c * F1, c < C, other=0).to(
                    tl.float32
                )
            else:
                scale = tl.load(S0 + b * A0 + c * A1, c < C, other=0).to(
                    tl.float32
                )
                shift = tl.load(T0 + b * B0 + c * B1, c < C, other=0).to(
                    tl.float32
                )
                gate = tl.load(G0 + b * C0 + c * C1, c < C, other=0).to(
                    tl.float32
                )
            tl.store(Y + row * C + c, z * (1.0 + scale) + shift, c < C)
            tl.store(G + row * C + c, gate, c < C)


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
        warps = 4
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
            x.dtype == torch.bfloat16
            and weight is None
            and bias is None
            and 2048 < c <= 4096
            and b <= 65535
            and l >= 1024
            and scale0.dtype == torch.bfloat16
            and shift0.dtype == torch.bfloat16
            and gate0.dtype == torch.bfloat16
            and scale1.dtype == torch.bfloat16
            and shift1.dtype == torch.bfloat16
            and gate1.dtype == torch.bfloat16
        ):
            groups = min(l, max(1, 40 // b), 65535 // b)
            _creuse[(b * groups,)](
                *plain,
                l,
                c,
                float(eps),
                block,
                groups,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return out, gate_out
        if weight is None and bias is None:
            if rows <= 40:
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
                _plain_many[(40,)](
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
        _affine[(min(rows, 40),)](
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
    warps = 4
    _bounded[(min(rows, 40),)](
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
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out, gate_out


__all__ = ["layernorm_scale_shift_gate_select01"]
