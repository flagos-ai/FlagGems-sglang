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
    if S0.dtype == S1.dtype and T0.dtype == T1.dtype and G0.dtype == G1.dtype:
        scale = tl.load(tl.where(sel, S1, S0) + b * C + c, m, other=0).to(
            tl.float32
        )
        shift = tl.load(tl.where(sel, T1, T0) + b * C + c, m, other=0).to(
            tl.float32
        )
        gate = tl.load(tl.where(sel, G1, G0) + b * C + c, m, other=0).to(
            tl.float32
        )
    elif sel:
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
def _group_rows(
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
    L: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HW: tl.constexpr,
    HB: tl.constexpr,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    if L % ROWS == 0:
        safe = rows
    else:
        safe = tl.minimum(rows, L - 1)
    col = tl.arange(0, BLOCK)
    cm = col < C
    values = tl.load(
        X + (batch * L + safe[:, None]) * C + col[None, :],
        cm[None, :],
        other=0,
    ).to(tl.float32)
    values = tl.where(cm[None, :], values, 0.0)
    mean = tl.sum(values, 1) / C
    centered = tl.where(cm[None, :], values - mean[:, None], 0.0)
    inv = tl.rsqrt(tl.sum(centered * centered, 1) / C + EPS)
    norm = centered * inv[:, None]
    if HW:
        norm = norm * tl.load(W + col, cm, other=0).to(tl.float32)[None, :]
    if HB:
        norm = norm + tl.load(Bias + col, cm, other=0).to(tl.float32)[None, :]
    sel = tl.load(I + batch * L + safe) != 0
    sp = tl.where(sel, S1 + batch * C, S0 + batch * C)
    tp = tl.where(sel, T1 + batch * C, T0 + batch * C)
    gp = tl.where(sel, G1 + batch * C, G0 + batch * C)
    scale = tl.load(sp[:, None] + col[None, :], cm[None, :], other=0).to(
        tl.float32
    )
    shift = tl.load(tp[:, None] + col[None, :], cm[None, :], other=0).to(
        tl.float32
    )
    gate = tl.load(gp[:, None] + col[None, :], cm[None, :], other=0).to(
        tl.float32
    )
    offsets = (batch * L + rows[:, None]) * C + col[None, :]
    mask = (rows[:, None] < L) & cm[None, :]
    tl.store(Y + offsets, norm * (1.0 + scale) + shift, mask)
    tl.store(G + offsets, gate, mask)


@triton.jit
def _grouped(
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
    ROWS: tl.constexpr,
):
    _group_rows(
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
        L,
        C,
        EPS,
        BLOCK,
        ROWS,
        False,
        False,
    )


@triton.jit
def _grouped_affine(
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
    L: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HW: tl.constexpr,
    HB: tl.constexpr,
):
    _group_rows(
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
        L,
        C,
        EPS,
        BLOCK,
        ROWS,
        HW,
        HB,
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
        block = max(64, triton.next_power_of_2(c))
        warps = 1
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
        tile_rows = 4 if block >= 2048 else 8
        grouped = (
            x.dtype != torch.float32
            and 128 <= c <= 4096
            and rows >= 128
            and b <= 65535
            and l <= 65535 * tile_rows
            and scale0.dtype == scale1.dtype
            and shift0.dtype == shift1.dtype
            and gate0.dtype == gate1.dtype
        )
        if weight is None and bias is None:
            if grouped:
                _grouped[(triton.cdiv(l, tile_rows), b)](
                    *plain,
                    l,
                    c,
                    float(eps),
                    block,
                    tile_rows,
                    num_warps=1,
                    num_stages=1,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                )
                return out, gate_out
            if rows <= 65535:
                _plain[(rows,)](
                    *plain,
                    lc,
                    float(eps),
                    block,
                    num_warps=warps,
                    num_stages=1,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
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
                    isCloseMemoryAsync=False,
                )
            return out, gate_out
        if grouped:
            _grouped_affine[(triton.cdiv(l, tile_rows), b)](
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
                l,
                c,
                float(eps),
                block,
                tile_rows,
                weight is not None,
                bias is not None,
                num_warps=1,
                num_stages=1,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
            )
        else:
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
                isCloseMemoryAsync=False,
            )
        return out, gate_out
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    gate_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out, gate_out
    block = max(64, triton.next_power_of_2(c))
    warps = 1
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
        isCloseMemoryAsync=False,
    )
    return out, gate_out


__all__ = ["layernorm_scale_shift_gate_select01"]
