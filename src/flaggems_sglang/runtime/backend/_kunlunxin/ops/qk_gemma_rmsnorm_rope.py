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
def _round(v, TYPE: tl.constexpr):
    if TYPE == tl.bfloat16:
        bits = v.to(tl.uint32, bitcast=True)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(
            tl.float32, bitcast=True
        )
        rounded = tl.where((bits & 0x7F800000) == 0x7F800000, v, rounded)
        return rounded
    else:
        return v.to(TYPE).to(tl.float32)


@triton.jit
def _norm_rope(
    X,
    W,
    C,
    P,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    E: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    NEOX: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    GRID: tl.constexpr,
    BR: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(N, BR), GRID):
        row = tile * BR + tl.arange(0, BR)
        source_row = tl.minimum(row, N - 1)
        token = source_row // H
        head = source_row % H
        d = tl.arange(0, B)[None, :]
        base = token[:, None] * XS0 + head[:, None] * D * XS1
        if D == B:
            safe = d
        else:
            safe = tl.minimum(d, D - 1)
        x = tl.load(X + base + safe * XS1).to(tl.float32)
        x = tl.where(d < D, x, 0.0)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + E)
        w = tl.load(W + safe * WS).to(tl.float32)
        y = _round(x * inv[:, None] * (w + 1.0), O.dtype.element_ty)
        if R > 0:
            half = R // 2
            if NEOX:
                partner = tl.where(d < half, d + half, d - half)
                ci = d % half
                first = d < half
            else:
                partner = d ^ 1
                ci = (d // 2) % half
                first = d % 2 == 0
            partner = tl.minimum(tl.maximum(partner, 0), D - 1)
            xp = tl.load(X + base + partner * XS1).to(tl.float32)
            wp = tl.load(W + partner * WS).to(tl.float32)
            yp = _round(xp * inv[:, None] * (wp + 1.0), O.dtype.element_ty)
            pos = tl.load(P + token * PS)
            c = tl.load(C + pos[:, None] * CS0 + ci * CS1).to(tl.float32)
            s = tl.load(C + pos[:, None] * CS0 + (ci + half) * CS1).to(
                tl.float32
            )
            rotated = tl.where(first, y * c - yp * s, y * c + yp * s)
            y = tl.where(d < R, rotated, y)
        tl.store(O + row[:, None] * D + d, y, (row[:, None] < N) & (d < D))


@triton.jit
def _fixed_pair(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    E: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    NEOX: tl.constexpr,
    B: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    GQ: tl.constexpr,
    GK: tl.constexpr,
):
    if NQ > 0:
        _norm_rope(
            Q,
            QW,
            C,
            P,
            QO,
            HQ,
            D,
            R,
            E,
            Q0,
            Q1,
            QWS,
            CS0,
            CS1,
            PS,
            NEOX,
            B,
            NQ,
            GQ,
            4,
        )
    if NK > 0:
        _norm_rope(
            K,
            KW,
            C,
            P,
            KO,
            HK,
            D,
            R,
            E,
            K0,
            K1,
            KWS,
            CS0,
            CS1,
            PS,
            NEOX,
            B,
            NK,
            GK,
            4,
        )


def qk_gemma_rmsnorm_rope(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    eps,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    hq = q.shape[1] // head_dim
    hk = k.shape[1] // head_dim
    nq = q.shape[0] * hq
    nk = k.shape[0] * hk
    gq = min(triton.cdiv(nq, 4), 12)
    gk = min(triton.cdiv(nk, 4), 12)
    if max(gq, gk):
        _fixed_pair[(max(gq, gk),)](
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q_out,
            k_out,
            hq,
            hk,
            head_dim,
            rotary_dim,
            eps,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q_weight.stride(0),
            k_weight.stride(0),
            cos_sin_cache.stride(0),
            cos_sin_cache.stride(1),
            positions.stride(0),
            is_neox_style,
            triton.next_power_of_2(head_dim),
            nq,
            nk,
            gq,
            gk,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out


__all__ = ["qk_gemma_rmsnorm_rope"]
