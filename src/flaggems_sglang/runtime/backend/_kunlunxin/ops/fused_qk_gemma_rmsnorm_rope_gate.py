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
def _rows_full(
    X,
    W,
    C,
    P,
    Y,
    G,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    BLOCK,
):
    row = BLOCK * R + tl.arange(0, R)
    token = row // H
    head = row % H
    base = token * XS0 + head * D * (1 + HAS_GATE) * XS1
    out = row * D
    half: tl.constexpr = D // 2
    pair = tl.arange(0, BP)
    pair_mask = (token[:, None] < T) & (pair[None, :] < half)
    x1 = tl.load(X + base[:, None] + pair[None, :] * XS1, pair_mask, 0).to(
        tl.float32
    )
    x2 = tl.load(
        X + base[:, None] + (half + pair[None, :]) * XS1, pair_mask, 0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x1 * x1 + x2 * x2, 1) / D + EPS)
    pos = tl.load(P + token * PS, token < T, 0)
    cos = tl.load(
        C + pos[:, None] * CS0 + pair[None, :] * CS1, pair_mask, 0
    ).to(tl.float32)
    sin = tl.load(
        C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pair_mask, 0
    ).to(tl.float32)
    w1 = tl.load(W + pair * WS, pair < half, 0).to(tl.float32)
    w2 = tl.load(W + (half + pair) * WS, pair < half, 0).to(tl.float32)
    n1 = _round_norm(
        x1 * inv[:, None] * (w1[None, :] + 1.0), Y.dtype.element_ty
    )
    n2 = _round_norm(
        x2 * inv[:, None] * (w2[None, :] + 1.0), Y.dtype.element_ty
    )
    tl.store(Y + out[:, None] + pair[None, :], n1 * cos - n2 * sin, pair_mask)
    tl.store(
        Y + out[:, None] + half + pair[None, :], n2 * cos + n1 * sin, pair_mask
    )
    if HAS_GATE:
        d = tl.arange(0, BD)
        mask = (token[:, None] < T) & (d[None, :] < D)
        gate = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, mask, 0)
        tl.store(G + out[:, None] + d[None, :], gate, mask)


@triton.jit
def _rows_partial(
    X,
    W,
    C,
    P,
    Y,
    G,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BLOCK,
):
    row = BLOCK * R + tl.arange(0, R)
    token = row // H
    head = row % H
    d = tl.arange(0, BD)
    base = token * XS0 + head * D * (1 + HAS_GATE) * XS1
    mask = (token[:, None] < T) & (d[None, :] < D)
    x = tl.load(X + base[:, None] + d[None, :] * XS1, mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    out = row * D
    half: tl.constexpr = RD // 2
    if RD > 0:
        pair = tl.arange(0, triton.next_power_of_2(half))
        pmask = (token[:, None] < T) & (pair[None, :] < half)
        x1 = tl.load(X + base[:, None] + pair[None, :] * XS1, pmask, 0).to(
            tl.float32
        )
        x2 = tl.load(
            X + base[:, None] + (half + pair[None, :]) * XS1, pmask, 0
        ).to(tl.float32)
        w1 = tl.load(W + pair * WS, pair < half, 0).to(tl.float32)
        w2 = tl.load(W + (half + pair) * WS, pair < half, 0).to(tl.float32)
        n1 = _round_norm(
            x1 * inv[:, None] * (w1[None, :] + 1.0), Y.dtype.element_ty
        )
        n2 = _round_norm(
            x2 * inv[:, None] * (w2[None, :] + 1.0), Y.dtype.element_ty
        )
        pos = tl.load(P + token * PS, token < T, 0)
        cos = tl.load(
            C + pos[:, None] * CS0 + pair[None, :] * CS1, pmask, 0
        ).to(tl.float32)
        sin = tl.load(
            C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pmask, 0
        ).to(tl.float32)
        tl.store(Y + out[:, None] + pair[None, :], n1 * cos - n2 * sin, pmask)
        tl.store(
            Y + out[:, None] + half + pair[None, :], n2 * cos + n1 * sin, pmask
        )
    if RD < D:
        tail = RD + tl.arange(0, triton.next_power_of_2(D - RD))
        tmask = (token[:, None] < T) & (tail[None, :] < D)
        xt = tl.load(X + base[:, None] + tail[None, :] * XS1, tmask, 0).to(
            tl.float32
        )
        wt = tl.load(W + tail * WS, tail < D, 0).to(tl.float32)
        nt = (xt * inv[:, None] * (wt[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(Y + out[:, None] + tail[None, :], nt, tmask)
    if HAS_GATE:
        gate = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, mask, 0)
        tl.store(G + out[:, None] + d[None, :], gate, mask)


@triton.jit
def _fused_full(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    T: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
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
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    NP: tl.constexpr,
):
    block = tl.program_id(0)
    _rows_full(
        Q,
        QW,
        C,
        P,
        QO,
        GO,
        T,
        NQ,
        D,
        EPS,
        QS0,
        QS1,
        QWS,
        CS0,
        CS1,
        PS,
        HAS_GATE,
        R,
        BD,
        BP,
        block,
    )
    if block < K_BLOCKS:
        _rows_full(
            K,
            KW,
            C,
            P,
            KO,
            GO,
            T,
            NK,
            D,
            EPS,
            KS0,
            KS1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            R,
            BD,
            BP,
            block,
        )


@triton.jit
def _fused_partial(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    T: tl.constexpr,
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
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    NP: tl.constexpr,
):
    block = tl.program_id(0)
    _rows_partial(
        Q,
        QW,
        C,
        P,
        QO,
        GO,
        T,
        NQ,
        D,
        RD,
        EPS,
        QS0,
        QS1,
        QWS,
        CS0,
        CS1,
        PS,
        HAS_GATE,
        R,
        BD,
        block,
    )
    if block < K_BLOCKS:
        _rows_partial(
            K,
            KW,
            C,
            P,
            KO,
            GO,
            T,
            NK,
            D,
            RD,
            EPS,
            KS0,
            KS1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            R,
            BD,
            block,
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
    rows = 16 if head_dim <= 128 else 4
    q_blocks = triton.cdiv(t * num_q_heads, rows)
    k_blocks = triton.cdiv(t * num_kv_heads, rows)
    programs = max(q_blocks, k_blocks)
    args = (
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
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    strides = (
        eps,
        q_gate.stride(0),
        q_gate.stride(1),
        k.stride(0),
        k.stride(1),
        q_weight.stride(0),
        k_weight.stride(0),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        positions.stride(0),
        has_gate,
        rows,
        triton.next_power_of_2(head_dim),
    )
    launch = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
    if rotary_dim == head_dim:
        _fused_full[(programs,)](
            *args,
            *strides,
            triton.next_power_of_2(head_dim // 2),
            q_blocks,
            k_blocks,
            programs,
            **launch,
        )
    else:
        _fused_partial[(programs,)](
            *args, rotary_dim, *strides, q_blocks, k_blocks, programs, **launch
        )
    return q_out, k_out, gate_out


@triton.jit
def _round_norm(v, DT: tl.constexpr):
    if DT == tl.bfloat16:
        bits = v.to(tl.uint32, bitcast=True)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(
            tl.float32, bitcast=True
        )
        return tl.where((bits & 0x7F800000) == 0x7F800000, v, rounded)
    return v.to(DT).to(tl.float32)


__all__ = ["fused_qk_gemma_rmsnorm_rope_gate"]
