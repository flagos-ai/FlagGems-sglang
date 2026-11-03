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
def base_rows(
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
    ROW_BLOCK: tl.constexpr,
    BD: tl.constexpr,
    PID,
):
    row = PID * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    token = row // H
    head = row % H
    d = tl.arange(0, BD)
    mask = (token[:, None] < T) & (d[None, :] < D)
    offset = (
        token[:, None] * XS0
        + (head[:, None] * D * (1 + HAS_GATE) + d[None, :]) * XS1
    )
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    w = tl.load(W + d * WS, d < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    normal = (
        (x * inv[:, None] * (w[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    if RD > 0:
        half: tl.constexpr = RD // 2
        partner = tl.where(d < RD, tl.where(d < half, d + half, d - half), d)
        paired = tl.gather(
            normal, tl.broadcast_to(partner[None, :], (ROW_BLOCK, BD)), 1
        )
        angle = tl.where(d < half, d, d - half)
        pos = tl.load(P + token * PS, token < T, 0)
        cmask = (token[:, None] < T) & (d[None, :] < RD)
        cos = tl.load(
            C + pos[:, None] * CS0 + angle[None, :] * CS1, cmask, 0
        ).to(tl.float32)
        sin = tl.load(
            C + pos[:, None] * CS0 + (half + angle[None, :]) * CS1, cmask, 0
        ).to(tl.float32)
        rotated = tl.where(
            d[None, :] < half,
            normal * cos - paired * sin,
            normal * cos + paired * sin,
        )
        normal = tl.where(d[None, :] < RD, rotated, normal)
    output_offset = row[:, None] * D + d[None, :]
    tl.store(Y + output_offset, normal, mask)
    if HAS_GATE:
        gate = tl.load(X + offset + D * XS1, mask, 0)
        tl.store(G + output_offset, gate, mask)


@triton.jit
def base_fused(
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
    ROW_BLOCK: tl.constexpr,
    BD: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < Q_BLOCKS:
        base_rows(
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
            ROW_BLOCK,
            BD,
            pid,
        )
    else:
        base_rows(
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
            ROW_BLOCK,
            BD,
            pid - Q_BLOCKS,
        )


def basefused_qk_gemma_rmsnorm_rope_gate(
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
    rows = 1 if t * (num_q_heads + num_kv_heads) < 512 else 4
    q_blocks = triton.cdiv(t * num_q_heads, rows)
    k_blocks = triton.cdiv(t * num_kv_heads, rows)
    base_fused[q_blocks + k_blocks,](
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
        rotary_dim,
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
        q_blocks,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return (q_out, k_out, gate_out)


@triton.jit
def pair_rows(
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
    BP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BLOCK,
):
    row = BLOCK * R + tl.arange(0, R)
    token = row // H
    head = row % H
    base = token * XS0 + head * D * (1 + HAS_GATE) * XS1
    out = row * D
    half: tl.constexpr = RD // 2
    pair = tl.arange(0, BP)
    pair_mask = (token[:, None] < T) & (pair[None, :] < half)
    x1 = tl.load(X + base[:, None] + pair[None, :] * XS1, pair_mask, 0).to(
        tl.float32
    )
    x2 = tl.load(
        X + base[:, None] + (half + pair[None, :]) * XS1, pair_mask, 0
    ).to(tl.float32)
    squares = tl.sum(x1 * x1 + x2 * x2, 1)
    if RD < D:
        tail = tl.arange(0, BT)
        tail_mask = (token[:, None] < T) & (tail[None, :] < D - RD)
        xt = tl.load(
            X + base[:, None] + (RD + tail[None, :]) * XS1, tail_mask, 0
        ).to(tl.float32)
        squares += tl.sum(xt * xt, 1)
    inv = tl.rsqrt(squares / D + EPS)
    if RD > 0:
        w1 = tl.load(W + pair * WS, pair < half, 0).to(tl.float32)
        w2 = tl.load(W + (half + pair) * WS, pair < half, 0).to(tl.float32)
        n1 = (
            (x1 * inv[:, None] * (w1[None, :] + 1.0))
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        n2 = (
            (x2 * inv[:, None] * (w2[None, :] + 1.0))
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        pos = tl.load(P + token * PS, token < T, 0)
        cos = tl.load(
            C + pos[:, None] * CS0 + pair[None, :] * CS1, pair_mask, 0
        ).to(tl.float32)
        sin = tl.load(
            C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pair_mask, 0
        ).to(tl.float32)
        tl.store(
            Y + out[:, None] + pair[None, :], n1 * cos - n2 * sin, pair_mask
        )
        tl.store(
            Y + out[:, None] + half + pair[None, :],
            n2 * cos + n1 * sin,
            pair_mask,
        )
    if RD < D:
        wt = tl.load(W + (RD + tail) * WS, tail < D - RD, 0).to(tl.float32)
        nt = (xt * inv[:, None] * (wt[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(Y + out[:, None] + RD + tail[None, :], nt, tail_mask)
    if HAS_GATE:
        if R == 8:
            d = tl.arange(0, BD)
            mask = (token[:, None] < T) & (d[None, :] < D)
            gate = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, mask, 0)
            tl.store(G + out[:, None] + d[None, :], gate, mask)
        else:
            d = tl.arange(0, BD // 2)
            m1 = (token[:, None] < T) & (d[None, :] < D)
            m2 = (token[:, None] < T) & (d[None, :] + BD // 2 < D)
            g1 = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, m1, 0)
            g2 = tl.load(
                X + base[:, None] + (D + BD // 2 + d[None, :]) * XS1, m2, 0
            )
            tl.store(G + out[:, None] + d[None, :], g1, m1)
            tl.store(G + out[:, None] + BD // 2 + d[None, :], g2, m2)


@triton.jit
def pair_fused(Q, K, QW, KW, C, P, QO, KO, GO, META: tl.constexpr):
    T: tl.constexpr = META[0]
    NQ: tl.constexpr = META[1]
    NK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    RD: tl.constexpr = META[4]
    EPS: tl.constexpr = META[5]
    QS0: tl.constexpr = META[6]
    QS1: tl.constexpr = META[7]
    KS0: tl.constexpr = META[8]
    KS1: tl.constexpr = META[9]
    QWS: tl.constexpr = META[10]
    KWS: tl.constexpr = META[11]
    CS0: tl.constexpr = META[12]
    CS1: tl.constexpr = META[13]
    PS: tl.constexpr = META[14]
    HAS_GATE: tl.constexpr = META[15]
    R: tl.constexpr = META[16]
    BP: tl.constexpr = META[17]
    BT: tl.constexpr = META[18]
    BD: tl.constexpr = META[19]
    Q_BLOCKS: tl.constexpr = META[20]
    block = tl.program_id(0)
    if block < Q_BLOCKS:
        pair_rows(
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
            BP,
            BT,
            BD,
            block,
        )
    else:
        pair_rows(
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
            BP,
            BT,
            BD,
            block - Q_BLOCKS,
        )


def pair_call(
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
    rows,
    warps,
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
    q_blocks = triton.cdiv(t * num_q_heads, rows)
    k_blocks = triton.cdiv(t * num_kv_heads, rows)
    pair_fused.run(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_out,
        k_out,
        gate_out if has_gate else q_out,
        (
            t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
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
            triton.next_power_of_2(max(1, rotary_dim // 2)),
            triton.next_power_of_2(max(1, head_dim - rotary_dim)),
            triton.next_power_of_2(head_dim),
            q_blocks,
        ),
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(q_blocks + k_blocks,),
        warmup=False,
    )
    return (q_out, k_out, gate_out)


def pairfused_qk_gemma_rmsnorm_rope_gate(
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
    rows = 8
    return pair_call(
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
        rows,
        4,
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
    if q_gate.shape[0] * (num_q_heads + num_kv_heads) >= 8192:
        return pair_call(
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
            (
                32
                if q_gate.shape[0] * (num_q_heads + num_kv_heads) < 65536
                else 8
            ),
            4,
        )
    return basefused_qk_gemma_rmsnorm_rope_gate(
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
    )


__all__ = ["fused_qk_gemma_rmsnorm_rope_gate"]
