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
def _rows(
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
    BP: tl.constexpr,
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
    if RD > 0:
        half: tl.constexpr = RD // 2
        pair = tl.arange(0, BP)
        pair_mask = (token[:, None] < T) & (pair[None, :] < half)
        pos = tl.load(P + token * PS, token < T, 0)
        cos = tl.load(
            C + pos[:, None] * CS0 + pair[None, :] * CS1, pair_mask, 0
        ).to(tl.float32)
        sin = tl.load(
            C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pair_mask, 0
        ).to(tl.float32)
        x1 = tl.load(X + base[:, None] + pair[None, :] * XS1, pair_mask, 0).to(
            tl.float32
        )
        x2 = tl.load(
            X + base[:, None] + (half + pair[None, :]) * XS1, pair_mask, 0
        ).to(tl.float32)
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
        r1 = n1 * cos - n2 * sin
        r2 = n2 * cos + n1 * sin
        tl.store(Y + out[:, None] + pair[None, :], r1, pair_mask)
        tl.store(Y + out[:, None] + half + pair[None, :], r2, pair_mask)
    if RD < D:
        w = tl.load(W + d * WS, d < D, 0).to(tl.float32)
        normal = (x * inv[:, None] * (w[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(
            Y + out[:, None] + d[None, :], normal, mask & (d[None, :] >= RD)
        )
    if HAS_GATE:
        gate = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, mask, 0)
        tl.store(G + out[:, None] + d[None, :], gate, mask)


@triton.jit
def _fused(
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
    BP: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    NP: tl.constexpr,
):
    physical = tl.program_id(0)
    if Q_BLOCKS * (Q_BLOCKS + K_BLOCKS) > 2147483647:
        physical = physical.to(tl.int64)
    q_before = physical * Q_BLOCKS // (Q_BLOCKS + K_BLOCKS)
    q_after = (physical + 1) * Q_BLOCKS // (Q_BLOCKS + K_BLOCKS)
    block = tl.where(
        q_after > q_before, q_before, Q_BLOCKS + physical - q_before
    )
    if block < Q_BLOCKS:
        _rows(
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
            BP,
            block,
        )
    else:
        _rows(
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
            BP,
            block - Q_BLOCKS,
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
    rows = 1 if t * (num_q_heads + num_kv_heads) < 512 else 16
    q_blocks = triton.cdiv(t * num_q_heads, rows)
    k_blocks = triton.cdiv(t * num_kv_heads, rows)
    programs = q_blocks + k_blocks
    _fused[programs,](
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
        triton.next_power_of_2(max(1, rotary_dim // 2)),
        q_blocks,
        k_blocks,
        programs,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (q_out, k_out, gate_out)


__all__ = ["fused_qk_gemma_rmsnorm_rope_gate"]
