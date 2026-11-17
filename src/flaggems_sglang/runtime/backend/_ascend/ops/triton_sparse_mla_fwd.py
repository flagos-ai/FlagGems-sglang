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
def _decode(q, FNUZ: tl.constexpr):
    codes = q.to(tl.int32)
    shape: tl.constexpr = q.shape
    count: tl.constexpr = q.numel
    q = tl.arange(0, 256)
    q = q.to(tl.int32)
    a = q & 127
    e = a >> 3
    m = a & 7
    if FNUZ:
        normal = (1.0 + m.to(tl.float32) * 0.125) * tl.exp2(
            e.to(tl.float32) - 8.0
        )
        v = tl.where(e == 0, m.to(tl.float32) * 0.0009765625, normal)
    else:
        normal = (1.0 + m.to(tl.float32) * 0.125) * tl.exp2(
            e.to(tl.float32) - 7.0
        )
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    v = tl.where(q & 128 != 0, -v, v)
    if FNUZ:
        lut = tl.where(q == 128, float("nan"), v)
    else:
        lut = tl.where(a == 127, float("nan"), v)
    return tl.gather(lut, tl.reshape(codes, (count,)), 0).reshape(shape)


@triton.jit
def _softmax(A, P, T: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, B)
    a = tl.load(A + row * T + n, n < T, -float("inf"))
    m = tl.max(a, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m)
    z = tl.sum(p, 0)
    p = p / tl.where(z > 0, z, 1.0)
    tl.store(P + row * T + n, p, n < T)


@triton.jit
def _packq(
    Q,
    R,
    O,
    H: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    HP: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, triton.next_power_of_2(V))
    a = tl.load(Q + s * Q0 + h * Q1 + d * Q2, (h < H) & (d < V), 0)
    tl.store(O + (s * HP + h) * DP + d, _decode(a, FNUZ), d < V)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2(D - V))
        b = tl.load(R + s * R0 + h * R1 + r * R2, (h < H) & (r < D - V), 0)
        tl.store(O + (s * HP + h) * DP + V + r, _decode(b, FNUZ), r < D - V)
    if DP > D:
        p = tl.arange(0, B)
        tl.store(O + (s * HP + h) * DP + p, 0.0, (p >= D) & (p < DP))


@triton.jit
def _packk(
    K,
    I,
    O,
    D: tl.constexpr,
    DP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    ROWS: tl.constexpr,
):
    lanes = tl.arange(0, ROWS)
    d = tl.arange(0, triton.next_power_of_2(V))
    for block in range(
        tl.program_id(0), triton.cdiv(S * TP, ROWS), tl.num_programs(0)
    ):
        row = block * ROWS + lanes
        s = row // TP
        t = row % TP
        idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2, row < S * TP, -1)
        a = tl.load(
            K + tl.maximum(idx[:, None], 0) * K0 + d[None, :] * K2,
            d[None, :] < V,
            0,
        )
        a = tl.where(((t < T) & (idx >= 0))[:, None], a, 0)
        tl.store(
            O + row[:, None] * DP + d[None, :],
            _decode(a, FNUZ),
            (row[:, None] < S * TP) & (d[None, :] < V),
        )
        if D > V:
            r = tl.arange(0, triton.next_power_of_2(D - V))
            b = tl.load(
                K + tl.maximum(idx[:, None], 0) * K0 + (V + r[None, :]) * K2,
                r[None, :] < D - V,
                0,
            )
            b = tl.where(((t < T) & (idx >= 0))[:, None], b, 0)
            tl.store(
                O + row[:, None] * DP + V + r[None, :],
                _decode(b, FNUZ),
                (row[:, None] < S * TP) & (r[None, :] < D - V),
            )
        if DP > D:
            p = tl.arange(0, B)
            tl.store(
                O + row[:, None] * DP + p[None, :],
                0.0,
                (row[:, None] < S * TP)
                & (p[None, :] >= D)
                & (p[None, :] < DP),
            )


@triton.jit
def _qk(
    Q,
    K,
    I,
    A,
    H: tl.constexpr,
    HP: tl.constexpr,
    DP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    k = tl.arange(0, 64)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(DP // 64):
        d = start * 64 + k
        q = tl.load(Q + (s * HP + h[:, None]) * DP + d[None, :])
        v = tl.load(K + (s * TP + n[None, :]) * DP + d[:, None])
        acc += tl.dot(q, v)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    score = tl.where((idx >= 0)[None, :], acc * SCALE, -float("inf"))
    tl.store(A + (s * HP + h[:, None]) * TP + n[None, :], score)


@triton.jit
def _pv(
    P,
    K,
    O,
    H: tl.constexpr,
    HP: tl.constexpr,
    DP: tl.constexpr,
    V: tl.constexpr,
    TP: tl.constexpr,
    BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    n = tl.arange(0, 32)
    acc = tl.zeros((16, BV), tl.float32)
    for start in range(TP // 32):
        t = start * 32 + n
        p = tl.load(P + (s * HP + h[:, None]) * TP + t[None, :])
        v = tl.load(
            K + (s * TP + t[:, None]) * DP + d[None, :], d[None, :] < V, 0
        )
        acc = tl.dot(p, v, acc)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d = kv.shape[-1]
    t = indices.shape[-1]
    hp = triton.cdiv(h, 16) * 16
    bn = 128 if t >= 512 else 64
    tp = triton.cdiv(t, bn) * bn
    dp = triton.cdiv(d, 64) * 64
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    q = torch.empty((s, hp, dp), device=q_nope.device, dtype=torch.bfloat16)
    k = torch.empty((s, tp, dp), device=q_nope.device, dtype=torch.bfloat16)
    fused = t <= 128
    a = (
        torch.empty((s, hp, tp), device=q_nope.device, dtype=torch.float32)
        if not fused
        else None
    )
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    if s * hp <= 512 and s * tp <= 8192:
        _pack_both.run(
            q_nope.view(torch.uint8),
            q_rope.view(torch.uint8),
            q,
            h,
            d_v,
            d,
            dp,
            hp,
            *q_nope.stride(),
            *q_rope.stride(),
            fnuz,
            triton.next_power_of_2(dp),
            kv.view(torch.uint8),
            indices,
            k,
            d,
            dp,
            t,
            tp,
            kv.stride(0),
            kv.stride(2),
            indices.stride(0),
            indices.stride(2),
            fnuz,
            triton.next_power_of_2(dp),
            s,
            d_v,
            4,
            s * hp,
            min(triton.cdiv(s * tp, 4), 40),
            num_warps=4,
            grid=(s * hp + min(triton.cdiv(s * tp, 4), 40),),
            warmup=False,
        )
    else:
        _packq.run(
            q_nope.view(torch.uint8),
            q_rope.view(torch.uint8),
            q,
            h,
            d_v,
            d,
            dp,
            hp,
            *q_nope.stride(),
            *q_rope.stride(),
            fnuz,
            triton.next_power_of_2(dp),
            num_warps=4,
            grid=(s, hp),
            warmup=False,
        )
        if dp == d and t >= 512 and (s * t >= 8 * kv.shape[0]):
            kt = torch.empty(
                (kv.shape[0], dp), device=q_nope.device, dtype=torch.bfloat16
            )
            _predecode_table.run(
                kv.view(torch.uint8),
                kt,
                kv.shape[0],
                d,
                d_v,
                kv.stride(0),
                kv.stride(2),
                fnuz,
                4,
                grid=(triton.cdiv(kv.shape[0], 4),),
                warmup=False,
                num_warps=4,
            )
            _copy_words.run(
                kt.view(torch.int32),
                indices,
                k.view(torch.int32),
                s,
                t,
                tp,
                dp,
                indices.stride(0),
                indices.stride(2),
                16,
                triton.next_power_of_2(dp // 2),
                num_warps=4,
                grid=(min(triton.cdiv(s * tp, 16), 40),),
                warmup=False,
            )
        else:
            _packk.run(
                kv.view(torch.uint8),
                indices,
                k,
                d,
                dp,
                t,
                tp,
                kv.stride(0),
                kv.stride(2),
                indices.stride(0),
                indices.stride(2),
                fnuz,
                triton.next_power_of_2(dp),
                s,
                d_v,
                4,
                num_warps=4,
                grid=(min(triton.cdiv(s * tp, 4), 40),),
                warmup=False,
            )
    if fused:
        _short_attention.run(
            q,
            k,
            indices,
            out,
            h,
            hp,
            dp,
            d_v,
            t,
            tp,
            indices.stride(0),
            indices.stride(2),
            sm_scale,
            triton.next_power_of_2(tp),
            128,
            grid=(s, hp // 16, triton.cdiv(d_v, 128)),
            num_warps=4,
            num_stages=1,
            warmup=False,
        )
        return out
    _qk.run(
        q,
        k,
        indices,
        a,
        h,
        hp,
        dp,
        t,
        tp,
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        bn,
        num_warps=4,
        num_stages=2,
        grid=(s, hp // 16, tp // bn),
        warmup=False,
    )
    if tp <= 256 or (tp <= 512 and s * triton.cdiv(h, 16) <= 16):
        _softmax_pv.run(
            a,
            k,
            out,
            h,
            hp,
            dp,
            d_v,
            tp,
            triton.next_power_of_2(tp),
            32 if tp > 256 else 64,
            num_warps=4,
            num_stages=1,
            grid=(s, hp // 16, triton.cdiv(d_v, 32 if tp > 256 else 64)),
            warmup=False,
        )
    else:
        p = torch.empty(
            (s, hp, tp), device=q_nope.device, dtype=torch.bfloat16
        )
        _softmax.run(
            a,
            p,
            tp,
            triton.next_power_of_2(tp),
            num_warps=4,
            grid=(s * hp,),
            warmup=False,
        )
        bv = 128
        _pv.run(
            p,
            k,
            out,
            h,
            hp,
            dp,
            d_v,
            tp,
            bv,
            num_warps=4,
            num_stages=2,
            grid=(s, hp // 16, triton.cdiv(d_v, bv)),
            warmup=False,
        )
    return out


@triton.jit
def _softmax_pv(
    A,
    K,
    O,
    H: tl.constexpr,
    HP: tl.constexpr,
    DP: tl.constexpr,
    V: tl.constexpr,
    TP: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    t = tl.arange(0, BT)
    a = tl.load(
        A + (s * HP + h[:, None]) * TP + t[None, :],
        (h[:, None] < H) & (t[None, :] < TP),
        -float("inf"),
    )
    m = tl.max(a, 1)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m[:, None])
    z = tl.sum(p, 1)
    p = (p / tl.where(z > 0, z, 1.0)[:, None]).to(tl.bfloat16)
    v = tl.load(
        K + (s * TP + t[:, None]) * DP + d[None, :],
        (t[:, None] < TP) & (d[None, :] < V),
        0,
    )
    acc = tl.dot(p, v)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


@triton.jit
def _packq_at(
    Q,
    R,
    O,
    H: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    HP: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
    s,
    h,
):
    d = tl.arange(0, triton.next_power_of_2(V))
    a = tl.load(Q + s * Q0 + h * Q1 + d * Q2, (h < H) & (d < V), 0)
    tl.store(O + (s * HP + h) * DP + d, _decode(a, FNUZ), d < V)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2(D - V))
        b = tl.load(R + s * R0 + h * R1 + r * R2, (h < H) & (r < D - V), 0)
        tl.store(O + (s * HP + h) * DP + V + r, _decode(b, FNUZ), r < D - V)
    if DP > D:
        p = tl.arange(0, B)
        tl.store(O + (s * HP + h) * DP + p, 0.0, (p >= D) & (p < DP))


@triton.jit
def _packk_at(
    K,
    I,
    O,
    D: tl.constexpr,
    DP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    ROWS: tl.constexpr,
    pid,
    npg: tl.constexpr,
):
    lanes = tl.arange(0, ROWS)
    d = tl.arange(0, triton.next_power_of_2(V))
    for block in range(pid, triton.cdiv(S * TP, ROWS), npg):
        row = block * ROWS + lanes
        s = row // TP
        t = row % TP
        idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2, row < S * TP, -1)
        a = tl.load(
            K + tl.maximum(idx[:, None], 0) * K0 + d[None, :] * K2,
            d[None, :] < V,
            0,
        )
        a = tl.where(((t < T) & (idx >= 0))[:, None], a, 0)
        tl.store(
            O + row[:, None] * DP + d[None, :],
            _decode(a, FNUZ),
            (row[:, None] < S * TP) & (d[None, :] < V),
        )
        if D > V:
            r = tl.arange(0, triton.next_power_of_2(D - V))
            b = tl.load(
                K + tl.maximum(idx[:, None], 0) * K0 + (V + r[None, :]) * K2,
                r[None, :] < D - V,
                0,
            )
            b = tl.where(((t < T) & (idx >= 0))[:, None], b, 0)
            tl.store(
                O + row[:, None] * DP + V + r[None, :],
                _decode(b, FNUZ),
                (row[:, None] < S * TP) & (r[None, :] < D - V),
            )
        if DP > D:
            p = tl.arange(0, B)
            tl.store(
                O + row[:, None] * DP + p[None, :],
                0.0,
                (row[:, None] < S * TP)
                & (p[None, :] >= D)
                & (p[None, :] < DP),
            )


@triton.jit
def _pack_both(
    q_Q,
    q_R,
    q_O,
    q_H: tl.constexpr,
    q_V: tl.constexpr,
    q_D: tl.constexpr,
    q_DP: tl.constexpr,
    q_HP: tl.constexpr,
    q_Q0: tl.constexpr,
    q_Q1: tl.constexpr,
    q_Q2: tl.constexpr,
    q_R0: tl.constexpr,
    q_R1: tl.constexpr,
    q_R2: tl.constexpr,
    q_FNUZ: tl.constexpr,
    q_B: tl.constexpr,
    k_K,
    k_I,
    k_O,
    k_D: tl.constexpr,
    k_DP: tl.constexpr,
    k_T: tl.constexpr,
    k_TP: tl.constexpr,
    k_K0: tl.constexpr,
    k_K2: tl.constexpr,
    k_I0: tl.constexpr,
    k_I2: tl.constexpr,
    k_FNUZ: tl.constexpr,
    k_B: tl.constexpr,
    k_S: tl.constexpr,
    k_V: tl.constexpr,
    k_ROWS: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NQ:
        _packq_at(
            q_Q,
            q_R,
            q_O,
            q_H,
            q_V,
            q_D,
            q_DP,
            q_HP,
            q_Q0,
            q_Q1,
            q_Q2,
            q_R0,
            q_R1,
            q_R2,
            q_FNUZ,
            q_B,
            pid // q_HP,
            pid % q_HP,
        )
    else:
        _packk_at(
            k_K,
            k_I,
            k_O,
            k_D,
            k_DP,
            k_T,
            k_TP,
            k_K0,
            k_K2,
            k_I0,
            k_I2,
            k_FNUZ,
            k_B,
            k_S,
            k_V,
            k_ROWS,
            pid - NQ,
            NK,
        )


@triton.jit
def _predecode_table(
    K,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    FNUZ: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, triton.next_power_of_2(V))
    a = tl.load(
        K + row[:, None] * K0 + d[None, :] * K2,
        (row[:, None] < N) & (d[None, :] < V),
        0,
    )
    tl.store(
        O + row[:, None] * D + d[None, :],
        _decode(a, FNUZ),
        (row[:, None] < N) & (d[None, :] < V),
    )
    if D > V:
        r = tl.arange(0, triton.next_power_of_2(D - V))
        b = tl.load(
            K + row[:, None] * K0 + (V + r[None, :]) * K2,
            (row[:, None] < N) & (r[None, :] < D - V),
            0,
        )
        tl.store(
            O + row[:, None] * D + V + r[None, :],
            _decode(b, FNUZ),
            (row[:, None] < N) & (r[None, :] < D - V),
        )


@triton.jit
def _copy_words(
    K,
    I,
    O,
    S: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    DP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    ROWS: tl.constexpr,
    B: tl.constexpr,
):
    lanes = tl.arange(0, ROWS)
    d = tl.arange(0, B)
    for block in range(
        tl.program_id(0), triton.cdiv(S * TP, ROWS), tl.num_programs(0)
    ):
        row = block * ROWS + lanes
        s = row // TP
        t = row % TP
        idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2, row < S * TP, -1)
        words = tl.load(
            K + tl.maximum(idx[:, None], 0) * (DP // 2) + d[None, :],
            (row[:, None] < S * TP) & (d[None, :] < DP // 2),
            0,
        )
        words = tl.where(((t < T) & (idx >= 0))[:, None], words, 0)
        tl.store(
            O + row[:, None] * (DP // 2) + d[None, :],
            words,
            (row[:, None] < S * TP) & (d[None, :] < DP // 2),
        )


@triton.jit
def _short_attention(
    Q,
    K,
    I,
    O,
    H: tl.constexpr,
    HP: tl.constexpr,
    DP: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.arange(0, B)
    r = tl.arange(0, 64)
    acc = tl.zeros((16, B), tl.float32)
    for start in range(DP // 64):
        d = start * 64 + r
        query = tl.load(Q + (s * HP + h[:, None]) * DP + d[None, :])
        key = tl.load(
            K + (s * TP + n[None, :]) * DP + d[:, None], n[None, :] < TP, 0
        )
        acc += tl.dot(query, key)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    scores = tl.where(
        ((n < T) & (idx >= 0))[None, :], acc * SCALE, -float("inf")
    )
    maximum = tl.max(scores, 1)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probability = tl.exp(scores - maximum[:, None])
    denominator = tl.sum(probability, 1)
    probability = (
        probability / tl.where(denominator > 0, denominator, 1.0)[:, None]
    )
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    value = tl.load(
        K + (s * TP + n[:, None]) * DP + d[None, :],
        (n[:, None] < TP) & (d[None, :] < V),
        0,
    )
    result = tl.dot(probability.to(tl.bfloat16), value)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        result,
        (h[:, None] < H) & (d[None, :] < V),
    )


__all__ = ["triton_sparse_mla_fwd"]
