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
    q = q.to(tl.uint32)
    a = q & 127
    e = a >> 3
    m = a & 7
    if FNUZ:
        normal = (e + 119 << 23 | m << 20).to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.0009765625, normal)
    else:
        normal = (e + 120 << 23 | m << 20).to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    v = (v.to(tl.uint32, bitcast=True) | (q & 128) << 24).to(
        tl.float32, bitcast=True
    )
    if FNUZ:
        return tl.where(q == 128, float("nan"), v)
    else:
        return tl.where(a == 127, float("nan"), v)


@triton.jit
def _softmax(
    A,
    P,
    I,
    T: tl.constexpr,
    B: tl.constexpr,
    REAL_T: tl.constexpr,
    HP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.arange(0, B)
    a = tl.load(A + row * T + n, n < T, -float("inf"))
    idx = tl.load(I + row // HP * I0 + n * I2, n < REAL_T, -1)
    a = tl.where(idx >= 0, a, -float("inf"))
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
):
    d = tl.arange(0, triton.next_power_of_2(V))
    row = tl.program_id(0) * TP + tl.program_id(1)
    s = row // TP
    t = row % TP
    idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2)
    a = tl.load(K + tl.maximum(idx, 0) * K0 + d * K2, d < V, 0)
    a = tl.where((t < T) & (idx >= 0), a, 0)
    x = _decode(a, FNUZ)
    tl.store(O + row * DP + d, x, d < V)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2(D - V))
        a = tl.load(K + tl.maximum(idx, 0) * K0 + (V + r) * K2, r < D - V, 0)
        a = tl.where((t < T) & (idx >= 0), a, 0)
        tl.store(O + row * DP + V + r, _decode(a, FNUZ), r < D - V)
    if DP > D:
        p = tl.arange(0, B)
        tl.store(O + row * DP + p, 0.0, (p >= D) & (p < DP))


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
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 64)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(DP // 64):
        d = start * 64 + k
        q = tl.load(Q + (s * HP + h[:, None]) * DP + d[None, :])
        v = tl.load(K + (s * TP + n[None, :]) * DP + d[:, None])
        acc = tl.dot(q, v, acc)
    score = acc * SCALE
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
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * 64 + tl.arange(0, 64)
    n = tl.arange(0, 32)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(TP // 32):
        t = start * 32 + n
        p = tl.load(P + (s * HP + h[:, None]) * TP + t[None, :])
        v = tl.load(K + (s * TP + t[:, None]) * DP + d[None, :])
        acc = tl.dot(p, v, acc)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


@triton.jit
def _expand2(x, FNUZ: tl.constexpr):
    x = x.to(tl.uint32)
    p = x & 255 | (x & 65280) << 8
    a = p & 8323199
    e = p & 7864440
    m = p & 458759
    if FNUZ:
        bias = 119
    else:
        bias = 120
    normal = (e >> 3) + bias * 65537 << 7 | m << 4
    normal_mask = ((e + 7864440 & 8388736) >> 7) * 65535
    mask4 = ((m & 262148) >> 2) * 65535
    mask2 = ((m + 393222 & 524296) >> 3) * 65535
    mask1 = ((m + 458759 & 524296) >> 3) * 65535
    sub4 = (bias - 1) * 128 * 65537 + (m << 5)
    sub2 = (bias - 2) * 128 * 65537 + (m << 6)
    sub1 = (bias - 2) * 128 * 65537
    sub = sub4 & mask4 | sub2 & ~mask4 & mask2 | sub1 & ~mask2 & mask1
    bits = normal & normal_mask | sub & ~normal_mask | (p & 8388736) << 8
    if FNUZ:
        nan_mask = ((p & 8388736 & ~(a + 8323199 & 8388736)) >> 7) * 65535
    else:
        nan_mask = ((a + 65537 & 8388736) >> 7) * 65535
    return bits & ~nan_mask | 2143322048 & nan_mask


@triton.jit
def _packq4(
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
    R0: tl.constexpr,
    R1: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, triton.next_power_of_2(V // 2))
    x = tl.load(Q + s * Q0 + h * Q1 + r, (h < H) & (r < V // 2), 0)
    b = _expand2(x, FNUZ)
    o = tl.arange(0, triton.next_power_of_2(V // 2))
    tl.store(O + (s * HP + h) * (DP // 2) + o, b, o < V // 2)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        x = tl.load(R + s * R0 + h * R1 + r, (h < H) & (r < (D - V) // 2), 0)
        b = _expand2(x, FNUZ)
        o = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        tl.store(
            O + (s * HP + h) * (DP // 2) + V // 2 + o, b, o < (D - V) // 2
        )
    if DP > D:
        o = tl.arange(0, triton.next_power_of_2(DP // 2))
        tl.store(
            O + (s * HP + h) * (DP // 2) + o, 0, (o >= D // 2) & (o < DP // 2)
        )


@triton.jit
def _packk4(
    K,
    I,
    O,
    V: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    K0: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    t = tl.program_id(1)
    idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2)
    valid = (t < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    r = tl.arange(0, triton.next_power_of_2(V // 2))
    x = tl.load(K + page * K0 + r, r < V // 2, 0)
    b = _expand2(tl.where(valid, x, 0), FNUZ)
    o = tl.arange(0, triton.next_power_of_2(V // 2))
    tl.store(O + (s * TP + t) * (DP // 2) + o, b, o < V // 2)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        x = tl.load(K + page * K0 + V // 2 + r, r < (D - V) // 2, 0)
        b = _expand2(tl.where(valid, x, 0), FNUZ)
        o = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        tl.store(
            O + (s * TP + t) * (DP // 2) + V // 2 + o, b, o < (D - V) // 2
        )
    if DP > D:
        o = tl.arange(0, triton.next_power_of_2(DP // 2))
        tl.store(
            O + (s * TP + t) * (DP // 2) + o, 0, (o >= D // 2) & (o < DP // 2)
        )


def _sparse(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d = kv.shape[-1]
    t = indices.shape[-1]
    hp = triton.cdiv(h, 16) * 16
    tp = triton.cdiv(t, 32) * 32
    dp = triton.cdiv(d, 64) * 64
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    q = torch.empty((s, hp, dp), device=q_nope.device, dtype=torch.bfloat16)
    k = torch.empty((s, tp, dp), device=q_nope.device, dtype=torch.bfloat16)
    a = torch.empty((s, hp, tp), device=q_nope.device, dtype=torch.float32)
    p = torch.empty((s, hp, tp), device=q_nope.device, dtype=torch.bfloat16)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    if (
        d_v % 2 == 0
        and (d - d_v) % 2 == 0
        and all(
            (
                x.stride(-1) == 1
                and x.storage_offset() % 2 == 0
                and all((st % 2 == 0 for st in x.stride()[:-1]))
                for x in (q_nope, q_rope, kv)
            )
        )
    ):
        _pack_both[s * (hp + tp),](
            q_nope.view(torch.int16),
            q_rope.view(torch.int16),
            kv.view(torch.int16),
            indices,
            q.view(torch.int32),
            k.view(torch.int32),
            s,
            h,
            d_v,
            d,
            dp,
            hp,
            t,
            tp,
            q_nope.stride(0) // 2,
            q_nope.stride(1) // 2,
            q_rope.stride(0) // 2,
            q_rope.stride(1) // 2,
            kv.stride(0) // 2,
            indices.stride(0),
            indices.stride(2),
            fnuz,
            num_warps=4,
        )
    else:
        _packq[s, hp](
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
        )
        _packk[s, tp](
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
            num_warps=4,
        )
    _qk[s, hp // 16, tp // 32](
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
        num_warps=4,
        num_stages=1,
    )
    _softmax[s * hp,](
        a,
        p,
        indices,
        tp,
        triton.next_power_of_2(tp),
        t,
        hp,
        indices.stride(0),
        indices.stride(2),
        num_warps=4,
    )
    _pv[s, hp // 16, triton.cdiv(d_v, 64)](
        p, k, out, h, hp, dp, d_v, tp, num_warps=4, num_stages=1
    )
    return out


@triton.jit
def _table2(
    K,
    O,
    N: tl.constexpr,
    NP: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    K0: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // (DP // 2)
    d = x % (DP // 2)
    q = tl.load(K + row * K0 + d, (row < N) & (d < D // 2), 0)
    tl.store(O + x, _expand2(q, FNUZ), x < NP * (DP // 2))


@triton.jit
def _table(
    K,
    O,
    N: tl.constexpr,
    NP: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // DP
    d = x % DP
    q = tl.load(K + row * K0 + d * K2, (row < N) & (d < D), 0)
    tl.store(O + x, _decode(q, FNUZ), x < NP * DP)


@triton.jit
def _dense_qk(
    Q,
    K,
    A,
    DP: tl.constexpr,
    NP: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    BM: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * 32 + tl.arange(0, 32)
    r = tl.arange(0, 64)
    acc = tl.zeros((BM, 32), tl.float32)
    for start in range(DP // 64):
        d = start * 64 + r
        if ROWS % BM == 0:
            q = tl.load(Q + row[:, None] * DP + d[None, :])
        else:
            q = tl.load(
                Q + row[:, None] * DP + d[None, :], row[:, None] < ROWS, 0
            )
        k = tl.load(K + n[None, :] * DP + d[:, None])
        acc = tl.dot(q, k, acc)
    if ROWS % BM == 0:
        tl.store(A + row[:, None] * NP + n[None, :], acc * SCALE)
    else:
        tl.store(
            A + row[:, None] * NP + n[None, :],
            acc * SCALE,
            row[:, None] < ROWS,
        )


@triton.jit
def _packq4_row(
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
    R0: tl.constexpr,
    R1: tl.constexpr,
    FNUZ: tl.constexpr,
    s,
    h,
):
    r = tl.arange(0, triton.next_power_of_2(V // 2))
    x = tl.load(Q + s * Q0 + h * Q1 + r, (h < H) & (r < V // 2), 0)
    b = _expand2(x, FNUZ)
    o = tl.arange(0, triton.next_power_of_2(V // 2))
    tl.store(O + (s * HP + h) * (DP // 2) + o, b, o < V // 2)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        x = tl.load(R + s * R0 + h * R1 + r, (h < H) & (r < (D - V) // 2), 0)
        b = _expand2(x, FNUZ)
        o = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        tl.store(
            O + (s * HP + h) * (DP // 2) + V // 2 + o, b, o < (D - V) // 2
        )
    if DP > D:
        o = tl.arange(0, triton.next_power_of_2(DP // 2))
        tl.store(
            O + (s * HP + h) * (DP // 2) + o, 0, (o >= D // 2) & (o < DP // 2)
        )


@triton.jit
def _packk4_row(
    K,
    I,
    O,
    V: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    K0: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
    s,
    t,
):
    idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2)
    valid = (t < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    r = tl.arange(0, triton.next_power_of_2(V // 2))
    x = tl.load(K + page * K0 + r, r < V // 2, 0)
    b = _expand2(tl.where(valid, x, 0), FNUZ)
    o = tl.arange(0, triton.next_power_of_2(V // 2))
    tl.store(O + (s * TP + t) * (DP // 2) + o, b, o < V // 2)
    if D > V:
        r = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        x = tl.load(K + page * K0 + V // 2 + r, r < (D - V) // 2, 0)
        b = _expand2(tl.where(valid, x, 0), FNUZ)
        o = tl.arange(0, triton.next_power_of_2((D - V) // 2))
        tl.store(
            O + (s * TP + t) * (DP // 2) + V // 2 + o, b, o < (D - V) // 2
        )
    if DP > D:
        o = tl.arange(0, triton.next_power_of_2(DP // 2))
        tl.store(
            O + (s * TP + t) * (DP // 2) + o, 0, (o >= D // 2) & (o < DP // 2)
        )


@triton.jit
def _pack_both(
    Q,
    R,
    K,
    I,
    OQ,
    OK,
    S: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    HP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    K0: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < S * HP:
        _packq4_row(
            Q,
            R,
            OQ,
            H,
            V,
            D,
            DP,
            HP,
            Q0,
            Q1,
            R0,
            R1,
            FNUZ,
            pid // HP,
            pid % HP,
        )
    else:
        row = pid - S * HP
        _packk4_row(
            K, I, OK, V, D, DP, T, TP, K0, I0, I2, FNUZ, row // TP, row % TP
        )


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    t = indices.shape[-1]
    pages, d = (kv.shape[0], kv.shape[-1])
    if t > 8192 or s * t < pages or t * 16 < pages:
        return _sparse(q_nope, q_rope, kv, indices, sm_scale, d_v)
    hp = triton.cdiv(h, 16) * 16
    dp = triton.cdiv(d, 64) * 64
    np = triton.cdiv(pages, 32) * 32
    tp = triton.cdiv(t, 32) * 32
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    q = torch.empty((s, hp, dp), device=q_nope.device, dtype=torch.bfloat16)
    k = torch.empty((np, dp), device=q_nope.device, dtype=torch.bfloat16)
    a = torch.empty((s, hp, np), device=q_nope.device, dtype=torch.float32)
    p = torch.empty((s, hp, tp), device=q_nope.device, dtype=torch.bfloat16)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    if (
        d_v % 2 == 0
        and (d - d_v) % 2 == 0
        and all(
            (
                x.stride(-1) == 1
                and x.storage_offset() % 2 == 0
                and all((st % 2 == 0 for st in x.stride()[:-1]))
                for x in (q_nope, q_rope, kv)
            )
        )
    ):
        _packq4[s, hp](
            q_nope.view(torch.int16),
            q_rope.view(torch.int16),
            q.view(torch.int32),
            h,
            d_v,
            d,
            dp,
            hp,
            q_nope.stride(0) // 2,
            q_nope.stride(1) // 2,
            q_rope.stride(0) // 2,
            q_rope.stride(1) // 2,
            fnuz,
            num_warps=4,
        )
        _table2[triton.cdiv(np * dp // 2, 1024),](
            kv.view(torch.int16),
            k.view(torch.int32),
            pages,
            np,
            d,
            dp,
            kv.stride(0) // 2,
            fnuz,
            1024,
            num_warps=4,
        )
    else:
        _packq[s, hp](
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
        )
        _table[triton.cdiv(np * dp, 1024),](
            kv.view(torch.uint8),
            k,
            pages,
            np,
            d,
            dp,
            kv.stride(0),
            kv.stride(2),
            fnuz,
            1024,
            num_warps=4,
        )
    _dense_qk[triton.cdiv(s * hp, 64), np // 32](
        q, k, a, dp, np, sm_scale, s * hp, 64, num_warps=4, num_stages=1
    )
    vp = triton.cdiv(d_v, 64) * 64
    kg = torch.empty((s, tp, vp), device=q_nope.device, dtype=torch.bfloat16)
    _joint[s * (hp + tp),](
        a,
        indices,
        p,
        k.view(torch.int32),
        kg.view(torch.int32),
        s,
        h,
        hp,
        np,
        t,
        tp,
        dp,
        vp,
        indices.stride(0),
        indices.stride(2),
        triton.next_power_of_2(tp),
        triton.next_power_of_2(dp // 2),
        num_warps=4,
    )
    _pv[s, hp // 16, triton.cdiv(d_v, 64)](
        p, kg, out, h, hp, vp, d_v, tp, num_warps=4, num_stages=1
    )
    return out


@triton.jit
def _indexed_softmax_row(
    A,
    I,
    P,
    H: tl.constexpr,
    HP: tl.constexpr,
    NP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    B: tl.constexpr,
    row,
):
    n = tl.arange(0, B)
    idx = tl.load(I + row // HP * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0) & (row % HP < H)
    a = tl.load(A + row * NP + tl.maximum(idx, 0))
    a = tl.where(valid, a, -float("inf"))
    maximum = tl.max(a, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    p = tl.exp(a - maximum)
    total = tl.sum(p, 0)
    p = p / tl.where(total > 0, total, 1.0)
    tl.store(P + row * TP + n, p, n < TP)


@triton.jit
def _gather_words_row(
    K,
    I,
    O,
    DP: tl.constexpr,
    VP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    B: tl.constexpr,
    s,
    t,
):
    idx = tl.load(I + s * I0 + tl.minimum(t, T - 1) * I2)
    valid = (t < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    d = tl.arange(0, B)
    value = tl.load(K + page * (DP // 2) + d, d < VP // 2, 0)
    tl.store(
        O + (s * TP + t) * (VP // 2) + d,
        tl.where(valid, value, 0),
        d < VP // 2,
    )


@triton.jit
def _joint(
    A,
    I,
    P,
    K,
    KG,
    S: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    NP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    DP: tl.constexpr,
    VP: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    BP: tl.constexpr,
    BW: tl.constexpr,
):
    worker = tl.program_id(0)
    if worker < S * HP:
        _indexed_softmax_row(A, I, P, H, HP, NP, T, TP, I0, I2, BP, worker)
    else:
        row = worker - S * HP
        _gather_words_row(
            K, I, KG, DP, VP, T, TP, I0, I2, BW, row // TP, row % TP
        )


__all__ = ["triton_sparse_mla_fwd"]
