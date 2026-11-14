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
def _manual_decode(q, FNUZ: tl.constexpr):
    q = q.to(tl.uint32)
    a = q & 127
    e = a >> 3
    m = a & 7
    if FNUZ:
        bias = 119
    else:
        bias = 120
    normal = e + bias << 7 | m << 4
    sub = tl.where(
        m >= 4,
        bias * 128 + (m - 4) * 32,
        tl.where(
            m >= 2,
            (bias - 1) * 128 + (m - 2) * 64,
            tl.where(m == 1, (bias - 2) * 128, 0),
        ),
    )
    bits = tl.where(e == 0, sub, normal) | (q & 128) << 8
    if FNUZ:
        bits = tl.where(q == 128, 32704, bits)
    else:
        bits = tl.where(a == 127, 32704, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _packboth(
    Q,
    R,
    K,
    I,
    QP,
    KP,
    S: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    T: tl.constexpr,
    GATHER: tl.constexpr,
    FNUZ: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    nq = triton.cdiv(S * H * D, B)
    if pid < nq:
        x = pid * B + tl.arange(0, B)
        s = x // (H * D)
        h = x // D % H
        d = x % D
        q = tl.load(Q + s * Q0 + h * Q1 + d * Q2, (s < S) & (d < V), 0)
        r = tl.load(R + s * R0 + h * R1 + (d - V) * R2, (s < S) & (d >= V), 0)
        tl.store(QP + x, _decode(tl.where(d < V, q, r), FNUZ), x < S * H * D)
    else:
        x = (pid - nq) * B + tl.arange(0, B)
        if GATHER:
            row = x // D
            idx = tl.load(I + row // T * I0 + row % T * I2, row < N, -1)
            q = tl.load(
                K + tl.maximum(idx, 0) * K0 + x % D * K2,
                (x < N * D) & (idx >= 0),
                0,
            )
        else:
            q = tl.load(K + x // D * K0 + x % D * K2, x < N * D, 0)
        tl.store(KP + x, _decode(q, FNUZ), x < N * D)


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
def _scores(
    Q,
    K,
    I,
    A,
    H: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    GATHER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + n
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(triton.cdiv(D, BK)):
        d = start * BK + r
        q = tl.load(
            Q + (s * H + h[:, None]) * D + d[None, :],
            (h[:, None] < H) & (d[None, :] < D),
            0,
        )
        k = tl.load(
            K + page[None, :] * D + d[:, None],
            valid[None, :] & (d[:, None] < D),
            0,
        )
        acc = tl.dot(q, k, acc)
    tl.store(
        A + (s * H + h[:, None]) * T + n[None, :],
        tl.where(valid[None, :], acc * SCALE, -float("inf")),
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _values(
    P,
    K,
    I,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    GATHER: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    n = tl.arange(0, BN)
    acc = tl.zeros((16, BV), tl.float32)
    for start in range(triton.cdiv(T, BN)):
        t = start * BN + n
        idx = tl.load(I + s * I0 + t * I2, t < T, -1)
        valid = (t < T) & (idx >= 0)
        page = tl.maximum(idx, 0)
        if GATHER:
            page = s * T + t
        p = tl.load(
            P + (s * H + h[:, None]) * T + t[None, :],
            (h[:, None] < H) & (t[None, :] < T),
            0,
        )
        v = tl.load(
            K + page[:, None] * D + d[None, :],
            valid[:, None] & (d[None, :] < V),
            0,
        )
        acc = tl.dot(p, v, acc)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


@triton.jit
def _softmax_values(
    A,
    K,
    I,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    GATHER: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    t = tl.arange(0, BT)
    a = tl.load(
        A + (s * H + h[:, None]) * T + t[None, :],
        (h[:, None] < H) & (t[None, :] < T),
        -float("inf"),
    )
    m = tl.max(a, 1)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m[:, None])
    z = tl.sum(p, 1)
    p = (p / tl.where(z > 0, z, 1.0)[:, None]).to(tl.bfloat16)
    idx = tl.load(I + s * I0 + t * I2, t < T, -1)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + t
    v = tl.load(
        K + page[:, None] * D + d[None, :],
        (t[:, None] < T) & (idx[:, None] >= 0) & (d[None, :] < V),
        0,
    )
    acc = tl.dot(p, v)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def _prepared(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d, t = (kv.shape[-1], indices.shape[-1])
    gather = s * t < kv.shape[0]
    nk = s * t if gather else kv.shape[0]
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    qp = torch.empty((s, h, d), device=q_nope.device, dtype=torch.bfloat16)
    kp = torch.empty((nk, d), device=q_nope.device, dtype=torch.bfloat16)
    a = torch.empty((s, h, t), device=q_nope.device, dtype=torch.float32)
    p = torch.empty((s, h, t), device=q_nope.device, dtype=torch.bfloat16)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    _packboth[triton.cdiv(s * h * d, 1024) + triton.cdiv(nk * d, 1024),](
        q_nope.view(torch.uint8),
        q_rope.view(torch.uint8),
        kv.view(torch.uint8),
        indices,
        qp,
        kp,
        s,
        h,
        nk,
        d,
        d_v,
        *q_nope.stride(),
        *q_rope.stride(),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        t,
        gather,
        fnuz,
        1024,
        num_warps=4,
    )
    _scores[s, triton.cdiv(h, 16), triton.cdiv(t, 64)](
        qp,
        kp,
        indices,
        a,
        h,
        d,
        t,
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        gather,
        64,
        64,
        num_warps=4,
        num_stages=2,
    )
    if t <= 256:
        _softmax_values[s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)](
            a,
            kp,
            indices,
            out,
            h,
            d,
            d_v,
            t,
            indices.stride(0),
            indices.stride(2),
            gather,
            triton.next_power_of_2(t),
            64,
            num_warps=4,
            num_stages=2,
        )
    else:
        _softmax[s * h,](a, p, t, triton.next_power_of_2(t), num_warps=4)
        _values[s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)](
            p,
            kp,
            indices,
            out,
            h,
            d,
            d_v,
            t,
            indices.stride(0),
            indices.stride(2),
            gather,
            128,
            64,
            num_warps=4,
            num_stages=2,
        )
    return out


@triton.jit
def _direct_scores(
    Q,
    R,
    K,
    I,
    A,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (idx >= 0) & (n < T)
    idx = tl.maximum(idx, 0)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(triton.cdiv(V, 64)):
        d = start * 64 + k
        q = _decode(
            tl.load(
                Q + s * Q0 + h[:, None] * Q1 + d[None, :] * Q2,
                (h[:, None] < H) & (d[None, :] < V),
                0,
            ),
            FNUZ,
        )
        v = _decode(
            tl.load(
                K + idx[None, :] * K0 + d[:, None] * K2,
                valid[None, :] & (d[:, None] < V),
                0,
            ),
            FNUZ,
        )
        acc = tl.dot(q.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    for start in range(triton.cdiv(D - V, 64)):
        d = start * 64 + k
        q = _decode(
            tl.load(
                R + s * R0 + h[:, None] * R1 + d[None, :] * R2,
                (h[:, None] < H) & (d[None, :] < D - V),
                0,
            ),
            FNUZ,
        )
        v = _decode(
            tl.load(
                K + idx[None, :] * K0 + (V + d[:, None]) * K2,
                valid[None, :] & (d[:, None] < D - V),
                0,
            ),
            FNUZ,
        )
        acc = tl.dot(q.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    score = tl.where(valid[None, :], acc * SCALE, -float("inf"))
    tl.store(
        A + (s * H + h[:, None]) * T + n[None, :],
        score,
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _direct_values(
    P,
    K,
    I,
    O,
    H: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * 64 + tl.arange(0, 64)
    n = tl.arange(0, 64)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(triton.cdiv(T, 64)):
        t = start * 64 + n
        idx = tl.load(I + s * I0 + t * I2, t < T, -1)
        valid = (idx >= 0) & (t < T)
        idx = tl.maximum(idx, 0)
        p = tl.load(
            P + (s * H + h[:, None]) * T + t[None, :],
            (h[:, None] < H) & (t[None, :] < T),
            0,
        )
        v = _decode(
            tl.load(
                K + idx[:, None] * K0 + d[None, :] * K2,
                valid[:, None] & (d[None, :] < V),
                0,
            ),
            FNUZ,
        ).to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def _direct(q_nope, q_rope, kv, indices, sm_scale, d_v):
    kv_bytes = kv.view(torch.uint8)
    s, h, _ = q_nope.shape
    d = kv.shape[-1]
    t = indices.shape[-1]
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    scores = torch.empty((s, h, t), device=q_nope.device, dtype=torch.float32)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    _direct_scores[s, triton.cdiv(h, 16), triton.cdiv(t, 64)](
        q_nope.view(torch.uint8),
        q_rope.view(torch.uint8),
        kv_bytes,
        indices,
        scores,
        h,
        d,
        d_v,
        t,
        *q_nope.stride(),
        *q_rope.stride(),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        fnuz,
        num_warps=4,
        num_stages=2,
    )
    if t <= 256:
        _direct_softmax_direct_values[
            s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)
        ](
            scores,
            kv_bytes,
            indices,
            out,
            h,
            d,
            d_v,
            t,
            indices.stride(0),
            indices.stride(2),
            False,
            triton.next_power_of_2(t),
            64,
            kv.stride(0),
            kv.stride(2),
            fnuz,
            num_warps=4,
            num_stages=2,
        )
    else:
        probs = torch.empty(
            (s, h, t), device=q_nope.device, dtype=torch.bfloat16
        )
        _softmax[s * h,](
            scores, probs, t, triton.next_power_of_2(t), num_warps=4
        )
        _direct_values[s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)](
            probs,
            kv_bytes,
            indices,
            out,
            h,
            d_v,
            t,
            kv.stride(0),
            kv.stride(2),
            indices.stride(0),
            indices.stride(2),
            fnuz,
            num_warps=4,
            num_stages=1,
        )
    return out


@triton.jit
def _direct_softmax_direct_values(
    A,
    K,
    I,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    GATHER: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    t = tl.arange(0, BT)
    a = tl.load(
        A + (s * H + h[:, None]) * T + t[None, :],
        (h[:, None] < H) & (t[None, :] < T),
        -float("inf"),
    )
    m = tl.max(a, 1)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m[:, None])
    z = tl.sum(p, 1)
    p = (p / tl.where(z > 0, z, 1.0)[:, None]).to(tl.bfloat16)
    idx = tl.load(I + s * I0 + t * I2, t < T, -1)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + t
    v = _decode(
        tl.load(
            K + page[:, None] * K0 + d[None, :] * K2,
            (t[:, None] < T) & (idx[:, None] >= 0) & (d[None, :] < V),
            0,
        ),
        FNUZ,
    ).to(tl.bfloat16)
    acc = tl.dot(p, v)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def _hybrid(q_nope, q_rope, kv, indices, sm_scale, d_v):
    if indices.shape[-1] <= 256:
        return _direct(q_nope, q_rope, kv, indices, sm_scale, d_v)
    return _prepared(q_nope, q_rope, kv, indices, sm_scale, d_v)


@triton.jit
def _manual_legacy_decode(q, FNUZ: tl.constexpr):
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
def _legacy_scores(
    Q,
    R,
    K,
    I,
    A,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    SCALE: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (idx >= 0) & (n < T)
    idx = tl.maximum(idx, 0)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(triton.cdiv(V, 64)):
        d = start * 64 + k
        q = _legacy_decode(
            tl.load(
                Q + s * Q0 + h[:, None] * Q1 + d[None, :] * Q2,
                (h[:, None] < H) & (d[None, :] < V),
                0,
            ),
            FNUZ,
        )
        v = _legacy_decode(
            tl.load(
                K + idx[None, :] * K0 + d[:, None] * K2,
                valid[None, :] & (d[:, None] < V),
                0,
            ),
            FNUZ,
        )
        acc = tl.dot(q.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    for start in range(triton.cdiv(D - V, 64)):
        d = start * 64 + k
        q = _legacy_decode(
            tl.load(
                R + s * R0 + h[:, None] * R1 + d[None, :] * R2,
                (h[:, None] < H) & (d[None, :] < D - V),
                0,
            ),
            FNUZ,
        )
        v = _legacy_decode(
            tl.load(
                K + idx[None, :] * K0 + (V + d[:, None]) * K2,
                valid[None, :] & (d[:, None] < D - V),
                0,
            ),
            FNUZ,
        )
        acc = tl.dot(q.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    score = tl.where(valid[None, :], acc * SCALE, -float("inf"))
    tl.store(
        A + (s * H + h[:, None]) * T + n[None, :],
        score,
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _legacy_values(
    P,
    K,
    I,
    O,
    H: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    K0: tl.constexpr,
    K2: tl.constexpr,
    I0: tl.constexpr,
    I2: tl.constexpr,
    FNUZ: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * 64 + tl.arange(0, 64)
    n = tl.arange(0, 64)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(triton.cdiv(T, 64)):
        t = start * 64 + n
        idx = tl.load(I + s * I0 + t * I2, t < T, -1)
        valid = (idx >= 0) & (t < T)
        idx = tl.maximum(idx, 0)
        p = tl.load(
            P + (s * H + h[:, None]) * T + t[None, :],
            (h[:, None] < H) & (t[None, :] < T),
            0,
        )
        v = _legacy_decode(
            tl.load(
                K + idx[:, None] * K0 + d[None, :] * K2,
                valid[:, None] & (d[None, :] < V),
                0,
            ),
            FNUZ,
        ).to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def _legacy(q_nope, q_rope, kv, indices, sm_scale, d_v):
    kv_bytes = kv.view(torch.uint8)
    s, h, _ = q_nope.shape
    d = kv.shape[-1]
    t = indices.shape[-1]
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    scores = torch.empty((s, h, t), device=q_nope.device, dtype=torch.float32)
    probs = torch.empty((s, h, t), device=q_nope.device, dtype=torch.bfloat16)
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    _legacy_scores[s, triton.cdiv(h, 16), triton.cdiv(t, 64)](
        q_nope.view(torch.uint8),
        q_rope.view(torch.uint8),
        kv_bytes,
        indices,
        scores,
        h,
        d,
        d_v,
        t,
        *q_nope.stride(),
        *q_rope.stride(),
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        sm_scale,
        fnuz,
        num_warps=4,
        num_stages=2,
    )
    _softmax[s * h,](scores, probs, t, triton.next_power_of_2(t), num_warps=4)
    _legacy_values[s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)](
        probs,
        kv_bytes,
        indices,
        out,
        h,
        d_v,
        t,
        kv.stride(0),
        kv.stride(2),
        indices.stride(0),
        indices.stride(2),
        fnuz,
        num_warps=4,
        num_stages=1,
    )
    return out


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    t = indices.shape[-1]
    if t > 256 and s * t * triton.cdiv(h, 16) <= 32768:
        return _legacy(q_nope, q_rope, kv, indices, sm_scale, d_v)
    return _hybrid(q_nope, q_rope, kv, indices, sm_scale, d_v)


@triton.jit
def _decode(q, FNUZ: tl.constexpr):
    if FNUZ:
        return _manual_decode(q, FNUZ)
    else:
        q = q.to(tl.uint8)
        value = q.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        value = tl.where(q & 127 == 127, float("nan"), value).to(tl.bfloat16)
        return value


@triton.jit
def _legacy_decode(q, FNUZ: tl.constexpr):
    if FNUZ:
        return _manual_legacy_decode(q, FNUZ)
    else:
        q = q.to(tl.uint8)
        value = q.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        value = tl.where(q & 127 == 127, float("nan"), value).to(tl.bfloat16)
        return value.to(tl.float32)


__all__ = ["triton_sparse_mla_fwd"]
