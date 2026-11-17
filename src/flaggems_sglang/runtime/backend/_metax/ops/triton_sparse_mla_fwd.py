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
    r = tl.arange(0, B)
    d = tl.arange(0, triton.next_power_of_2(V))
    nq = triton.cdiv(S * H, B)
    if pid < nq:
        row = pid * B + r
        s = row // H
        h = row % H
        q = tl.load(
            Q + s[:, None] * Q0 + h[:, None] * Q1 + d[None, :] * Q2,
            (row[:, None] < S * H) & (d[None, :] < V),
            0,
        )
        tl.store(
            QP + row[:, None] * D + d[None, :],
            _decode(q, FNUZ),
            (row[:, None] < S * H) & (d[None, :] < V),
        )
        if D > V:
            tail = tl.arange(0, triton.next_power_of_2(D - V))
            q = tl.load(
                R + s[:, None] * R0 + h[:, None] * R1 + tail[None, :] * R2,
                (row[:, None] < S * H) & (tail[None, :] < D - V),
                0,
            )
            tl.store(
                QP + row[:, None] * D + V + tail[None, :],
                _decode(q, FNUZ),
                (row[:, None] < S * H) & (tail[None, :] < D - V),
            )
    else:
        row = (pid - nq) * B + r
        if GATHER:
            idx = tl.load(I + row // T * I0 + row % T * I2, row < N, -1)
        else:
            idx = row
        valid = (row < N) & (idx >= 0)
        page = tl.maximum(idx, 0)
        k = tl.load(
            K + page[:, None] * K0 + d[None, :] * K2,
            valid[:, None] & (d[None, :] < V),
            0,
        )
        tl.store(
            KP + row[:, None] * D + d[None, :],
            _decode(k, FNUZ),
            (row[:, None] < N) & (d[None, :] < V),
        )
        if D > V:
            tail = tl.arange(0, triton.next_power_of_2(D - V))
            k = tl.load(
                K + page[:, None] * K0 + (V + tail[None, :]) * K2,
                valid[:, None] & (tail[None, :] < D - V),
                0,
            )
            tl.store(
                KP + row[:, None] * D + V + tail[None, :],
                _decode(k, FNUZ),
                (row[:, None] < N) & (tail[None, :] < D - V),
            )


@triton.jit
def _softmax(A, P, T: tl.constexpr, B: tl.constexpr, PS: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, B)
    a = tl.load(A + row * T + n, n < T, -float("inf"))
    m = tl.max(a, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m)
    z = tl.sum(p, 0)
    tl.store(P + row * PS + n, p * 32768.0, n < T)
    tl.store(A + row * T + T - 1, z)


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
    Z,
    K,
    I,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    PS: tl.constexpr,
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
            P + (s * H + h[:, None]) * PS + t[None, :],
            (h[:, None] < H) & (t[None, :] < T),
            0,
        )
        v = tl.load(
            K + page[:, None] * D + d[None, :],
            valid[:, None] & (d[None, :] < V),
            0,
        )
        acc = tl.dot(p, v, acc)
    z = tl.load(Z + (s * H + h) * T + T - 1, h < H, 1.0)
    acc = acc / (tl.where(z > 0, z, 1.0) * 32768.0)[:, None]
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
    p = (p * 32768.0).to(tl.float16)
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
    acc = acc / (tl.where(z > 0, z, 1.0) * 32768.0)[:, None]
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


@triton.jit
def _scores_split(
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
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
):
    part = tl.program_id(2) % PARTS
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(2) // PARTS * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    idx = tl.load(I + s * I0 + n * I2, n < T, -1)
    valid = (n < T) & (idx >= 0)
    page = tl.maximum(idx, 0)
    if GATHER:
        page = s * T + n
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(triton.cdiv(D, PARTS * BK)):
        d = (part * triton.cdiv(D, PARTS * BK) + start) * BK + r
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
        A + part * ROWS * T + (s * H + h[:, None]) * T + n[None, :],
        tl.where(valid[None, :], acc * SCALE, -float("inf")),
        (h[:, None] < H) & (n[None, :] < T),
    )


@triton.jit
def _softmax_split(
    A,
    P,
    T: tl.constexpr,
    B: tl.constexpr,
    PS: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.arange(0, B)
    a = tl.zeros((B,), tl.float32)
    for part in range(PARTS):
        a += tl.load(A + row * T + n + part * ROWS * T, n < T, -float("inf"))
    m = tl.max(a, 0)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m)
    z = tl.sum(p, 0)
    tl.store(P + row * PS + n, p * 32768.0, n < T)
    tl.store(A + row * T + T - 1, z)


@triton.jit
def _softmax_values_split(
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
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.program_id(2) * BV + tl.arange(0, BV)
    t = tl.arange(0, BT)
    a = tl.zeros((16, BT), tl.float32)
    for part in range(PARTS):
        a += tl.load(
            A + (s * H + h[:, None]) * T + t[None, :] + part * ROWS * T,
            (h[:, None] < H) & (t[None, :] < T),
            -float("inf"),
        )
    m = tl.max(a, 1)
    m = tl.where(m == -float("inf"), 0.0, m)
    p = tl.exp(a - m[:, None])
    z = tl.sum(p, 1)
    p = (p * 32768.0).to(tl.float16)
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
    acc = acc / (tl.where(z > 0, z, 1.0) * 32768.0)[:, None]
    tl.store(
        O + (s * H + h[:, None]) * V + d[None, :],
        acc,
        (h[:, None] < H) & (d[None, :] < V),
    )


def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v):
    s, h, _ = q_nope.shape
    d, t = (kv.shape[-1], indices.shape[-1])
    gather = s * t < kv.shape[0]
    nk = s * t if gather else kv.shape[0]
    parts = (
        3
        if d == 576 and s * triton.cdiv(h, 16) * triton.cdiv(t, 64) < 64
        else 1
    )
    fnuz = q_nope.dtype == torch.float8_e4m3fnuz
    qp = torch.empty((s, h, d), device=q_nope.device, dtype=torch.float16)
    kp = torch.empty((nk, d), device=q_nope.device, dtype=torch.float16)
    a = torch.empty(
        (parts, s, h, t), device=q_nope.device, dtype=torch.float32
    )
    out = torch.empty(
        (1, s, h, d_v), device=q_nope.device, dtype=torch.bfloat16
    )
    _packboth.run(
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
        4,
        num_warps=4,
        grid=(triton.cdiv(s * h, 4) + triton.cdiv(nk, 4),),
        warmup=False,
    )
    if parts > 1:
        _scores_split.run(
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
            s * h,
            parts,
            num_warps=4,
            num_stages=3,
            grid=(s, triton.cdiv(h, 16), triton.cdiv(t, 64) * parts),
            warmup=False,
        )
    else:
        _scores.run(
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
            num_stages=3,
            grid=(s, triton.cdiv(h, 16), triton.cdiv(t, 64)),
            warmup=False,
        )
    if t <= 256:
        if parts > 1:
            _softmax_values_split.run(
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
                s * h,
                parts,
                num_warps=4,
                num_stages=2,
                grid=(s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)),
                warmup=False,
            )
        else:
            _softmax_values.run(
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
                grid=(s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)),
                warmup=False,
            )
    else:
        p = a.view(torch.float16)
        if parts > 1:
            _softmax_split.run(
                a,
                p,
                t,
                triton.next_power_of_2(t),
                2 * t,
                s * h,
                parts,
                num_warps=4,
                grid=(s * h,),
                warmup=False,
            )
        else:
            _softmax.run(
                a,
                p,
                t,
                triton.next_power_of_2(t),
                2 * t,
                num_warps=4,
                grid=(s * h,),
                warmup=False,
            )
        pv_bn = (
            128 if s * triton.cdiv(h, 16) * triton.cdiv(d_v, 64) < 64 else 64
        )
        _values.run(
            p,
            a,
            kp,
            indices,
            out,
            h,
            d,
            d_v,
            t,
            2 * t,
            indices.stride(0),
            indices.stride(2),
            gather,
            pv_bn,
            64,
            num_warps=4,
            num_stages=3,
            grid=(s, triton.cdiv(h, 16), triton.cdiv(d_v, 64)),
            warmup=False,
        )
    return out


__all__ = ["triton_sparse_mla_fwd"]
