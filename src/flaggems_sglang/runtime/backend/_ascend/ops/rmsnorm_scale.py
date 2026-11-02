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
def _offset(row, SH: tl.constexpr, ST: tl.constexpr):
    base = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SH) - 1, -1, -1):
        base += (row % SH[j]).to(tl.int64) * ST[j]
        row //= SH[j]
    return base


@triton.jit
def _row(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    SSH: tl.constexpr,
    SST: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        x = tl.load(
            X + _offset(row, XSH, XST) + c * XST[-1], c < D, other=0
        ).to(tl.float32)
        w = tl.load(W + c, c < D, other=0).to(tl.float32)
        s = tl.load(
            S + _offset(row // REP, SSH, SST) + c * SST[-1], c < D, other=0
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = (x * inv).to(tl.bfloat16).to(tl.float32)
        z = (n * w).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * D + c, z * s, c < D)


@triton.jit
def _lean(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    B: tl.constexpr = P[4]
    col = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        w = tl.load(W + col, col < D, other=0).to(tl.float32)
        s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = (x * inv).to(tl.bfloat16).to(tl.float32)
        z = (n * w).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * D + col, z * s, col < D)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _small(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    B: tl.constexpr = P[4]
    col = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        w = tl.load(W + col, col < D, other=0).to(tl.float32)
        s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = (x * inv).to(tl.bfloat16).to(tl.float32)
        z = (n * w).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * D + col, z * s, col < D)


def _flat_rows(t):
    if t.ndim <= 2:
        return True
    row_stride = None
    rows = 1
    for size, stride in zip(reversed(t.shape[:-1]), reversed(t.stride()[:-1])):
        if size == 1:
            continue
        if row_stride is None:
            row_stride = stride
        elif stride != row_stride * rows:
            return False
        rows *= size
    return True


@triton.jit
def _wide_partials(
    X,
    P,
    M: tl.constexpr,
    D: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.arange(0, B)
    TILES: tl.constexpr = tl.cdiv(D, B)
    for block in tl.range(
        tl.program_id(0), M * TILES, tl.num_programs(0), num_stages=1
    ):
        row = block // TILES
        col = block % TILES * B + c
        x = tl.load(
            X + _offset(row, XSH, XST) + col * XST[-1], col < D, other=0
        ).to(tl.float32)
        tl.store(P + block, tl.sum(x * x, 0))


@triton.jit
def _wide_finish(
    X,
    W,
    S,
    P,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    SSH: tl.constexpr,
    SST: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    BT: tl.constexpr,
):
    c = tl.arange(0, B)
    t = tl.arange(0, BT)
    TILES: tl.constexpr = tl.cdiv(D, B)
    for block in tl.range(
        tl.program_id(0), M * TILES, tl.num_programs(0), num_stages=1
    ):
        row = block // TILES
        col = block % TILES * B + c
        partials = tl.load(P + row * TILES + t, t < TILES, other=0)
        inv = tl.rsqrt(tl.sum(partials, 0) / D + EPS)
        x = tl.load(
            X + _offset(row, XSH, XST) + col * XST[-1], col < D, other=0
        ).to(tl.float32)
        w = tl.load(W + col, col < D, other=0).to(tl.float32)
        s = tl.load(
            S + _offset(row // REP, SSH, SST) + col * SST[-1], col < D, other=0
        ).to(tl.float32)
        n = (x * inv).to(tl.bfloat16).to(tl.float32)
        z = (n * w).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * D + col, z * s, col < D)


@triton.jit
def _asc_parts2(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
):
    c = tl.arange(0, D // 2)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0), M // R, tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        c0 = c + 0 * (D // 2)
        x0 = tl.load(X + row[:, None] * D + c0[None, :]).to(tl.float32)
        c1 = c + 1 * (D // 2)
        x1 = tl.load(X + row[:, None] * D + c1[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, 1) / D + EPS)
        w0 = tl.load(W + c0).to(tl.float32)
        s0 = tl.load(S + row[:, None] // REP * D + c0[None, :]).to(tl.float32)
        n0 = (x0 * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        z0 = (n0 * w0[None, :]).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row[:, None] * D + c0[None, :], z0 * s0)
        w1 = tl.load(W + c1).to(tl.float32)
        s1 = tl.load(S + row[:, None] // REP * D + c1[None, :]).to(tl.float32)
        n1 = (x1 * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        z1 = (n1 * w1[None, :]).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row[:, None] * D + c1[None, :], z1 * s1)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _bf_small(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    B: tl.constexpr = P[4]
    col = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        w = tl.load(W + col, col < D, other=0)
        s = tl.load(S + row // REP * D + col, col < D, other=0)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = (x * inv).to(tl.bfloat16)
        z = (n * w).to(tl.bfloat16)
        tl.store(Y + row * D + col, z * s, col < D)


@triton.jit
def _bf_tile(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    CONT: tl.constexpr,
    ROUND: tl.constexpr,
):
    c = tl.arange(0, B)
    r = tl.arange(0, R)
    w = tl.load(W + c, c < D, other=0)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(M, R), tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        if CONT:
            x = tl.load(X + row[:, None] * D + c[None, :]).to(tl.float32)
            if REP == 1:
                s = tl.load(S + row[:, None] * D + c[None, :])
            else:
                s = tl.load(S + row[:, None] // REP * D + c[None, :])
        else:
            x = tl.load(
                X + row[:, None] * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
            s = tl.load(
                S + row[:, None] // REP * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            )
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        if ROUND:
            n = (x * inv[:, None]).to(tl.bfloat16)
            z = (n * w[None, :]).to(tl.bfloat16)
        else:
            n = (x * inv[:, None]).to(tl.bfloat16)
            z = (n * w[None, :]).to(tl.bfloat16)
        if CONT:
            tl.store(Y + row[:, None] * D + c[None, :], z * s)
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                z * s,
                (row[:, None] < M) & (c[None, :] < D),
            )


@triton.jit
def _small_exact3(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
):
    c = tl.arange(0, D // 3)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0), M // R, tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        c0 = c + 0 * (D // 3)
        x0 = tl.load(X + row[:, None] * D + c0[None, :]).to(tl.float32)
        c1 = c + 1 * (D // 3)
        x1 = tl.load(X + row[:, None] * D + c1[None, :]).to(tl.float32)
        c2 = c + 2 * (D // 3)
        x2 = tl.load(X + row[:, None] * D + c2[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2, 1) / D + EPS)
        n0 = (x0 * inv[:, None]).to(tl.bfloat16)
        w0 = tl.load(W + c0)
        z0 = (n0 * w0[None, :]).to(tl.bfloat16)
        s0 = tl.load(S + row[:, None] // REP * D + c0[None, :])
        tl.store(Y + row[:, None] * D + c0[None, :], z0 * s0)
        n1 = (x1 * inv[:, None]).to(tl.bfloat16)
        w1 = tl.load(W + c1)
        z1 = (n1 * w1[None, :]).to(tl.bfloat16)
        s1 = tl.load(S + row[:, None] // REP * D + c1[None, :])
        tl.store(Y + row[:, None] * D + c1[None, :], z1 * s1)
        n2 = (x2 * inv[:, None]).to(tl.bfloat16)
        w2 = tl.load(W + c2)
        z2 = (n2 * w2[None, :]).to(tl.bfloat16)
        s2 = tl.load(S + row[:, None] // REP * D + c2[None, :])
        tl.store(Y + row[:, None] * D + c2[None, :], z2 * s2)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _delayed_tile_compact(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    R: tl.constexpr = P[4]
    MODE: tl.constexpr = P[5]
    c = tl.arange(0, D)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0), M // R, tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        x = tl.load(X + row[:, None] * D + c[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        n = (x * inv[:, None]).to(tl.bfloat16)
        w = tl.load(W + c)
        z = (n * w[None, :]).to(tl.bfloat16)
        s = tl.load(S + row[:, None] // REP * D + c[None, :])
        if MODE == 1:
            tl.store(Y + row[:, None] * D + c[None, :], z * s)
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                z.to(tl.float32) * s.to(tl.float32),
            )


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _bf_tile_compact(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    B: tl.constexpr = P[4]
    R: tl.constexpr = P[5]
    CONT: tl.constexpr = P[6]
    ROUND: tl.constexpr = P[7]
    c = tl.arange(0, B)
    r = tl.arange(0, R)
    w = tl.load(W + c, c < D, other=0)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(M, R), tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        if CONT:
            x = tl.load(X + row[:, None] * D + c[None, :]).to(tl.float32)
            if REP == 1:
                s = tl.load(S + row[:, None] * D + c[None, :])
            else:
                s = tl.load(S + row[:, None] // REP * D + c[None, :])
        else:
            x = tl.load(
                X + row[:, None] * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
            s = tl.load(
                S + row[:, None] // REP * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            )
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        if ROUND:
            n = (x * inv[:, None]).to(tl.bfloat16)
            z = (n * w[None, :]).to(tl.bfloat16)
        else:
            n = (x * inv[:, None]).to(tl.bfloat16)
            z = (n * w[None, :]).to(tl.bfloat16)
        if CONT:
            tl.store(Y + row[:, None] * D + c[None, :], z * s)
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                z * s,
                (row[:, None] < M) & (c[None, :] < D),
            )


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _small_exact3_compact(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    R: tl.constexpr = P[4]
    c = tl.arange(0, D // 3)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0), M // R, tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        c0 = c + 0 * (D // 3)
        x0 = tl.load(X + row[:, None] * D + c0[None, :]).to(tl.float32)
        c1 = c + 1 * (D // 3)
        x1 = tl.load(X + row[:, None] * D + c1[None, :]).to(tl.float32)
        c2 = c + 2 * (D // 3)
        x2 = tl.load(X + row[:, None] * D + c2[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2, 1) / D + EPS)
        n0 = (x0 * inv[:, None]).to(tl.bfloat16)
        w0 = tl.load(W + c0)
        z0 = (n0 * w0[None, :]).to(tl.bfloat16)
        s0 = tl.load(S + row[:, None] // REP * D + c0[None, :])
        tl.store(Y + row[:, None] * D + c0[None, :], z0 * s0)
        n1 = (x1 * inv[:, None]).to(tl.bfloat16)
        w1 = tl.load(W + c1)
        z1 = (n1 * w1[None, :]).to(tl.bfloat16)
        s1 = tl.load(S + row[:, None] // REP * D + c1[None, :])
        tl.store(Y + row[:, None] * D + c1[None, :], z1 * s1)
        n2 = (x2 * inv[:, None]).to(tl.bfloat16)
        w2 = tl.load(W + c2)
        z2 = (n2 * w2[None, :]).to(tl.bfloat16)
        s2 = tl.load(S + row[:, None] // REP * D + c2[None, :])
        tl.store(Y + row[:, None] * D + c2[None, :], z2 * s2)


@triton.jit
def _tile(X, W, S, Y, PARAMS: tl.constexpr):
    M: tl.constexpr = PARAMS[0]
    D: tl.constexpr = PARAMS[1]
    REP: tl.constexpr = PARAMS[2]
    EPS: tl.constexpr = PARAMS[3]
    B: tl.constexpr = PARAMS[4]
    R: tl.constexpr = PARAMS[5]
    CONT: tl.constexpr = PARAMS[6]
    ROUND: tl.constexpr = PARAMS[7]
    c = tl.arange(0, B)
    r = tl.arange(0, R)
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(M, R), tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        if CONT:
            x = tl.load(X + row[:, None] * D + c[None, :]).to(tl.float32)
            if REP == 1:
                s = tl.load(S + row[:, None] * D + c[None, :]).to(tl.float32)
            else:
                s = tl.load(S + row[:, None] // REP * D + c[None, :]).to(
                    tl.float32
                )
        else:
            x = tl.load(
                X + row[:, None] * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
            s = tl.load(
                S + row[:, None] // REP * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        if ROUND:
            n = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
            z = (n * w[None, :]).to(tl.bfloat16).to(tl.float32)
        else:
            n = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
            z = (n * w[None, :]).to(tl.bfloat16).to(tl.float32)
        if CONT:
            tl.store(Y + row[:, None] * D + c[None, :], z * s)
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                z * s,
                (row[:, None] < M) & (c[None, :] < D),
            )


@triton.jit
def _delayed_tile(X, W, S, Y, PARAMS: tl.constexpr):
    M: tl.constexpr = PARAMS[0]
    D: tl.constexpr = PARAMS[1]
    REP: tl.constexpr = PARAMS[2]
    EPS: tl.constexpr = PARAMS[3]
    R: tl.constexpr = PARAMS[4]
    MODE: tl.constexpr = PARAMS[5]
    c = tl.arange(0, D)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0), M // R, tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        x = tl.load(X + row[:, None] * D + c[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        n = (x * inv[:, None]).to(tl.bfloat16)
        w = tl.load(W + c)
        z = (n * w[None, :]).to(tl.bfloat16)
        s = tl.load(S + row[:, None] // REP * D + c[None, :])
        if MODE == 1:
            tl.store(Y + row[:, None] * D + c[None, :], z * s)
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                z.to(tl.float32) * s.to(tl.float32),
            )


def rmsnorm_scale(x, weight, scale, eps):
    xshape = x.shape
    sshape = scale.shape
    if not xshape or not sshape:
        return None
    if not x.dtype == weight.dtype == scale.dtype == torch.bfloat16:
        return None
    if not x.device == weight.device == scale.device:
        return None
    d = xshape[-1]
    if (
        not 0 < d <= 8192
        or weight.shape != (d,)
        or (not weight.is_contiguous())
        or (sshape[-1] != d)
    ):
        return None
    m, sm = (xshape.numel() // d, sshape.numel() // d)
    if sm == 0 or m % sm != 0:
        return None
    xc, sc = (x.is_contiguous(), scale.is_contiguous())
    if not xc and (not _flat_rows(x)) or (not sc and (not _flat_rows(scale))):
        return None
    y = x.new_empty(xshape)
    if m:
        b = 1 << max(6, (d - 1).bit_length())
        if d > 4096:
            tiles = triton.cdiv(d, 512)
            partials = torch.empty(
                (m * tiles,), dtype=torch.float32, device=x.device
            )
            _wide_partials.run(
                x,
                partials,
                m,
                d,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                512,
                grid=(min(m * tiles, 40),),
                warmup=False,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            _wide_finish.run(
                x,
                weight,
                scale,
                partials,
                y,
                m,
                d,
                m // sm,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                tuple(scale.shape[:-1]),
                tuple(scale.stride()),
                float(eps),
                512,
                triton.next_power_of_2(tiles),
                grid=(min(m * tiles, 40),),
                warmup=False,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        elif xc and sc:
            if 32 <= m < 1024 and m % 4 == 0 and (d in (1536, 2048, 4096)):
                rows = 2 if m < 64 or d == 4096 else 4
                if d == 1536 and m >= 64:
                    _small_exact3_compact.run(
                        x,
                        weight,
                        scale,
                        y,
                        (m, d, m // sm, float(eps), rows),
                        grid=(min(m // rows, 40),),
                        warmup=False,
                        num_warps=4,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                elif d == 1536:
                    _bf_tile_compact.run(
                        x,
                        weight,
                        scale,
                        y,
                        (m, d, m // sm, float(eps), b, rows, False, True),
                        grid=(min(m // rows, 40),),
                        warmup=False,
                        num_warps=4,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _delayed_tile_compact.run(
                        x,
                        weight,
                        scale,
                        y,
                        (m, d, m // sm, float(eps), rows, 1),
                        grid=(min(m // rows, 40),),
                        warmup=False,
                        num_warps=4,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
            elif 1024 <= m < 8192 and m % 4 == 0 and (d == 2048):
                _delayed_tile.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), 4, 1),
                    grid=(min(m // 4, 40),),
                    warmup=False,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            elif 1024 <= m < 8192 and m % 2 == 0 and (d == 4096):
                _delayed_tile.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), 2, 1),
                    grid=(min(m // 2, 40),),
                    warmup=False,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            elif m >= 8192:
                rows = 4 if d <= 2048 else 2
                complete = m % rows == 0 and d == b
                limit = 40 if rows == 4 else 80
                _tile.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), b, rows, complete, False),
                    grid=(min((m + rows - 1) // rows, limit),),
                    warmup=False,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            else:
                (_bf_small if m < 1024 else _lean).run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), b),
                    grid=(min(m, 40),),
                    warmup=False,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
        else:
            _row.run(
                x,
                weight,
                scale,
                y,
                m,
                d,
                m // sm,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                tuple(scale.shape[:-1]),
                tuple(scale.stride()),
                float(eps),
                b,
                grid=(min(m, 40),),
                warmup=False,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return y


__all__ = ["rmsnorm_scale"]
