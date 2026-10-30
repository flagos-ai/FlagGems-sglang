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

import triton
import triton.language as tl


@triton.jit
def _tile(
    X,
    W,
    group,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    WS: tl.constexpr,
    E: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    row = group * R + tl.arange(0, R)
    col = tl.arange(0, B)
    if S == H * D:
        offset = row[:, None] * D + col[None, :]
    else:
        offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
    mask = (row[:, None] < M * H) & (col[None, :] < D)
    if M * H % R == 0 and D == B:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
    if SCALE:
        if D == B:
            w = tl.load(W + col * WS).to(tl.float32)
        else:
            w = tl.load(W + col * WS, col < D, other=0).to(tl.float32)
        y = y * w[None, :]
    if M * H % R == 0 and D == B:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, mask)


@triton.jit
def _dispatch(
    Q,
    K,
    V,
    WQ,
    WK,
    group,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    WQS: tl.constexpr,
    WKS: tl.constexpr,
    E: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    GQ: tl.constexpr = (M * HQ + R - 1) // R
    GK: tl.constexpr = (M * HK + R - 1) // R
    if group < GQ:
        _tile(Q, WQ, group, M, HQ, D, SQ, WQS, E, True, B, R)
    elif KV:
        if group < GQ + GK:
            _tile(K, WK, group - GQ, M, HK, D, SK, WKS, E, True, B, R)
        else:
            _tile(V, WQ, group - GQ - GK, M, HK, D, SV, WQS, E, False, B, R)


@triton.jit
def _wide_stream(
    X,
    W,
    pid,
    P: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SCALE: tl.constexpr,
    R: tl.constexpr,
):
    G: tl.constexpr = M * H // R
    row = tl.arange(0, R)
    col = tl.arange(0, D)
    if SCALE:
        w = tl.load(W + col).to(tl.float32)
    for group in range(pid * G // P, (pid + 1) * G // P):
        offset = (group * R + row[:, None]) * D + col[None, :]
        x = tl.load(X + offset).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
        if SCALE:
            y = y * w[None, :]
        tl.store(X + offset, y)


@triton.jit
def _kernel(Q, K, V, WQ, WK, META: tl.constexpr):
    M: tl.constexpr = META[0]
    HQ: tl.constexpr = META[1]
    HK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SV: tl.constexpr = META[6]
    WQS: tl.constexpr = META[7]
    WKS: tl.constexpr = META[8]
    E: tl.constexpr = META[9]
    KV: tl.constexpr = META[10]
    B: tl.constexpr = META[11]
    R: tl.constexpr = META[12]
    PERSIST: tl.constexpr = META[13]
    if R == 128:
        pid = tl.program_id(0)
        if KV:
            PQ0: tl.constexpr = 40 * HQ // (HQ + 2 * HK)
            PQ: tl.constexpr = 1 if PQ0 < 1 else (38 if PQ0 > 38 else PQ0)
            PK: tl.constexpr = (40 - PQ) // 2
            PV: tl.constexpr = 40 - PQ - PK
            if pid < PQ:
                _wide_stream(Q, WQ, pid, PQ, M, HQ, D, E, True, R)
            elif pid < PQ + PK:
                _wide_stream(K, WK, pid - PQ, PK, M, HK, D, E, True, R)
            else:
                _wide_stream(V, WQ, pid - PQ - PK, PV, M, HK, D, E, False, R)
        else:
            _wide_stream(Q, WQ, pid, 40, M, HQ, D, E, True, R)
    elif PERSIST:
        TOTAL: tl.constexpr = (M * HQ + R - 1) // R + (
            2 * ((M * HK + R - 1) // R) if KV else 0
        )
        CHUNK = (TOTAL + tl.num_programs(0) - 1) // tl.num_programs(0)
        for local in range(0, CHUNK):
            group = tl.program_id(0) * CHUNK + local
            if group < TOTAL:
                _dispatch(
                    Q,
                    K,
                    V,
                    WQ,
                    WK,
                    group,
                    M,
                    HQ,
                    HK,
                    D,
                    SQ,
                    SK,
                    SV,
                    WQS,
                    WKS,
                    E,
                    KV,
                    B,
                    R,
                )
    else:
        _dispatch(
            Q,
            K,
            V,
            WQ,
            WK,
            tl.program_id(0),
            M,
            HQ,
            HK,
            D,
            SQ,
            SK,
            SV,
            WQS,
            WKS,
            E,
            KV,
            B,
            R,
        )


def gemma_qkv_rmsnorm(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    m = q.shape[0]
    kv = k is not None and v is not None
    rows = (64 if head_dim <= 128 else 32) if m >= 256 else 16
    wide = (
        m >= 1024
        and head_dim == 128
        and q.element_size() == 2
        and q.stride(0) == num_q_heads * head_dim
        and m * num_q_heads % 128 == 0
        and q_weight.stride(0) == 1
    )
    if wide and kv:
        wide = (
            k.element_size() == v.element_size() == 2
            and k.stride(0) == v.stride(0) == num_kv_heads * head_dim
            and m * num_kv_heads % 128 == 0
            and k_weight.stride(0) == 1
        )
    if wide:
        rows = 128
    groups = ((m * num_q_heads + rows - 1) // rows) + (
        2 * ((m * num_kv_heads + rows - 1) // rows) if kv else 0
    )
    if groups:
        if wide:
            _kernel.run(
                q,
                k if kv else None,
                v if kv else None,
                q_weight,
                k_weight if kv else None,
                (
                    m,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    q.stride(0),
                    k.stride(0) if kv else 0,
                    v.stride(0) if kv else 0,
                    q_weight.stride(0),
                    k_weight.stride(0) if kv else 0,
                    eps,
                    kv,
                    (1 << (head_dim - 1).bit_length()),
                    rows,
                    True,
                ),
                grid=(40,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
                multibuffer=False,
            )
        else:
            _kernel.run(
                q,
                k if kv else None,
                v if kv else None,
                q_weight,
                k_weight if kv else None,
                (
                    m,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    q.stride(0),
                    k.stride(0) if kv else 0,
                    v.stride(0) if kv else 0,
                    q_weight.stride(0),
                    k_weight.stride(0) if kv else 0,
                    eps,
                    kv,
                    (1 << (head_dim - 1).bit_length()),
                    rows,
                    True,
                ),
                grid=(min(groups, 48),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q, k, v)


__all__ = ["gemma_qkv_rmsnorm"]
