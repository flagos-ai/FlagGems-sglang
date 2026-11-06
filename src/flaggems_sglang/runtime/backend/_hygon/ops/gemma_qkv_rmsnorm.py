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
    if (M < 512 and R >= 16) or (M >= 512 and D >= 256 and D <= 512):
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, B)
        offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
        mask = (row[:, None] < M * H) & (col[None, :] < D)
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
        if SCALE:
            w = tl.load(
                tl.broadcast_to(W + col[None, :] * WS, (R, B)), mask, other=0
            ).to(tl.float32)
            y = y * w
        tl.store(X + offset, y, mask)
    else:
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, B)
        offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
        mask = (row[:, None] < M * H) & (col[None, :] < D)
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / D + E)
        if SCALE:
            w = tl.load(W + col * WS, col < D, other=0).to(tl.float32)
            y = y * w[None, :]
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
    if PERSIST:
        TOTAL: tl.constexpr = (M * HQ + R - 1) // R + (
            2 * ((M * HK + R - 1) // R) if KV else 0
        )
        for group in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
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
    rows = 16 if m >= 512 else 4
    if m < 512 and m * num_q_heads >= 4096:
        rows = 16
    groups = ((m * num_q_heads + rows - 1) // rows) + (
        2 * ((m * num_kv_heads + rows - 1) // rows) if kv else 0
    )
    if groups:
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
                False,
            ),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(groups,),
            warmup=False,
        )
    return (q, k, v)


__all__ = ["gemma_qkv_rmsnorm"]
