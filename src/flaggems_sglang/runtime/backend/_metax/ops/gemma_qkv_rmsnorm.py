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
import triton.language.core as core


@core.extern
def _hardware_rsq(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.float32,): ("llvm.mxc.rsq.f32", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _normal_rsqrt(x, E: tl.constexpr):
    if E >= 7.888609052210118e-31:
        return _hardware_rsq(x)
    else:
        return tl.rsqrt(x)


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
    offset = row[:, None] // H * S + row[:, None] % H * D + col[None, :]
    mask = (row[:, None] < M * H) & (col[None, :] < D)
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    if B >= 64 and (B & (B - 1)) == 0:
        squared_sum_tile = tl.sum(
            tl.sum(tl.reshape(x * x, (R, B // 64, 64)), 2), 1
        )
    else:
        squared_sum_tile = tl.sum(x * x, 1)
    y = x * _normal_rsqrt(squared_sum_tile[:, None] / D + E, E)
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
def _kernel(TENSORS, META: tl.constexpr):
    Q, K, V, WQ, WK = TENSORS
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
    rows = (8 if head_dim >= 256 else 32) if m >= 256 else 4
    warps = 2 if m >= 256 and (head_dim == 128 or head_dim >= 256) else 4
    if m < 256 and head_dim <= 256 and m * num_q_heads >= 512:
        rows = 4
        warps = 1
    if m < 256 and head_dim <= 256 and m * num_q_heads >= 2048:
        rows = 8
    groups = ((m * num_q_heads + rows - 1) // rows) + (
        2 * ((m * num_kv_heads + rows - 1) // rows) if kv else 0
    )
    if groups:
        _kernel.run(
            (
                q,
                k if kv else None,
                v if kv else None,
                q_weight,
                k_weight if kv else None,
            ),
            (
                m,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q.stride()[0],
                k.stride()[0] if kv else 0,
                v.stride()[0] if kv else 0,
                q_weight.stride()[0],
                k_weight.stride()[0] if kv else 0,
                eps,
                kv,
                (1 << (head_dim - 1).bit_length()),
                rows,
                False,
            ),
            num_warps=warps,
            grid=(groups,),
            warmup=False,
        )
    return (q, k, v)


__all__ = ["gemma_qkv_rmsnorm"]
