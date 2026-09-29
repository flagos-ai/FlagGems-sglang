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
def _pad(
    A,
    P,
    M: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    m = x // K
    k = x % K
    value = tl.load(A + m * A0 + k * A1, (m < M) & (x < 16 * K), other=0)
    tl.store(P + x, value, x < 16 * K)


@triton.jit
def _gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.arange(0, 16)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + m[:, None] * K + k[None, :]
    bp = B + k[:, None] * B0 + n[None, :] * B1
    acc = tl.zeros((16, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        ap += BK
        bp += BK * B0
    tl.store(C + m[:, None] * N + n[None, :], acc, m[:, None] < M)


def _run(mat_a, mat_b, config):
    bn, bk, stages, direct = config
    m, k = mat_a.shape
    n = mat_b.shape[1]
    bn = bn if n % bn == 0 else 16
    bk = bk if k % bk == 0 else 256
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    if direct and m == 16 and (mat_a.stride() == (k, 1)):
        padded = mat_a
    else:
        padded = torch.empty((16, k), dtype=mat_a.dtype, device=mat_a.device)
        _pad.run(
            mat_a,
            padded,
            m,
            k,
            mat_a.stride(0),
            mat_a.stride(1),
            4096,
            grid=(triton.cdiv(16 * k, 4096),),
            warmup=False,
        )
    _gemm.run(
        padded,
        mat_b,
        out,
        m,
        n,
        k,
        mat_b.stride(0),
        mat_b.stride(1),
        bn,
        bk,
        grid=(triton.cdiv(n, bn),),
        warmup=False,
        num_warps=4,
        num_stages=stages,
    )
    return out


def dsv3_fused_a_gemm(mat_a, mat_b):
    return _run(mat_a, mat_b, (64, 512, 2, 1))


__all__ = ["dsv3_fused_a_gemm"]
