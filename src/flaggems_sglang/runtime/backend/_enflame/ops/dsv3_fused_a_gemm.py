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
def _gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + m[:, None] * A0 + k[None, :] * A1
    bp = B + k[:, None] * B0 + n[None, :] * B1
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(
            ap, (m[:, None] < M) & (k[None, :] < K - start * BK), other=0
        )
        b = tl.load(
            bp, (k[:, None] < K - start * BK) & (n[None, :] < N), other=0
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
        ap += BK * A1
        bp += BK * B0
    tl.store(
        C + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


def dsv3_fused_a_gemm(mat_a, mat_b):
    m, k = mat_a.shape
    n = mat_b.shape[1]
    bn, bk = (64, 2048) if k <= 4096 else (128, 1024)
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    _gemm.run(
        mat_a,
        mat_b,
        out,
        m,
        n,
        k,
        mat_a.stride(0),
        mat_a.stride(1),
        mat_b.stride(0),
        mat_b.stride(1),
        triton.next_power_of_2(m),
        bn,
        bk,
        grid=(triton.cdiv(n, bn),),
        warmup=False,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["dsv3_fused_a_gemm"]
