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
    P,
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
    SPLITS: tl.constexpr,
):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    k = tl.arange(0, BK) + split * BK
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(triton.cdiv(K, BK * SPLITS)):
        kk = k + step * BK * SPLITS
        a = tl.load(
            A + m[:, None] * A0 + kk[None, :] * A1,
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + kk[:, None] * B0 + n[None, :] * B1,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        P + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _finish(
    P,
    OUT,
    SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    POWER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, POWER)
    values = tl.load(
        P + part[:, None] * SIZE + x[None, :],
        (part[:, None] < SPLITS) & (x[None, :] < SIZE),
        0,
    )
    tl.store(OUT + x, tl.sum(values, 0), x < SIZE)


@triton.jit
def _full_gemv(
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
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + row * A0 + kk * A1, kk < K, 0).to(tl.float32)
        b = tl.load(
            B + n[:, None] * B1 + kk[None, :] * B0,
            (n[:, None] < N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc += a[None, :] * b
    tl.store(C + row * N + n, tl.sum(acc, 1), n < N)


def dsv3_fused_a_gemm(mat_a, mat_b):
    m, k = mat_a.shape
    n = mat_b.shape[1]
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    strides = (
        mat_a.stride(0),
        mat_a.stride(1),
        mat_b.stride(0),
        mat_b.stride(1),
    )
    if m == 1:
        _full_gemv.run(
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            *strides,
            1,
            1024,
            grid=(triton.cdiv(n, 1), m),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    else:
        splits = min(8, triton.cdiv(k, 64))
        partial = (
            torch.empty(
                (splits, m, n), dtype=torch.float32, device=mat_a.device
            )
            if splits > 1
            else out
        )
        _gemm.run(
            mat_a,
            mat_b,
            partial,
            m,
            n,
            k,
            *strides,
            16,
            64,
            64,
            splits,
            grid=(triton.cdiv(n, 64), splits),
            warmup=False,
            num_warps=4,
            num_stages=3,
        )
        if splits > 1:
            _finish.run(
                partial,
                out,
                m * n,
                splits,
                triton.next_power_of_2(splits),
                256,
                grid=(triton.cdiv(m * n, 256),),
                warmup=False,
                num_warps=1,
                num_stages=1,
            )
    return out


__all__ = ["dsv3_fused_a_gemm"]
