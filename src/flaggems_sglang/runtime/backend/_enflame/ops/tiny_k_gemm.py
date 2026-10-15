# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
def _tiny_k_reduce(
    x,
    w,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sxm,
    sxk,
    swn,
    swk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    m = pid // n_blocks
    n = (pid % n_blocks) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    xv = tl.load(x + m * sxm + k * sxk, mask=k < K, other=0.0).to(tl.float32)
    wv = tl.load(
        w + n[:, None] * swn + k[None, :] * swk,
        mask=(n[:, None] < N) & (k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(wv * xv[None, :], axis=1)
    tl.store(out + m * N + n, acc, mask=n < N)


@triton.jit
def _tiny_k_dot(
    x,
    w,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sxm,
    sxk,
    swn,
    swk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + kk
        a = tl.load(
            x + rows[:, None] * sxm + ks[None, :] * sxk,
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            w + cols[None, :] * swn + ks[:, None] * swk,
            mask=(cols[None, :] < N) & (ks[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def tiny_k_gemm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    if m == 0 or n == 0:
        return out
    _tiny_k_dot[(triton.cdiv(n, 64),)](
        x,
        w,
        out,
        M=m,
        N=n,
        K=k,
        sxm=x.stride(0),
        sxk=x.stride(1),
        swn=w.stride(0),
        swk=w.stride(1),
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=2,
        num_stages=3,
    )
    return out


__all__ = ["tiny_k_gemm"]
