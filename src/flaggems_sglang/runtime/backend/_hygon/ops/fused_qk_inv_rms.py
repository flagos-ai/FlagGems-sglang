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
def _add_pair(a, b, c, d):
    return (a + c, b + d)


@triton.jit
def _component(
    X, Q, K, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    component = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    v = tl.load(X + (row * 3 + component) * C + col, col < C, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, 0) / C + EPS)
    if component == 0:
        tl.store(Q + row, inv)
    else:
        tl.store(K + row, inv)


@triton.jit
def _component_full(X, Q, K, C: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    component = tl.program_id(1)
    v = tl.load(X + (row * 3 + component) * C + tl.arange(0, C)).to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, 0) / C + EPS)
    if component == 0:
        tl.store(Q + row, inv)
    else:
        tl.store(K + row, inv)


@triton.jit
def _row_full(X, Q, K, C: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, C)
    q = tl.load(X + row * 3 * C + col).to(tl.float32)
    k = tl.load(X + row * 3 * C + C + col).to(tl.float32)
    qs, ks = tl.reduce((q * q, k * k), 0, _add_pair)
    tl.store(Q + row, tl.rsqrt(qs / C + EPS))
    tl.store(K + row, tl.rsqrt(ks / C + EPS))


@triton.jit(do_not_specialize=["X", "Q", "K"])
def _aligned_full(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.float32),
    K: tl.pointer_type(tl.float32),
    C: tl.constexpr,
    EPS: tl.constexpr,
    ALIGN: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, C)
    qbase = tl.multiple_of(X + row * 3 * C, ALIGN)
    kbase = tl.multiple_of(X + row * 3 * C + C, ALIGN)
    q = tl.load(qbase + col).to(tl.float32)
    k = tl.load(kbase + col).to(tl.float32)
    qs, ks = tl.reduce((q * q, k * k), 0, _add_pair)
    tl.store(Q + row, tl.rsqrt(qs / C + EPS))
    tl.store(K + row, tl.rsqrt(ks / C + EPS))


def fused_qk_inv_rms(qkv, eps):
    b, n, _, h, d = qkv.shape
    m = b * n
    c = h * d
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    if m:
        if c & (c - 1):
            _component.run(
                qkv,
                q,
                k,
                c,
                eps,
                triton.next_power_of_2(c),
                grid=(m, 2),
                warmup=False,
                num_warps=1,
            )
        elif m >= 4096 and qkv.dtype == torch.bfloat16:
            align = 4 if qkv.data_ptr() % 4 == 0 and c % 2 == 0 else 1
            _aligned_full.run(
                qkv, q, k, c, eps, align, grid=(m,), warmup=False, num_warps=4
            )
        else:
            _component_full.run(
                qkv, q, k, c, eps, grid=(m, 2), warmup=False, num_warps=1
            )
    return q, k


__all__ = ["fused_qk_inv_rms"]
