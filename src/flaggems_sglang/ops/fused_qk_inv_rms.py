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
def _component(
    X, Q, K, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    component = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    values = tl.load(
        X + (row * 3 + component) * C + col, col < C, 0, cache_modifier=".cg"
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, 0) / C + EPS)
    if component == 0:
        tl.store(Q + row, inv)
    else:
        tl.store(K + row, inv)


@triton.jit
def _component_full(X, Q, K, C: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    component = tl.program_id(1)
    values = tl.load(
        X + (row * 3 + component) * C + tl.arange(0, C), cache_modifier=".cg"
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, 0) / C + EPS)
    if component == 0:
        tl.store(Q + row, inv)
    else:
        tl.store(K + row, inv)


def fused_qk_inv_rms(qkv, eps):
    b, n, _, h, d = qkv.shape
    c = h * d
    q = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    k = torch.empty((b, n), device=qkv.device, dtype=torch.float32)
    if b * n:
        if c & (c - 1) == 0:
            _component_full.run(
                qkv, q, k, c, eps, num_warps=1, grid=(b * n, 2), warmup=False
            )
        else:
            _component.run(
                qkv,
                q,
                k,
                c,
                eps,
                triton.next_power_of_2(c),
                num_warps=1,
                grid=(b * n, 2),
                warmup=False,
            )
    return q, k


__all__ = ["fused_qk_inv_rms"]
