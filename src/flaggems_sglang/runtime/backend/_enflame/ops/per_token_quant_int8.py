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
def _per_token_quant_int8(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    mask = rmask[:, None] & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    scale = amax / 127.0
    inv = 127.0 / amax
    q = tl.clamp(x * inv[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + rows, scale, mask=rmask)


def per_token_quant_int8(x):
    M, N = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    bn = triton.next_power_of_2(N)
    bm = max(1, min(triton.next_power_of_2(M), 65536 // bn))
    grid = (triton.cdiv(M, bm),)
    _per_token_quant_int8[grid](
        x,
        q,
        scales,
        M,
        N=N,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=4,
        num_stages=1,
    )
    return q, scales


__all__ = ["per_token_quant_int8"]
