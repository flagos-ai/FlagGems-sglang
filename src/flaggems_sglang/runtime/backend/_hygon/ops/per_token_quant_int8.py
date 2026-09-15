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
    x_ptr, q_ptr, s_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    if N == BLOCK_N:
        x = tl.load(x_ptr + row * N + offs).to(tl.float32)
    else:
        x = tl.load(x_ptr + row * N + offs, mask=offs < N, other=0.0).to(
            tl.float32
        )
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-10)
    scale = amax / 127.0
    inv = 127.0 / amax
    q = tl.clamp(x * inv, -128.0, 127.0).to(tl.int8)
    if N == BLOCK_N:
        tl.store(q_ptr + row * N + offs, q)
    else:
        tl.store(q_ptr + row * N + offs, q, mask=offs < N)
    tl.store(s_ptr + row, scale)


def per_token_quant_int8(x):
    M, N = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    block = 1 << (N - 1).bit_length()
    _per_token_quant_int8[(M, 1, 1)](
        x,
        q,
        scales,
        N=N,
        BLOCK_N=block,
        num_warps=4,
        num_stages=1,
    )
    return q, scales


__all__ = ["per_token_quant_int8"]
