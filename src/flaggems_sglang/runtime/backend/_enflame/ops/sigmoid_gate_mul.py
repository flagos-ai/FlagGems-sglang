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
def _gate(X, G, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    g = tl.load(G + i, i < N, other=0).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(Y + i, y, i < N)


@triton.jit
def _wide(
    X,
    G,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(N, B), P):
        if R:
            i = (
                tile * B
                + tl.arange(0, R)[:, None] * C
                + tl.arange(0, C)[None, :]
            )
        else:
            i = tile * B + tl.arange(0, B)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        g = tl.load(G + i, i < N, other=0).to(tl.float32)
        y = x * (1.0 / (1.0 + tl.exp(-g)))
        tl.store(Y + i, y, i < N)


def sigmoid_gate_mul(x, gate):
    n = x.numel()
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if not n:
        return output
    source, gates = (x.contiguous(), gate.contiguous())
    if n < 32768:
        block = min(16384, triton.next_power_of_2(n))
        _gate.run(
            source,
            gates,
            output,
            n,
            block,
            grid=(triton.cdiv(n, block),),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
    else:
        if n <= 262144:
            block, warps = (65536, 4) if n <= 65536 else (131072, 8)
        elif n <= 524288:
            block, warps = (131072, 4)
        else:
            block, warps = (65536, 2)
        programs = min(12, triton.cdiv(n, block))
        _wide.run(
            source,
            gates,
            output,
            n,
            block,
            programs,
            0,
            0,
            grid=(programs,),
            warmup=False,
            num_warps=warps,
            num_stages=1,
        )
    return output


__all__ = ["sigmoid_gate_mul"]
