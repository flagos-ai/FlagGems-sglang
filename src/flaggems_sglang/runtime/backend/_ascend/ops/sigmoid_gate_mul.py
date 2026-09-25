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


@triton.jit(do_not_specialize=["X", "G", "Y"])
def _stream(
    X,
    G,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(N, B)
    if CONTIG:
        start = pid * tl.cdiv(tiles, P)
        end = tl.minimum(start + tl.cdiv(tiles, P), tiles)
        step = 1
    else:
        start = pid
        end = tiles
        step = P
    for tile in range(start, end, step):
        i = tile * B + tl.arange(0, B)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        g = tl.load(G + i, i < N, other=0).to(tl.float32)
        y = x * (1.0 / (1.0 + tl.exp(-g)))
        tl.store(Y + i, y, i < N)


def sigmoid_gate_mul(x, gate):
    source, gates = x.contiguous(), gate.contiguous()
    output = torch.empty_like(source)
    n = source.numel()
    if not n:
        return output
    block, programs = (4096, 40) if n <= 524288 else (16384, 32)
    _stream[(programs, 1, 1)](
        source,
        gates,
        output,
        n,
        block,
        programs,
        False,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["sigmoid_gate_mul"]
