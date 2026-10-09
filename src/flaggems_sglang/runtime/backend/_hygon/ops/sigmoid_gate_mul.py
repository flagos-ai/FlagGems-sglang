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
def _flat(
    X,
    G,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    LC: tl.constexpr,
    SC: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0, cache_modifier=LC).to(tl.float32)
    g = tl.load(G + i, i < N, other=0, cache_modifier=LC).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(Y + i, y, i < N, cache_modifier=SC)


def sigmoid_gate_mul(x, gate):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    if not n:
        return output
    source, gates = (x.contiguous(), gate.contiguous())
    block = 4096 if n >= 2097152 else 1024
    warps = 8 if n >= 16777216 else 4
    lc, sc = ("", "")
    _flat.run(
        source,
        gates,
        output,
        n,
        block,
        lc,
        sc,
        grid=(triton.cdiv(n, block),),
        warmup=False,
        num_warps=warps,
        num_stages=1,
    )
    return output


__all__ = ["sigmoid_gate_mul"]
