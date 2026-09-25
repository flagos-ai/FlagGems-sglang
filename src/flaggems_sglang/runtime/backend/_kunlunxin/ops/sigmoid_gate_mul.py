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
from triton.language.extra.xpu import libdevice


@triton.jit
def _native(X, G, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    g = tl.load(G + i, i < N, other=0).to(tl.float32)
    y = x * (0.5 * (1.0 + libdevice.tanh(0.5 * g)))
    tl.store(Y + i, y, i < N)


def sigmoid_gate_mul(x, gate):
    source, gates = (x.contiguous(), gate.contiguous())
    output = torch.empty_like(source)
    n = source.numel()
    if n:
        block = min(16384, triton.next_power_of_2(n))
        _native.run(
            source,
            gates,
            output,
            n,
            block,
            grid=(triton.cdiv(n, block),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["sigmoid_gate_mul"]
