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

"""Square ReLU, direct masked flat memory accesses."""
import torch
import triton
import triton.language as tl


@triton.jit
def _flat(
    X,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
):
    for tile in range(triton.cdiv(N, B * P)):
        i = (tile * P + tl.program_id(0)) * B + tl.arange(0, B)
        if S1 == 1 and S0 == D:
            offset = i
        else:
            offset = (i // D) * S0 + (i % D) * S1
        x = tl.load(X + offset, i < N, other=0).to(tl.float32)
        z = tl.where(x < 0.0, 0.0, x)
        tl.store(Y + i, z * z, i < N)


def relu2(input):
    rows, dim = input.shape
    out = torch.empty((rows, dim), dtype=input.dtype, device=input.device)
    n = rows * dim
    if n:
        block = 512
        programs = min(triton.cdiv(n, block), 65535)
        _flat[(programs,)](
            input,
            out,
            n,
            dim,
            *input.stride(),
            block,
            programs,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["relu2"]
