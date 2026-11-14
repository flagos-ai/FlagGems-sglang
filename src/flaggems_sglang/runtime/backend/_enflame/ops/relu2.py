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
    B: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(triton.cdiv(N, B * P)):
        i = (tile * P + pid) * B + tl.arange(0, B)
        m = i < N
        x = tl.load(X + i, m, other=0).to(tl.float32)
        z = tl.where(x < 0.0, 0.0, x)
        tl.store(Y + i, z * z, m)


@triton.jit
def _rows(
    X,
    Y,
    R,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    for r in range(pid, R, P):
        xbase = r * S0
        ybase = r * D
        for c0 in range(0, D, B):
            cols = c0 + tl.arange(0, B)
            m = cols < D
            x = tl.load(X + xbase + cols * S1, m, other=0).to(tl.float32)
            z = tl.where(x < 0.0, 0.0, x)
            tl.store(Y + ybase + cols, z * z, m)


def _next_pow2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _block_cap(itemsize):
    return min(32768 // itemsize, 16384)


def relu2(input):
    rows, dim = input.shape
    out = torch.empty((rows, dim), dtype=input.dtype, device=input.device)
    n = rows * dim
    if not n:
        return out
    itemsize = input.element_size()
    if input.is_contiguous():
        root = int(n**0.5)
        factor = 32 if itemsize <= 2 else 16
        block = min(_next_pow2(max(factor * root, 256)), _block_cap(itemsize))
        programs = min(triton.cdiv(n, block), 48)
        _flat[(programs,)](
            input,
            out,
            n,
            block,
            programs,
            num_warps=4 if block >= 4096 else 2,
            num_stages=1,
        )
    else:
        block = max(min(_next_pow2(dim), _block_cap(itemsize)), 2)
        programs = min(rows, 48)
        _rows[(programs,)](
            input,
            out,
            rows,
            dim,
            input.stride(0),
            input.stride(1),
            block,
            programs,
            num_warps=4 if block >= 4096 else 2,
            num_stages=1,
        )
    return out


__all__ = ["relu2"]
