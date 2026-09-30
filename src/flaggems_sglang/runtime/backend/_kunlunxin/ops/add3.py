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
def _add3_kernel_ab(
    a_ptr,
    b_ptr,
    ab_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    ab = a + b
    bits = ab.to(tl.int32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded_bits = (bits + 32767 + lsb) & -65536
    rounded_fp32 = rounded_bits.to(tl.float32, bitcast=True)
    tl.store(ab_ptr + offsets, rounded_fp32.to(tl.bfloat16), mask=mask)


@triton.jit
def _add3_kernel_abc(
    ab_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offsets < n_elements
    ab = tl.load(ab_ptr + offsets, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + offsets, mask=mask).to(tl.float32)
    out = ab + c
    bits = out.to(tl.int32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded_bits = (bits + 32767 + lsb) & -65536
    rounded_fp32 = rounded_bits.to(tl.float32, bitcast=True)
    tl.store(out_ptr + offsets, rounded_fp32.to(tl.bfloat16), mask=mask)


def add3(a, b, c):
    out = torch.empty_like(a)
    n_elements = a.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = min(8192, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    ab = torch.empty(n_elements, dtype=torch.bfloat16, device=a.device)

    _add3_kernel_ab[grid](
        a,
        b,
        ab,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )

    _add3_kernel_abc[grid](
        ab,
        c,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )

    return out


__all__ = ["add3"]
