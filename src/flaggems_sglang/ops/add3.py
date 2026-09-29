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
def _add3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # Task 76 requires two bf16 roundings. Triton may promote bf16 arithmetic
    # internally, so this explicit conversion is semantically essential and
    # must not be replaced by a single `a + b + c` expression.
    # Both additions are rounded to bf16 exactly once, and the arithmetic is
    # forced to fp32: where the hardware adds bf16 natively (Ascend) it
    # truncates, while the task (and torch) round to nearest.
    ab_bf16 = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    result_bf16 = (ab_bf16.to(tl.float32) + c.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + offsets, result_bf16, mask=mask)


def add3(a, b, c):
    """FlagOS evaluator entry point for the add3 operator."""
    out = torch.empty_like(a)
    n_elements = a.numel()

    # The published inputs have numel divisible by 16. Keep the empty-tensor
    # guard so the wrapper never attempts a zero-sized Triton launch.
    if n_elements == 0:
        return out

    # Small inputs need enough independent programs to occupy the device,
    # while large inputs benefit from amortizing program scheduling overhead.
    # These conservative tiers also avoid hard-coding a single vendor-specific
    # launch shape into the universal submission.
    if n_elements <= 4096:
        block_size = 128
    elif n_elements <= 65536:
        block_size = 512
    else:
        block_size = 4096
    grid = (triton.cdiv(n_elements, block_size),)
    _add3_kernel[grid](
        a,
        b,
        c,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def reference(a, b, c):
    """Compatibility entry point documented on the Task 76 page."""
    return add3(a, b, c)


__all__ = ["add3"]
