# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

# gfx1100: 512 output elements per 16-warp program was the fastest or within
# noise on every shape tried (d from 128 to 8192, 1 to 32768 rows).
_BLOCK = 512
_WARPS = 16


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    total,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Blocks of the flattened output, so short rows share a program and long
    # rows need no padding to a power of two.
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // d
    src = x_ptr + row * (2 * d) + (offs - row * d)
    x1 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(src + d, mask=mask, other=0.0).to(tl.float32)

    silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))

    tl.store(out_ptr + offs, silu_x1 * x2, mask=mask)


def silu_and_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    d = shape[-1] // 2
    numel = hidden_states.numel()
    out = hidden_states.new_empty(shape[:-1] + (d,))
    if d == 0 or numel == 0:
        return out
    x = hidden_states.contiguous()
    total = numel // 2
    _silu_and_mul_kernel[(triton.cdiv(total, _BLOCK),)](
        x,
        out,
        total,
        d=d,
        BLOCK=_BLOCK,
        WIDE=numel >= 2**31,
        num_warps=_WARPS,
    )
    return out


__all__ = ["silu_and_mul"]
