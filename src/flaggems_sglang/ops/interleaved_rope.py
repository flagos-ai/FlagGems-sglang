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

from typing import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _interleaved_rope_kernel(
    x_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    section_1: tl.constexpr,
    section_2: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    mask = (rows < S) & (cols < D)
    offsets = rows * D + cols
    x0 = tl.load(x_ptr + offsets, mask=mask)
    x1 = tl.load(x_ptr + S * D + offsets, mask=mask)
    x2 = tl.load(x_ptr + 2 * S * D + offsets, mask=mask)
    use_1 = (cols % 3 == 1) & (cols < section_1 * 3)
    use_2 = (cols % 3 == 2) & (cols < section_2 * 3)
    output = tl.where(use_1, x1, tl.where(use_2, x2, x0))
    tl.store(out_ptr + offsets, output, mask=mask)


def interleaved_rope(
    x: torch.Tensor, mrope_section: Sequence[int]
) -> torch.Tensor:
    _, s, d = x.shape
    output = torch.empty((s, d), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(s, 64), triton.cdiv(d, 128))
    _interleaved_rope_kernel[grid](
        x,
        output,
        S=s,
        D=d,
        section_1=mrope_section[1],
        section_2=mrope_section[2],
        BLOCK_S=64,
        BLOCK_D=128,
        num_warps=4,
    )
    return output


__all__ = ["interleaved_rope"]
