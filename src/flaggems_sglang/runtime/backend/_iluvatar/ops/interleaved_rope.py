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
    numel: tl.constexpr,
    stride: tl.constexpr,
    dim: tl.constexpr,
    section_1: tl.constexpr,
    section_2: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < numel
    cols = offsets % dim
    use_1 = (cols % 3 == 1) & (cols < section_1 * 3)
    use_2 = (cols % 3 == 2) & (cols < section_2 * 3)
    source = tl.where(use_1, 1, tl.where(use_2, 2, 0))
    values = tl.load(x_ptr + source * stride + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def interleaved_rope(
    x: torch.Tensor, mrope_section: Sequence[int]
) -> torch.Tensor:
    _, seq_len, dim = x.shape
    output = torch.empty((seq_len, dim), dtype=x.dtype, device=x.device)
    numel = seq_len * dim
    block = 1024
    _interleaved_rope_kernel[(triton.cdiv(numel, block),)](
        x,
        output,
        numel=numel,
        stride=numel,
        dim=dim,
        section_1=mrope_section[1],
        section_2=mrope_section[2],
        block=block,
        num_warps=4,
    )
    return output


__all__ = ["interleaved_rope"]
