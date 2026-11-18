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


@triton.jit
def _interleaved_rope_flat_const_kernel(
    x_ptr,
    out_ptr,
    N,
    D: tl.constexpr,
    S13: tl.constexpr,
    S23: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    np = tl.num_programs(0)
    j = tl.arange(0, BLOCK)
    col = j % D
    use_1 = (col % 3 == 1) & (col < S13)
    use_2 = (col % 3 == 2) & (col < S23)
    for start in tl.range(pid * BLOCK, N, np * BLOCK, num_stages=3):
        offs = start + j
        x0 = tl.load(x_ptr + offs)
        x1 = tl.load(x_ptr + N + offs)
        x2 = tl.load(x_ptr + 2 * N + offs)
        out = tl.where(use_1, x1, tl.where(use_2, x2, x0))
        tl.store(out_ptr + offs, out)


@triton.jit
def _interleaved_rope_flat_runtime_kernel(
    x_ptr,
    out_ptr,
    N,
    D: tl.constexpr,
    S13: tl.constexpr,
    S23: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    np = tl.num_programs(0)
    j = tl.arange(0, BLOCK)
    for start in tl.range(pid * BLOCK, N, np * BLOCK):
        offs = start + j
        col = offs % D
        x0 = tl.load(x_ptr + offs)
        x1 = tl.load(x_ptr + N + offs)
        x2 = tl.load(x_ptr + 2 * N + offs)
        use_1 = (col % 3 == 1) & (col < S13)
        use_2 = (col % 3 == 2) & (col < S23)
        out = tl.where(use_1, x1, tl.where(use_2, x2, x0))
        tl.store(out_ptr + offs, out)


_MAX_GRID = 48
_FLAT_BLOCK_CAP = 4096


def _floor_log2(n):
    return n.bit_length() - 1


def _pick_const_block(s, d):
    k = min(
        (s & -s).bit_length() - 1,
        _floor_log2(max(1, s // 8)),
        _floor_log2(max(1, _FLAT_BLOCK_CAP // d)),
    )
    return d << k


def _pick_runtime_block(n):
    block = 1
    cap = min(_FLAT_BLOCK_CAP, max(16, n // 8))
    while n % (block * 2) == 0 and block * 2 <= cap:
        block *= 2
    return block


def interleaved_rope(
    x: torch.Tensor, mrope_section: Sequence[int]
) -> torch.Tensor:
    _, s, d = x.shape
    output = torch.empty((s, d), dtype=x.dtype, device=x.device)
    n = s * d
    s13 = 3 * mrope_section[1]
    s23 = 3 * mrope_section[2]
    if n < (1 << 31) // 3 and d <= 8192:
        if d & (d - 1) == 0:
            block = _pick_const_block(s, d)
            grid = min(_MAX_GRID, n // block)
            num_warps = 1 if block <= 128 else 4
            _interleaved_rope_flat_const_kernel[(grid,)](
                x,
                output,
                n,
                D=d,
                S13=s13,
                S23=s23,
                BLOCK=block,
                num_warps=num_warps,
            )
            return output
        block = _pick_runtime_block(n)
        if block >= 16:
            grid = min(_MAX_GRID, n // block)
            num_warps = 1 if block <= 128 else 4
            _interleaved_rope_flat_runtime_kernel[(grid,)](
                x,
                output,
                n,
                D=d,
                S13=s13,
                S23=s23,
                BLOCK=block,
                num_warps=num_warps,
            )
            return output
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
