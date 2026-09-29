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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""clamp_position -- Iluvatar specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit
def _clamp_position_exact_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, eviction_policy="evict_first")


@triton.jit
def _clamp_position_vec4_exact_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    # VEC consecutive elements per lane -> 128-bit vectorized ld/st. The 2D
    # offset grid is flattened back so the element order is identical to the
    # 1D kernel.
    pid = tl.program_id(0)
    base = pid * BLOCK * VEC
    offs = tl.reshape(
        base + tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :],
        [BLOCK * VEC],
    )
    x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, eviction_policy="evict_first")


@triton.jit
def _clamp_position_vec4_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * VEC
    offs = tl.reshape(
        base + tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :],
        [BLOCK * VEC],
    )
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit
def _clamp_position_strided_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    x_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(
        x_ptr + offs * x_stride, mask=mask, eviction_policy="evict_first"
    )
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, mask=mask, eviction_policy="evict_first")


def clamp_position(seq_lens):
    n = seq_lens.numel()
    out = torch.empty_like(seq_lens)

    if n == 0:
        return out

    if seq_lens.stride(0) == 1:
        # Size-tiered launch config (see module docstring). Small inputs are
        # fully covered by one small CTA; large divisible sizes use the
        # vectorized 128-bit kernel, and other large sizes use the plain 1D
        # kernel with the bounds mask dropped when the block tiles n exactly.
        if n <= 64:
            BLOCK, num_warps = 64, 1
        elif n <= 2048:
            BLOCK, num_warps = 512, 8
        else:
            BLOCK, num_warps = 2048, 32
        if n > 2048 and n % 4096 == 0:
            grid = (n // 4096,)
            _clamp_position_vec4_exact_kernel[grid](
                seq_lens, out, BLOCK=1024, VEC=4, num_warps=32
            )
        elif n % BLOCK == 0:
            grid = (n // BLOCK,)
            _clamp_position_exact_kernel[grid](
                seq_lens, out, BLOCK=BLOCK, num_warps=num_warps
            )
        else:
            grid = (triton.cdiv(n, BLOCK),)
            _clamp_position_kernel[grid](
                seq_lens, out, n, BLOCK=BLOCK, num_warps=num_warps
            )
    else:
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _clamp_position_strided_kernel[grid](
            seq_lens, out, n, seq_lens.stride(0), BLOCK=BLOCK, num_warps=4
        )
    return out


__all__ = ["clamp_position"]
