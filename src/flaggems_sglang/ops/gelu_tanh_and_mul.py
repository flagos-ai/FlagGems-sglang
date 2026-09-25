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

"""Triton implementation of tanh-approximated gated GELU:

out = gelu_tanh(x[..., :d]) * x[..., d:]

Fully memory-bandwidth bound: flat 1-D addressing (in-register row/col
recovery), size-tiered launch config and cache hints, and the GELU term
collapsed to a single fp32 exponential (v * sigmoid(inner)).
"""

import torch
import triton
import triton.language as tl

# 2 * sqrt(2/pi): the single scale of the sigmoid fold above.
_SCALE = tl.constexpr(1.5957691216057308)
_COEFF = tl.constexpr(0.044715)


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    out_ptr,
    n_elem,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < n_elem

    # Flat 1-D addressing: recover the (row, col) pair in-register so both
    # halves of the interleaved row are read as one contiguous span each.
    row = offs // D
    col = offs % D
    base = row * (2 * D) + col

    x1 = tl.load(
        x_ptr + base, mask=mask, other=0.0, cache_modifier=LOAD_CACHE
    ).to(tl.float32)
    x3 = tl.load(
        x_ptr + base + D, mask=mask, other=0.0, cache_modifier=LOAD_CACHE
    ).to(tl.float32)

    # gelu_tanh(v) = v * sigmoid(2 * sqrt(2/pi) * (v + 0.044715 v^3)), fp32.
    cubic = x1 * x1 * x1
    inner = _SCALE * (x1 + _COEFF * cubic)
    y = x1 * tl.fdiv(1.0, 1.0 + tl.exp(-inner)) * x3

    tl.store(
        out_ptr + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
        cache_modifier=STORE_CACHE,
    )


def _pick_config(n_elem):
    """Empirical best (BLOCK_D, num_warps, load cache, store cache) per size class.

    The largest streaming shapes saturate DRAM with a 2048-wide block at 4
    warps and benefit from a stream-once store hint (the write is never
    re-read). Mid-size working sets keep the default store policy so the
    output stays cacheable while the input halves still use a streaming read
    hint. Small problems ride the platform's launch/allocation floor and any
    config ties there, so they get a narrow single wave with default policy.
    """
    if n_elem >= 1048576:
        return 2048, 4, ".cg", ".cs"
    if n_elem >= 32768:
        return 1024, 4, ".cg", ""
    if n_elem <= 8192:
        return 256, 1, "", ""
    return 1024, 4, "", ""


def gelu_tanh_and_mul(input):
    """out = gelu_tanh(input[..., :d]) * input[..., d:], out dtype = input dtype.

    ``input``: [..., 2 * d]; the last dimension is split into the gate half
    ``[..., :d]`` and the up half ``[..., d:]``.
    """
    if not input.is_contiguous():
        input = input.contiguous()
    d = input.shape[-1] // 2
    n_elem = input.numel() // 2

    out = torch.empty(
        input.shape[:-1] + (d,), dtype=input.dtype, device=input.device
    )

    block_d, num_warps, load_cache, store_cache = _pick_config(n_elem)
    _gelu_tanh_and_mul_kernel[(triton.cdiv(n_elem, block_d),)](
        input,
        out,
        n_elem,
        D=d,
        BLOCK_D=block_d,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["gelu_tanh_and_mul"]
