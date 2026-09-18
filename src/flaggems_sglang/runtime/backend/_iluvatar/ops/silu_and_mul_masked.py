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

import torch
import triton
import triton.language as tl

_LARGE_HALF_TILE = 2048


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=["HALF", "E", "T"],
)
@triton.jit
def _kernel_full(
    in_ptr,  # *bf16  [E, T, H]
    mm_ptr,  # *int32 [E]
    out_ptr,  # *bf16  [E, T, half]
    E: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    pid = tl.program_id(0)

    e = pid // T
    row = pid % T

    n = tl.load(mm_ptr + e)
    if row >= n:
        return

    in_row = e * T * H + row * H
    out_row = e * T * HALF + row * HALF

    cols = tl.arange(0, BLOCK_H)
    gate = tl.load(in_ptr + in_row + cols).to(tl.float32)
    up = tl.load(in_ptr + in_row + HALF + cols).to(tl.float32)

    val = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + out_row + cols, val.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=["HALF", "E", "T"],
)
@triton.jit
def _kernel_mask(
    in_ptr,  # *bf16  [E, T, H]
    mm_ptr,  # *int32 [E]
    out_ptr,  # *bf16  [E, T, half]
    E: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    pid = tl.program_id(0)
    hblk = tl.program_id(1)

    e = pid // T
    row = pid % T

    n = tl.load(mm_ptr + e)
    if row >= n:
        return

    in_row = e * T * H + row * H
    out_row = e * T * HALF + row * HALF

    cols = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < HALF

    gate = tl.load(in_ptr + in_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_ptr + in_row + HALF + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    val = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + out_row + cols, val.to(out_ptr.dtype.element_ty), mask=mask
    )


def silu_and_mul_masked(input, masked_m):
    E, T, H = input.shape
    half = H // 2

    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)

    if E == 0 or T == 0 or half == 0:
        return out

    if half <= 2048:

        if half & (half - 1) == 0:
            block_h = half
            grid = (E * T,)
            _kernel_full[grid](
                input,
                masked_m,
                out,
                E=E,
                T=T,
                H=H,
                HALF=half,
                BLOCK_H=block_h,
            )
        else:
            block_h = triton.next_power_of_2(half)
            grid = (E * T,)
            _kernel_mask[grid](
                input,
                masked_m,
                out,
                E=E,
                T=T,
                H=H,
                HALF=half,
                BLOCK_H=block_h,
            )
    else:

        block_h = _LARGE_HALF_TILE
        grid = (E * T, triton.cdiv(half, block_h))
        _kernel_mask[grid](
            input,
            masked_m,
            out,
            E=E,
            T=T,
            H=H,
            HALF=half,
            BLOCK_H=block_h,
        )

    return out


__all__ = ["silu_and_mul_masked"]
