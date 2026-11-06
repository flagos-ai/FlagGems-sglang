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

"""silu_and_mul_masked for RDNA3 (gfx1100): ``silu(gate) * up`` on the first
``masked_m[e]`` rows of every expert (Triton).

The generic op autotunes two kernels over 57 and 123 configs. On gfx1100 its
row-kernel candidates hold up to 32 rows of the whole half row in registers,
spill, and take minutes each to compile: the first call on (T, H) = (256,
4096) autotunes for 31 minutes. Here a program takes one row and up to 512
columns of it, two elements a lane, and the launch is fixed, so every shape
compiles one kernel. Rows past ``masked_m`` leave at once. Columns are
masked, so the half row need not be a power of two, and gate and up are read
from their own halves at every width.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator — the whole computation is done
inside the Triton kernel.
"""

import torch
import triton
import triton.language as tl

# Measured on the W7900D: 512 columns at two a lane is within 8% of the
# fastest launch tried on every shape; 256 columns lose up to 32% where most
# rows are past masked_m, and at two a lane run at half speed on a 768-wide
# half row.
_MAX_BLOCK = 512
_COLS_PER_WARP = 64


@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    out_ptr,
    masked_m_ptr,
    T,
    stride_e_in,
    stride_t_in,
    stride_h_in,
    stride_e_out,
    stride_t_out,
    stride_h_out,
    half: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Columns are the fastest grid axis, so neighbouring programs read
    # neighbouring parts of one row.
    row = tl.program_id(1)
    e = row // T
    t = row % T
    if t >= tl.load(masked_m_ptr + e):
        return

    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < half
    in_row = input_ptr + e.to(tl.int64) * stride_e_in
    in_row += t.to(tl.int64) * stride_t_in
    gate = tl.load(in_row + cols * stride_h_in, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        in_row + (cols + half) * stride_h_in, mask=mask, other=0.0
    ).to(tl.float32)

    val = gate * tl.sigmoid(gate) * up

    out_row = out_ptr + e.to(tl.int64) * stride_e_out
    out_row += t.to(tl.int64) * stride_t_out
    tl.store(
        out_row + cols * stride_h_out,
        val.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def silu_and_mul_masked(input, masked_m):
    E, T, H = input.shape
    half = H // 2

    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)
    if E * T == 0 or half == 0:
        return out

    if not input.is_contiguous():
        input = input.contiguous()

    block = min(_MAX_BLOCK, triton.next_power_of_2(half))
    _silu_and_mul_masked_kernel[(triton.cdiv(half, block), E * T)](
        input,
        out,
        masked_m,
        T,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        half=half,
        BLOCK=block,
        num_warps=max(1, block // _COLS_PER_WARP),
        num_stages=1,
    )
    return out


__all__ = ["silu_and_mul_masked"]
