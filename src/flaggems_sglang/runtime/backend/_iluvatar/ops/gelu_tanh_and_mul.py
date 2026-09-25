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

"""tanh-approximated gated GELU -- Iluvatar specialization.
"""

import torch
import triton
import triton.language as tl

_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_GELU_TANH_COEFF = tl.constexpr(0.044715)

_AUTOTUNE_CONFIGS = [
    # Small row counts cannot fill the device with wide blocks, so tiny
    # BLOCK_D tiles (more programs) win there; large shapes prefer wide
    # blocks with 8 elements per thread for 16B vectorised access.
    triton.Config({"BLOCK_D": 64}, num_warps=1),
    triton.Config({"BLOCK_D": 128}, num_warps=2),
    triton.Config({"BLOCK_D": 256}, num_warps=2),
    triton.Config({"BLOCK_D": 256}, num_warps=4),
    triton.Config({"BLOCK_D": 512}, num_warps=4),
    triton.Config({"BLOCK_D": 512}, num_warps=8),
    triton.Config({"BLOCK_D": 1024}, num_warps=4),
    triton.Config({"BLOCK_D": 1024}, num_warps=8),
    triton.Config({"BLOCK_D": 2048}, num_warps=8),
    triton.Config({"BLOCK_D": 2048}, num_warps=16),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["D"])
@triton.jit
def _gelu_tanh_and_mul_kernel_2d(
    x_ptr,
    out_ptr,
    D: tl.constexpr,
    stride_row,
    BLOCK_D: tl.constexpr,
):
    """2D-grid variant: grid is ``(row, column-block)``.

    Used for small workloads where the kernel runtime is a few microseconds
    and the per-program div/mod of the flattened grid is a measurable
    fraction of the launch.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    cols = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
    row_base = pid_row * stride_row

    if D % BLOCK_D == 0:
        x1 = tl.load(x_ptr + row_base + cols).to(tl.float32)
        x3 = tl.load(x_ptr + row_base + D + cols).to(tl.float32)
    else:
        mask = cols < D
        x1 = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + row_base + D + cols, mask=mask, other=0.0).to(
            tl.float32
        )

    # gelu_tanh(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
    # 0.5 * (1 + tanh(z)) == sigmoid(2z), so y = x1 * sigmoid(2*inner) * x3.
    inner = _SQRT_2_OVER_PI * x1 * (1.0 + _GELU_TANH_COEFF * x1 * x1)
    y = x1 * tl.sigmoid(2.0 * inner) * x3

    out_base = pid_row * D
    if D % BLOCK_D == 0:
        tl.store(out_ptr + out_base + cols, y.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptr + out_base + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=cols < D,
        )


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["D", "ROWS"])
@triton.jit
def _gelu_tanh_and_mul_kernel_1d(
    x_ptr,
    out_ptr,
    D: tl.constexpr,
    stride_row,
    ROWS,
    BLOCK_D: tl.constexpr,
):
    """Flattened 1D grid: one program computes a BLOCK_D-wide slice of one row.

    Memory layout of a row of ``input`` (length 2*D): ``[x1 (D) | x3 (D)]``.
    The grid is flattened over (row, column-block) pairs so that small row
    counts still launch enough programs to occupy every SM.
    """
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(D, BLOCK_D)
    pid_row = pid // num_blocks
    pid_col = pid % num_blocks

    cols = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)

    row_base = pid_row.to(tl.int64) * stride_row
    if D % BLOCK_D == 0:
        x1 = tl.load(x_ptr + row_base + cols).to(tl.float32)
        x3 = tl.load(x_ptr + row_base + D + cols).to(tl.float32)
    else:
        mask = cols < D
        x1 = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + row_base + D + cols, mask=mask, other=0.0).to(
            tl.float32
        )

    # gelu_tanh(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
    # 0.5 * (1 + tanh(z)) == sigmoid(2z), so y = x1 * sigmoid(2*inner) * x3.
    inner = _SQRT_2_OVER_PI * x1 * (1.0 + _GELU_TANH_COEFF * x1 * x1)
    y = x1 * tl.sigmoid(2.0 * inner) * x3

    out_base = pid_row.to(tl.int64) * D
    if D % BLOCK_D == 0:
        tl.store(out_ptr + out_base + cols, y.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptr + out_base + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=cols < D,
        )


# Workloads dispatched to the 2D-grid kernel: narrow rows (D up to 4096) with
# at most 512 (row, column-block) programs at BLOCK_D=512. Beyond that the
# flattened 1D grid benchmarked as fast or faster (it wins for very wide rows
# such as D=8192 even at tiny row counts, and once the device is saturated).
_MAX_2D_PROGRAMS = 512
_MAX_2D_D = 4096


def gelu_tanh_and_mul(input):
    d = input.shape[-1] // 2
    x = input if input.is_contiguous() else input.contiguous()
    x2 = x.view(-1, 2 * d)
    rows = x2.shape[0]

    out = torch.empty((rows, d), dtype=input.dtype, device=x.device)
    if d <= _MAX_2D_D and rows * triton.cdiv(d, 512) <= _MAX_2D_PROGRAMS:
        grid = lambda meta: (rows, triton.cdiv(d, meta["BLOCK_D"]))
        _gelu_tanh_and_mul_kernel_2d[grid](x2, out, d, x2.stride(0))
    else:
        grid = lambda meta: (rows * triton.cdiv(d, meta["BLOCK_D"]),)
        _gelu_tanh_and_mul_kernel_1d[grid](x2, out, d, x2.stride(0), rows)
    return out.view(*input.shape[:-1], d)


__all__ = ["gelu_tanh_and_mul"]
