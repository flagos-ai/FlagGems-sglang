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

"""Single-pass dual residual RMSNorm implemented in portable Triton."""

import torch
import triton
import triton.language as tl

__all__ = ["fused_dual_residual_rmsnorm"]


@triton.jit
def _fused_dual_residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    output_ptr,
    mid_ptr,
    stride_x,
    stride_residual,
    stride_output,
    stride_mid,
    hidden: tl.constexpr,
    eps,
    RMS1_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_sq1 = tl.sum(x * x, axis=0)
    rms1 = tl.sqrt(sum_sq1 * (1.0 / hidden) + eps) * RMS1_SCALE
    weight1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # The reference rounds the first normalized value before adding residual,
    # then rounds the sum again to residual.dtype. Preserve both boundaries.
    normalized1 = ((x / rms1) * weight1).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + row * stride_residual + cols, mask=mask, other=0.0
    )
    mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
    tl.store(mid_ptr + row * stride_mid + cols, mid, mask=mask)

    mid32 = mid.to(tl.float32)
    sum_sq2 = tl.sum(mid32 * mid32, axis=0)
    rms2 = tl.sqrt(sum_sq2 * (1.0 / hidden) + eps)
    weight2 = tl.load(weight2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    output = (mid32 / rms2) * weight2
    tl.store(output_ptr + row * stride_output + cols, output, mask=mask)


@triton.jit
def _normalized1_kernel(
    x_ptr,
    weight_ptr,
    normalized_ptr,
    stride_x,
    stride_normalized,
    hidden: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized = ((values / rms) * weight).to(normalized_ptr.dtype.element_ty)
    tl.store(
        normalized_ptr + row * stride_normalized + cols, normalized, mask=mask
    )


@triton.jit
def _add_residual_kernel(
    normalized_ptr,
    residual_ptr,
    mid_ptr,
    stride_normalized,
    stride_residual,
    stride_mid,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    normalized = tl.load(
        normalized_ptr + row * stride_normalized + cols, mask=mask, other=0.0
    )
    residual = tl.load(
        residual_ptr + row * stride_residual + cols, mask=mask, other=0.0
    )
    tl.store(
        mid_ptr + row * stride_mid + cols, residual + normalized, mask=mask
    )


@triton.jit
def _normalized2_kernel(
    mid_ptr,
    weight_ptr,
    output_ptr,
    stride_mid,
    stride_output,
    hidden: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(
        mid_ptr + row * stride_mid + cols, mask=mask, other=0.0
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output + cols,
        (values / rms) * weight,
        mask=mask,
    )


@triton.jit
def _quotient_kernel(
    value_ptr,
    quotient_ptr,
    stride_value,
    stride_quotient,
    hidden: tl.constexpr,
    eps,
    RMS_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(
        value_ptr + row * stride_value + cols, mask=mask, other=0.0
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    rms = rms * RMS_SCALE
    tl.store(
        quotient_ptr + row * stride_quotient + cols, values / rms, mask=mask
    )


@triton.jit
def _apply_weight_kernel(
    quotient_ptr,
    weight_ptr,
    result_ptr,
    stride_quotient,
    stride_result,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    quotient = tl.load(
        quotient_ptr + row * stride_quotient + cols, mask=mask, other=0.0
    )
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        result_ptr + row * stride_result + cols, quotient * weight, mask=mask
    )


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
    batch, hidden = x.shape
    output = torch.empty_like(x)
    mid = torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    if hidden > 8192:
        quotient = torch.empty_like(x, dtype=torch.float32)
        normalized = torch.empty_like(residual)
        _quotient_kernel[(batch,)](
            x,
            quotient,
            x.stride(0),
            quotient.stride(0),
            hidden=hidden,
            eps=eps,
            RMS_SCALE=0.9999998807907104,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _apply_weight_kernel[(batch,)](
            quotient,
            weight1,
            normalized,
            quotient.stride(0),
            normalized.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _add_residual_kernel[(batch,)](
            normalized,
            residual,
            mid,
            normalized.stride(0),
            residual.stride(0),
            mid.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _quotient_kernel[(batch,)](
            mid,
            quotient,
            mid.stride(0),
            quotient.stride(0),
            hidden=hidden,
            eps=eps,
            RMS_SCALE=1.0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _apply_weight_kernel[(batch,)](
            quotient,
            weight2,
            output,
            quotient.stride(0),
            output.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return output, mid
    _fused_dual_residual_rmsnorm_kernel[(batch,)](
        x,
        residual,
        weight1,
        weight2,
        output,
        mid,
        x.stride(0),
        residual.stride(0),
        output.stride(0),
        mid.stride(0),
        hidden=hidden,
        eps=eps,
        RMS1_SCALE=0.9999998807907104 if hidden >= 8192 else 1.0,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output, mid
