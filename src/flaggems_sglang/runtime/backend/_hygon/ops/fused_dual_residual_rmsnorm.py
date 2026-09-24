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
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    rms1 = tl.sqrt(tl.sum(x * x, axis=0) * (1.0 / hidden) + eps)
    weight1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized1 = ((x / rms1) * weight1).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + row * stride_residual + cols, mask=mask, other=0.0
    )
    mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
    tl.store(mid_ptr + row * stride_mid + cols, mid, mask=mask)
    mid32 = mid.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(mid32 * mid32, axis=0) * (1.0 / hidden) + eps)
    weight2 = tl.load(weight2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output + cols,
        (mid32 / rms2) * weight2,
        mask=mask,
    )


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
    batch, hidden = x.shape
    output, mid = torch.empty_like(x), torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
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
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size <= 2048 else 8,
        num_stages=1,
    )
    return output, mid
