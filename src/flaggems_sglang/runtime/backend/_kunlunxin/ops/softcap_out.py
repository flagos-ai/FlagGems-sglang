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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _softcap_kernel(
    x_ptr,
    out_ptr,
    softcap_const,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_i64 = offs.to(tl.int64)
    mask = offs_i64 < n_elements

    x = tl.load(x_ptr + offs_i64, mask=mask, other=0.0).to(tl.float32)

    z = x / softcap_const

    abs_z = tl.abs(z)
    exp_neg2 = tl.exp(-2.0 * abs_z)

    numerator = 1.0 - exp_neg2
    denominator = 1.0 + exp_neg2

    sign_z = tl.where(z > 0.0, 1.0, tl.where(z < 0.0, -1.0, 0.0))

    tanh_z = tl.where(z == 0.0, 0.0, sign_z * numerator / denominator)

    out = softcap_const * tanh_z

    tl.store(out_ptr + offs_i64, out.to(tl.float32), mask=mask)


def softcap_out(x: torch.Tensor, softcap_const: float) -> torch.Tensor:
    if x.numel() == 0:
        return x.to(torch.float32).contiguous()

    x_fp32 = x.to(torch.float32).contiguous()
    out = torch.empty_like(x_fp32)

    n_elements = x_fp32.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _softcap_kernel[grid](
        x_fp32,
        out,
        float(softcap_const),
        n_elements,
    )

    return out

__all__ = ["softcap_out"]