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

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_token_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    out_ptr,
    V,
    NW,
    NWB: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_w = pid_w * NWB + tl.arange(0, NWB)
    words = tl.load(
        bitmask_ptr + pid_b * NW + offs_w,
        mask=offs_w < NW,
        other=0,
    )
    bits = tl.arange(0, 32)
    keep = tl.reshape(
        ((words[:, None] >> bits[None, :]) & 1) != 0,
        (NWB * 32,),
    )

    offs_v = pid_w * (NWB * 32) + tl.arange(0, NWB * 32)
    valid = offs_v < V
    values = tl.load(
        logits_ptr + pid_b * V + offs_v,
        mask=valid,
        other=0.0,
    )
    result = tl.where(keep, values, float("-inf"))
    tl.store(out_ptr + pid_b * V + offs_v, result, mask=valid)


def apply_token_bitmask(logits, bitmask):
    B, V = logits.shape
    logits = logits.contiguous()
    bitmask = bitmask.contiguous()
    out = torch.empty_like(logits)
    if B == 0 or V == 0:
        return out

    NW = bitmask.shape[1]
    NWB = 32
    _apply_token_bitmask_kernel[(B, triton.cdiv(NW, NWB))](
        logits,
        bitmask,
        out,
        V,
        NW,
        NWB=NWB,
        num_warps=1,
    )
    return out


reference = apply_token_bitmask

__all__ = ["apply_token_bitmask"]
