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

import torch
import triton
import triton.language as tl

MAX_ELEMS = 4096


@triton.jit
def _quant_exact(
    x_ptr, q_ptr, s_ptr, GROUP_SIZE: tl.constexpr, GPP: tl.constexpr
):
    g = tl.program_id(0) * GPP + tl.arange(0, GPP)
    offs = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 127.0 / amax
    q = tl.clamp(x * inv[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q)
    tl.store(s_ptr + g, amax / 127.0)


@triton.jit
def _quant_masked(
    x_ptr,
    q_ptr,
    s_ptr,
    NUM_GROUPS,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GPP: tl.constexpr,
):
    g = tl.program_id(0) * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, BLOCK)
    gmask = g < NUM_GROUPS
    mask = gmask[:, None] & (lanes[None, :] < GROUP_SIZE)
    offs = g[:, None] * GROUP_SIZE + lanes[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 127.0 / amax
    q = tl.clamp(x * inv[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + g, amax / 127.0, mask=gmask)


def per_token_group_quant_int8(x, group_size, dtype=torch.int8):
    shape = x.shape
    cols = shape[-1]
    num_groups = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    gpp = 1
    while gpp * 2 * block <= MAX_ELEMS and num_groups % (gpp * 2) == 0:
        gpp *= 2
    warps = 16
    q = torch.empty(shape, device=x.device, dtype=dtype)
    s = torch.empty(
        shape[:-1] + (cols // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(num_groups, gpp), 1, 1)
    if block == group_size:
        _quant_exact[grid](
            x,
            q,
            s,
            GROUP_SIZE=group_size,
            GPP=gpp,
            num_warps=warps,
            num_stages=1,
        )
    else:
        _quant_masked[grid](
            x,
            q,
            s,
            num_groups,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GPP=gpp,
            num_warps=warps,
            num_stages=1,
        )
    return q, s


__all__ = ["per_token_group_quant_int8"]
