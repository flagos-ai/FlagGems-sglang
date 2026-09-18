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


@triton.jit
def _per_token_group_quant_int8_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    GROUP_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    lanes = tl.arange(0, GROUP_SIZE)
    offsets = group_id * GROUP_SIZE + lanes
    values = tl.load(x_ptr + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
    scale = amax / 127.0
    quantized = tl.clamp(
        tl.div_rn(values, scale),
        -128.0,
        127.0,
    ).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offsets, quantized)
    tl.store(x_s_ptr + group_id, scale)


@triton.jit
def _per_token_group_quant_int8_grouped_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    GROUP_SIZE: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    program_id = tl.program_id(0)
    group_ids = program_id * NGROUPS + tl.arange(0, NGROUPS)
    lanes = tl.arange(0, GROUP_SIZE)
    offsets = group_ids[:, None] * GROUP_SIZE + lanes[None, :]
    values = tl.load(x_ptr + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-10)
    scale = amax / 127.0
    quantized = tl.clamp(
        tl.div_rn(values, scale[:, None]),
        -128.0,
        127.0,
    ).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offsets, quantized)
    tl.store(x_s_ptr + group_ids, scale)


def _groups_per_program(num_groups, group_size):
    for n in (16, 8, 4, 2, 1):
        if num_groups % n != 0:
            continue
        if n * group_size > 2048:
            continue
        if num_groups // n > 65535:
            continue
        return n
    return 1


def _num_warps(elems):
    warps = elems // 256
    if warps >= 8:
        return 8
    if warps >= 4:
        return 4
    if warps >= 2:
        return 2
    return 1


def per_token_group_quant_int8(x, group_size, dtype=torch.int8):
    K = x.shape[-1]
    groups_per_row = K // group_size
    q = torch.empty_like(x, dtype=dtype)
    s = torch.empty(
        x.shape[:-1] + (groups_per_row,),
        device=x.device,
        dtype=torch.float32,
    )
    num_groups = x.numel() // group_size
    ngroups = _groups_per_program(num_groups, group_size)
    warps = _num_warps(group_size if ngroups == 1 else ngroups * group_size)
    grid = num_groups if ngroups == 1 else num_groups // ngroups
    if ngroups == 1:
        _per_token_group_quant_int8_kernel[(grid, 1, 1)](
            x,
            q,
            s,
            GROUP_SIZE=group_size,
            num_warps=warps,
            num_stages=1,
        )
    else:
        _per_token_group_quant_int8_grouped_kernel[(grid, 1, 1)](
            x,
            q,
            s,
            GROUP_SIZE=group_size,
            NGROUPS=ngroups,
            num_warps=warps,
            num_stages=1,
        )
    return q, s


__all__ = ["per_token_group_quant_int8"]
