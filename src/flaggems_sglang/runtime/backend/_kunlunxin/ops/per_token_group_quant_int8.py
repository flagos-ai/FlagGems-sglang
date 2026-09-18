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
def _quant_1d(x_ptr, x_q_ptr, x_s_ptr, GROUP_SIZE: tl.constexpr):
    group_id = tl.program_id(0)
    lanes = tl.arange(0, GROUP_SIZE)
    offsets = group_id * GROUP_SIZE + lanes
    values = tl.load(x_ptr + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
    scale = amax / 127.0
    quantized = tl.minimum(
        tl.maximum(tl.div_rn(values, scale), -128.0),
        127.0,
    ).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offsets, quantized)
    tl.store(x_s_ptr + group_id, scale)


@triton.jit
def _quant_vec(
    x_ptr, x_q_ptr, x_s_ptr, GROUP_SIZE: tl.constexpr, NGROUPS: tl.constexpr
):
    group_ids = tl.program_id(0) * NGROUPS + tl.arange(0, NGROUPS)
    lanes = tl.arange(0, GROUP_SIZE)
    offsets = group_ids[:, None] * GROUP_SIZE + lanes[None, :]
    values = tl.load(x_ptr + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-10)
    scale = amax / 127.0
    quantized = tl.minimum(
        tl.maximum(tl.div_rn(values, scale[:, None]), -128.0),
        127.0,
    ).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offsets, quantized)
    tl.store(x_s_ptr + group_ids, scale)


def _groups_per_program(num_groups, group_size):
    for n in (64, 32, 16, 8, 4, 2, 1):
        if num_groups % n != 0:
            continue
        if n * group_size > 8192:
            continue
        if num_groups // n > 65535:
            continue
        return n
    return 1


def per_token_group_quant_int8(x, group_size, dtype=torch.int8):
    x_q = torch.empty_like(x, dtype=dtype)
    groups_per_row = x.shape[-1] // group_size
    x_s = torch.empty(
        x.shape[:-1] + (groups_per_row,),
        device=x.device,
        dtype=torch.float32,
    )
    num_groups = x.numel() // group_size
    ngroups = _groups_per_program(num_groups, group_size)
    if ngroups == 1:
        _quant_1d[(num_groups,)](
            x,
            x_q,
            x_s,
            GROUP_SIZE=group_size,
        )
    else:
        _quant_vec[(num_groups // ngroups,)](
            x,
            x_q,
            x_s,
            GROUP_SIZE=group_size,
            NGROUPS=ngroups,
        )
    return x_q, x_s


__all__ = ["per_token_group_quant_int8"]
