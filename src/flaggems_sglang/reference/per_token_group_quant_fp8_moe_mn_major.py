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

"""Reference implementation for quantization/per_token_group_quant_fp8_moe_mn_major."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_AMAX_MIN = 1e-4


def reference(A, expert_offsets, problem_sizes, group_size):
    M, K = A.shape
    groups = K // group_size
    row = torch.arange(M, device=A.device)
    if problem_sizes.shape[0] == 0:
        return (
            torch.zeros((M, K), dtype=torch.uint8, device=A.device).view(
                _FP8_DTYPE
            ),
            torch.zeros((M, groups), dtype=torch.float32, device=A.device),
        )
    expert = (
        torch.searchsorted(expert_offsets.contiguous(), row, right=True) - 1
    )
    safe = expert.clamp(0, problem_sizes.shape[0] - 1).long()
    off = expert_offsets[safe].long()
    count = problem_sizes[safe, 0].long()
    valid = (expert >= 0) & (row >= off) & (row < off + count)
    values = A.float().reshape(M, groups, group_size)
    amax = values.abs().amax(dim=-1).clamp(min=_AMAX_MIN)
    q = (
        (values * (_FP8_MAX / amax[..., None]))
        .clamp(-_FP8_MAX, _FP8_MAX)
        .reshape(M, K)
    )
    q = torch.where(valid[:, None], q, torch.zeros_like(q))
    a_q = q.to(_FP8_DTYPE)
    sfa = torch.zeros((M, groups), dtype=torch.float32, device=A.device)
    gi = torch.arange(groups, device=A.device)
    dest = (
        off[:, None] * groups
        + gi[None, :] * count[:, None]
        + (row - off)[:, None]
    )
    sfa.view(-1)[dest[valid].reshape(-1)] = (amax[valid] / _FP8_MAX).reshape(-1)
    return a_q, sfa
