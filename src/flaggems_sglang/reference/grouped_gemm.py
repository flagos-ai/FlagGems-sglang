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

"""Reference implementation for gemm/grouped_gemm."""

import torch


def reference(
    a,
    b,
    num_experts,
    num_tokens_per_expert,
    expert_token_offs,
    expert_block_offs,
    expert_block_schedule,
):
    m, _ = a.shape
    _, n, _ = b.shape
    c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
    offs = expert_token_offs.tolist()
    for e in range(num_experts):
        lo, hi = offs[e], offs[e + 1]
        if hi <= lo:
            continue
        c[lo:hi] = (
            a[lo:hi].to(torch.float32) @ b[e].to(torch.float32).t()
        ).to(a.dtype)
    return c
