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

"""Reference implementation for dsa/triton_sparse_mla_fwd."""

import torch


def reference(q_nope, q_rope, kv, indices, sm_scale, d_v):
    seq, _, _ = q_nope.shape
    dim = kv.shape[-1]
    topk = indices.shape[-1]

    q = torch.cat([q_nope.to(torch.float32), q_rope.to(torch.float32)], dim=-1)
    idx = indices.reshape(seq, topk)
    valid = idx >= 0
    gathered = kv.reshape(-1, dim).to(torch.float32)[
        idx.clamp(min=0).to(torch.int64)
    ]

    # kv[idx, :d_v] serves as both K's main half and V.
    qk = torch.einsum("shd,skd->shk", q, gathered) * sm_scale
    qk = torch.where(valid[:, None, :], qk, torch.full_like(qk, float("-inf")))

    # A row whose every key is invalid must produce zeros, not NaN.
    p = torch.softmax(qk, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)

    out = torch.einsum("shk,skd->shd", p, gathered[..., :d_v])
    return out.to(torch.bfloat16).unsqueeze(0)
