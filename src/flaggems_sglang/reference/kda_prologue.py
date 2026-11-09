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

"""Reference implementation for fla/kda_prologue."""

import torch

_BT = 64


def reference(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    outputs = [torch.zeros_like(q, dtype=torch.bfloat16) for _ in range(5)]
    g_cu_out = torch.zeros_like(q, dtype=torch.float32)
    if num_chunks == 0 or q.shape[0] == 0:
        return (*outputs, g_cu_out)
    chunks = chunk_indices[:num_chunks].long()
    start = cu_seqlens[chunks[:, 0]].long() + chunks[:, 1] * _BT
    end = cu_seqlens[chunks[:, 0] + 1].long()
    idx = start[:, None] + torch.arange(_BT, device=q.device)
    valid = idx < end[:, None]
    safe = idx.clamp(0, q.shape[0] - 1)
    g = torch.where(
        valid[..., None, None],
        g_act.index_select(0, safe.reshape(-1))
        .reshape(*safe.shape, *g_act.shape[1:])
        .float(),
        0.0,
    )
    qs = (
        q.index_select(0, safe.reshape(-1))
        .reshape(*safe.shape, *q.shape[1:])
        .float()
    )
    ks = (
        k.index_select(0, safe.reshape(-1))
        .reshape(*safe.shape, *k.shape[1:])
        .float()
    )
    gc = g.cumsum(dim=1)
    gl = gc - g.sum(dim=1, keepdim=True)
    dest = idx[valid]
    g_cu_out[dest] = gc[valid]
    outputs[0][dest] = (ks * gl.exp())[valid].to(torch.bfloat16)
    outputs[1][dest] = (ks * (-gl).exp())[valid].to(torch.bfloat16)
    outputs[2][dest] = (ks * gc.exp())[valid].to(torch.bfloat16)
    outputs[3][dest] = (scale * qs * gc.exp())[valid].to(torch.bfloat16)
    outputs[4][dest] = (scale * qs * gl.exp())[valid].to(torch.bfloat16)
    return (*outputs, g_cu_out)
