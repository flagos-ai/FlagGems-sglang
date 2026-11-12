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

"""Construct the output and split-KV buffers for native decode attention."""

import torch


def decode_inputs(inputs, device):
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale = inputs
    batch, query_heads, _ = q.shape
    value_dim = v_buffer.shape[-1]
    max_splits = 8
    out = q.new_zeros((batch, query_heads, value_dim))
    logits = torch.zeros(
        (batch, query_heads, max_splits, value_dim),
        dtype=torch.float32,
        device=device,
    )
    lse = torch.zeros(
        (batch, query_heads, max_splits), dtype=torch.float32, device=device
    )
    num_splits = torch.full((batch,), 4, dtype=torch.int32, device=device)
    return (
        q,
        k_buffer,
        v_buffer,
        out,
        kv_indptr,
        kv_indices,
        logits,
        lse,
        num_splits,
        max_splits,
        sm_scale,
        0.0,
    ), {}
