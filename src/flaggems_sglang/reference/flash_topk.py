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


"""Independent PyTorch reference for the SGLang DeepSeek V3.2 indexer."""

import torch


def score_row(query, weights, index_k_cache, page_table, length):
    """Score a single visible request window, independently of the GPU kernel."""
    tokens = torch.arange(length, device=query.device)
    physical = page_table[tokens // 64].long()
    keys = index_k_cache[:, :8192].view(torch.float8_e4m3fn)
    keys = keys.reshape(-1, 64, 128).float()[physical, tokens % 64]
    scales = index_k_cache[:, 8192:].view(torch.float32)[physical, tokens % 64]
    dots = keys @ query.float().T
    return (dots.relu() * weights.reshape(-1).float()).sum(dim=1) * scales


def reference(
    query,
    weights,
    index_k_cache,
    index_page_table,
    row_to_request,
    visible_lengths,
    max_context,
    out,
    workspace=None,
    *,
    prefill=False,
    output_page_table=None,
    output_offsets=None,
):
    """Reference result with the same consumer-space output convention."""
    out.fill_(-1)
    for row, length in enumerate(visible_lengths.tolist()):
        if not length:
            continue
        request = int(row_to_request[row])
        count = min(2048, length)
        if length <= 2048:
            ids = torch.arange(length, device=query.device)
        else:
            scores = score_row(
                query[row],
                weights[row],
                index_k_cache,
                index_page_table[request],
                length,
            )
            ids = scores.topk(count).indices
        if output_page_table is not None:
            ids = output_page_table[request, ids // 64] * 64 + ids % 64
        elif output_offsets is not None:
            ids = ids + output_offsets[row]
        out[row, :count].copy_(ids)
    return out
