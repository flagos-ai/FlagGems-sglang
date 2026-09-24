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


def reference(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    # page_table[i, p] = req_to_token[pool_i, p * page_size] // page_size
    #   for p < ceil(cache_seqlens[i] / page_size)
    # Vectorized: build the (bs, max_pages) index matrix once and gather.
    out = page_table.clone()
    bs = req_pool_indices.shape[0]
    max_pages = out.shape[1]
    device = out.device

    n_pages = (cache_seqlens + page_size - 1) // page_size  # [bs]
    page_idx = torch.arange(max_pages, device=device).unsqueeze(
        0
    )  # [1, max_pages]
    mask = page_idx < n_pages.unsqueeze(1)  # [bs, max_pages]

    pool = req_pool_indices.unsqueeze(1)  # [bs, 1]
    tok = req_to_token[pool, (page_idx * page_size).expand(bs, max_pages)]
    out[mask] = (tok // page_size)[mask].to(out.dtype)
    return out
