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

"""Reference implementation for speculative/assign_extend_cache_locs_uniform."""

import torch


def reference(
    req_pool_indices,
    req_to_token,
    start_offset,
    batch_size,
    draft_token_num,
    device,
):
    # Every row extends by exactly draft_token_num, so the output offset is
    # pid * draft_token_num: no end-offset tensor and no prefix sum needed.
    offsets = torch.arange(draft_token_num, device=device)
    cols = start_offset.long()[:, None] + offsets[None, :]
    rows = req_pool_indices.long()[:, None].expand(-1, draft_token_num)
    return req_to_token[rows, cols].reshape(-1).to(torch.int64)
