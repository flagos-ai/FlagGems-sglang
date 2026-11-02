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
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    topk,
    speculative_num_steps,
):
    out = out_cache_loc.clone()
    n = topk * speculative_num_steps
    columns = seq_lens[: req_pool_indices.numel(), None] + torch.arange(
        n, device=seq_lens.device
    )
    values = req_to_token[req_pool_indices[:, None].long(), columns.long()]
    out[: values.numel()] = values.reshape(-1)
    return out
