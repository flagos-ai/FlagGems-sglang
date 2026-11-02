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
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    rows = out_cache_loc.reshape(
        num_correct_drafts.shape[0], num_verify_tokens
    )
    pos = torch.arange(num_verify_tokens, device=rows.device)[None, :]
    tgt = rows[pos < (num_correct_drafts[:, None] + 1)]
    free = rows[pos >= (num_verify_tokens - to_free_num_slots[:, None])]
    # The contract reserves a zero sentinel even for an empty result.
    if tgt.numel() == 0:
        tgt = out_cache_loc.new_zeros(1)
    if free.numel() == 0:
        free = out_cache_loc.new_zeros(1)
    return tgt, free
