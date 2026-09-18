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
    next_token_logits, positions, draft_tokens=None, draft_token_column=0
):
    bs = next_token_logits.shape[0]
    topk_index = next_token_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
    topk_p = torch.ones(
        bs, 1, dtype=torch.float32, device=next_token_logits.device
    )

    out_positions = positions + 1
    out_draft_tokens = None
    if draft_tokens is not None:
        out_draft_tokens = draft_tokens.clone()
        out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)

    return topk_p, topk_index, out_positions, out_draft_tokens
