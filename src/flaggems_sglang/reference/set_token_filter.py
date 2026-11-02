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


def reference(vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask):
    out = vocab_mask.clone()
    if reset_vocab_mask:
        out[batch_idx] = 0 if is_allowed else -1
    ids = torch.as_tensor(token_ids, dtype=torch.int64, device=out.device)
    bits = torch.zeros(
        (out.shape[1], 32), dtype=torch.int64, device=out.device
    )
    # Repeated token IDs are idempotent; summing bit values directly is not.
    bits.view(-1)[ids] = 1
    shifts = torch.arange(32, dtype=torch.int64, device=out.device)
    mask = (bits << shifts).sum(dim=-1).to(out.dtype)
    out[batch_idx] = (
        out[batch_idx] | mask if is_allowed else out[batch_idx] & ~mask
    )
    return out
