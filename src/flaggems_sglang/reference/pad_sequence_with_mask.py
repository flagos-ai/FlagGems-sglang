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

"""Reference implementation for attention/pad_sequence_with_mask."""

import torch


def reference(input_emb, offsets, lengths, max_len):
    b = offsets.shape[0]
    hidden = input_emb.shape[1]
    output = torch.zeros(
        (b, max_len, hidden), device=input_emb.device, dtype=input_emb.dtype
    )
    seq = torch.arange(max_len, device=input_emb.device)
    valid = seq[None, :] < lengths.long()[:, None]
    rows = offsets.long()[:, None] + seq[None, :]
    gathered = input_emb[rows.clamp(min=0, max=input_emb.shape[0] - 1)]
    output = torch.where(valid[..., None], gathered, output)
    # The mask comes back flattened to [b * max_len]: the kernel's own layout.
    return b, output, valid.reshape(-1)
