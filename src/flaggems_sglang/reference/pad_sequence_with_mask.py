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

"""Reference for attention/pad_sequence_with_mask."""

import torch


def reference(input_emb, offsets, lengths, max_len):
    batch = int(offsets.shape[0])
    hidden = int(input_emb.shape[1])

    padded = torch.zeros(
        (batch, max_len, hidden),
        dtype=input_emb.dtype,
        device=input_emb.device,
    )
    valid_mask = torch.zeros(
        (batch, max_len),
        dtype=torch.bool,
        device=input_emb.device,
    )

    for b in range(batch):
        start = int(offsets[b])
        # A sequence longer than the padded slot is truncated, matching the
        # kernel's ``rows < length`` combined with ``rows < max_len``.
        count = min(int(lengths[b]), max_len)
        if count > 0:
            padded[b, :count] = input_emb[start : start + count]
            valid_mask[b, :count] = True

    return batch, padded, valid_mask.view(-1)
