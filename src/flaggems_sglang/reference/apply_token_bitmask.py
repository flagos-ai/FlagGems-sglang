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


def reference(logits, bitmask):
    B, V = logits.shape
    v_idx = torch.arange(V, device=logits.device)
    word_idx = v_idx // 32
    bit_idx = v_idx % 32
    bits = (bitmask[:, word_idx] >> bit_idx) & 1
    return torch.where(
        bits == 0, torch.full_like(logits, float("-inf")), logits
    )
