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

"""Reference implementation for diffusion/pack_qkv_destination_major."""

import torch


def reference(q, k, v, world_size, out):
    rows, global_heads, head_size = q.shape
    local_heads = global_heads // world_size

    def _split(t):
        # Heads are cut into `world_size` contiguous groups; group r is the
        # slice destined for rank r.
        return t.reshape(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )

    return torch.cat([_split(q), _split(k), _split(v)], dim=-1).contiguous()
