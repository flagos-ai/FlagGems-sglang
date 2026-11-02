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


def reference(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE):
    b, h, _, d = q.shape
    cache = kv_caches.clone()
    out = torch.empty_like(q)

    decay = torch.exp(-slope_rate.to(torch.float32)).reshape(h)  # per head

    for bi in range(b):
        slot = int(slot_idx[bi])
        if slot == -1:
            # Padding slot: neither the cache nor the output row is touched.
            continue
        qi = q[bi, :, 0].to(torch.float32)  # [H, D]
        ki = k[bi, :, 0].to(torch.float32)
        vi = v[bi, :, 0].to(torch.float32)

        kv_outer = ki[:, :, None] * vi[:, None, :]  # [H, D(k), D(v)]
        new_state = kv_outer + decay[:, None, None] * cache[slot].to(
            torch.float32
        )
        cache[slot] = new_state.to(cache.dtype)
        out[bi, :, 0] = (qi[:, :, None] * new_state).sum(dim=1).to(out.dtype)

    # The baseline returns [B, H*D]: `rearrange(b h n d -> b n (h d))` then
    # squeeze of the length-1 token axis.
    return out.squeeze(2).reshape(b, h * d).contiguous(), cache
