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
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    b = mixed_qkvz.shape[0]
    r = num_heads_v // num_heads_qk

    qkvz = mixed_qkvz.reshape(b, num_heads_qk, 2 * head_qk + 2 * r * head_v)
    q = qkvz[:, :, :head_qk]
    k = qkvz[:, :, head_qk : 2 * head_qk]
    v = qkvz[:, :, 2 * head_qk : 2 * head_qk + r * head_v]
    z = qkvz[:, :, 2 * head_qk + r * head_v :]

    # mixed_qkv packs all Q heads, then all K heads, then all V.
    mixed_qkv = torch.cat(
        [q.reshape(b, -1), k.reshape(b, -1), v.reshape(b, -1)], dim=-1
    ).contiguous()

    ba = mixed_ba.reshape(b, num_heads_qk, 2 * r)
    b_out = ba[:, :, :r].reshape(b, num_heads_v).contiguous()
    a_out = ba[:, :, r:].reshape(b, num_heads_v).contiguous()
    return (
        mixed_qkv,
        z.reshape(b, num_heads_v, head_v).contiguous(),
        b_out,
        a_out,
    )
