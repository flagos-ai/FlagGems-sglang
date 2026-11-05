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

"""Reference implementation for diffusion/fused_qk_inv_rms."""

import torch


def reference(qkv, eps):
    # qkv: (B, N, 3, H, D). The RMS is taken over the whole (H, D) block of a
    # token, not per head.
    x = qkv.to(torch.float32)
    q, k = x[:, :, 0], x[:, :, 1]
    q_inv = torch.rsqrt(q.pow(2).mean(dim=(-2, -1)) + eps)
    k_inv = torch.rsqrt(k.pow(2).mean(dim=(-2, -1)) + eps)
    return q_inv.to(torch.float32), k_inv.to(torch.float32)
