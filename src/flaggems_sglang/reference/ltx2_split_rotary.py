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


def reference(x, cos, sin):
    batch, seq_len, inner = x.shape
    _, num_heads, _, half = cos.shape
    head_dim = half * 2
    xv = x.reshape(batch, seq_len, num_heads, head_dim)
    # cos/sin are [B, H, T, half]; align to [B, T, H, half].
    c = cos.permute(0, 2, 1, 3)
    s = sin.permute(0, 2, 1, 3)

    x1, x2 = xv[..., :half], xv[..., half:]
    # The intermediate bf16 rounding is deliberate: it reproduces the original
    # PyTorch order, where x * cos materializes as bf16 before the fp32
    # addcmul_. Keeping the product in fp32 throughout would not match.
    o1 = (x1 * c).to(torch.bfloat16).float() - x2.float() * s.float()
    o2 = (x2 * c).to(torch.bfloat16).float() + x1.float() * s.float()
    out = torch.cat([o1, o2], dim=-1)
    return out.reshape(batch, seq_len, inner).to(x.dtype)
