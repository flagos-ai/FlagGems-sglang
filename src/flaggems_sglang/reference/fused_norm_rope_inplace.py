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

"""Reference implementation for norm/fused_norm_rope_inplace."""

import torch


def reference(kv, weight, eps, freqs_cis, positions=None):
    k = kv.float()
    inv = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = k * inv
    if weight is not None:
        y = y * weight.float()
    rot = freqs_cis.shape[-2] * 2
    d = y.shape[-1]
    head = y[..., : d - rot]
    tail = y[..., d - rot :]
    if positions is not None:
        table = freqs_cis[positions.long()]
    else:
        table = freqs_cis[: y.shape[0]]
    c = table[..., 0].unsqueeze(-2).float()
    s = table[..., 1].unsqueeze(-2).float()
    x1 = tail[..., 0::2]
    x2 = tail[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    rot_out = torch.stack([o1, o2], dim=-1).reshape(tail.shape)
    return torch.cat([head, rot_out], dim=-1).to(kv.dtype)
