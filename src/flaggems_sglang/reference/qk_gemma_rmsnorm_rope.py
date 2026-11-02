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

"""Reference implementation for rope/qk_gemma_rmsnorm_rope."""

import torch


def _norm_rope(
    x, weight, cos_sin_cache, positions, eps, head_dim, rotary_dim, neox
):
    m, cols = x.shape
    heads = cols // head_dim
    x32 = x.to(torch.float32).reshape(m, heads, head_dim)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps) * (1.0 + weight.to(torch.float32))
    # Match the unfused path: GemmaRMSNorm writes bf16/fp16 and RoPE reads that
    # rounded value back.
    normed = normed.to(x.dtype).to(torch.float32)

    half = rotary_dim // 2
    cache = cos_sin_cache.to(torch.float32)[
        positions.long()
    ]  # [M, rotary_dim]
    cos_half, sin_half = cache[:, :half], cache[:, half:rotary_dim]

    rot = normed[..., :rotary_dim]
    if neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        c, s = cos_half[:, None, :], sin_half[:, None, :]
        rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        c, s = cos_half[:, None, :], sin_half[:, None, :]
        o1, o2 = x1 * c - x2 * s, x2 * c + x1 * s
        rotated = torch.stack([o1, o2], dim=-1).flatten(-2)

    out = torch.cat([rotated, normed[..., rotary_dim:]], dim=-1)
    return out.to(x.dtype).reshape(m, cols)


def reference(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    eps,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_out = _norm_rope(
        q,
        q_weight,
        cos_sin_cache,
        positions,
        eps,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    k_out = _norm_rope(
        k,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    return q_out, k_out
