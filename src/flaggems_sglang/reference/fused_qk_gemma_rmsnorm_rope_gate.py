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

"""Reference implementation for rope/fused_qk_gemma_rmsnorm_rope_gate."""

import torch


def _norm_rope(x, weight, cos, sin, eps, rotary_dim, out_dtype):
    """x: [T, H, head_dim]; cos/sin: [T, rotary_dim//2]."""
    x32 = x.to(torch.float32)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    # GemmaRMSNorm: weight is a delta around 1. The normalized value is rounded
    # to the output dtype *before* RoPE, matching the unfused two-kernel path.
    xn = (x32 * torch.rsqrt(var + eps) * (weight.to(torch.float32) + 1.0)).to(
        out_dtype
    )
    xn = xn.to(torch.float32)

    half = rotary_dim // 2
    x1, x2 = xn[..., :half], xn[..., half:rotary_dim]
    c, s = cos[:, None, :], sin[:, None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2, xn[..., rotary_dim:]], dim=-1).to(out_dtype)


def reference(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    t = q_gate.shape[0]
    dt = q_gate.dtype
    half = rotary_dim // 2
    cache = cos_sin_cache.to(torch.float32)[positions.long()]
    cos, sin = cache[:, :half], cache[:, half:rotary_dim]

    if has_gate:
        qg = q_gate.reshape(t, num_q_heads, 2, head_dim)
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :].contiguous()
    else:
        q, gate = q_gate.reshape(t, num_q_heads, head_dim), None

    q_out = _norm_rope(q, q_weight, cos, sin, eps, rotary_dim, dt)
    k_out = _norm_rope(
        k.reshape(t, num_kv_heads, head_dim),
        k_weight,
        cos,
        sin,
        eps,
        rotary_dim,
        dt,
    )
    return (
        q_out.reshape(t, num_q_heads * head_dim),
        k_out.reshape(t, num_kv_heads * head_dim),
        gate if has_gate else None,
    )
