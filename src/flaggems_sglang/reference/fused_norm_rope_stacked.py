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


def reference(
    kv,
    k_norm_weight,
    eps,
    cos_sin_cache,
    positions,
    num_kv_heads,
    head_dim,
    rotary_dim,
):
    T, L, _ = kv.shape
    H, D = num_kv_heads, head_dim
    kv_size = H * D
    half_rd = rotary_dim // 2

    k_all = kv[..., :kv_size].float().view(T, L, H, D)
    v_all = kv[..., kv_size:].view(T, L, H, D)

    w = k_norm_weight.float().view(1, L, 1, D)
    eps_l = eps.float().view(1, L, 1, 1)
    inv_rms = (k_all.pow(2).mean(dim=-1, keepdim=True) + eps_l).rsqrt()
    k_normed = k_all * inv_rms * w

    pos = positions.long()
    cos = cos_sin_cache[pos, :half_rd].float().view(T, 1, 1, half_rd)
    sin = cos_sin_cache[pos, half_rd:rotary_dim].float().view(T, 1, 1, half_rd)

    k1 = k_normed[..., :half_rd]
    k2 = k_normed[..., half_rd:rotary_dim]
    rot1 = k1 * cos - k2 * sin
    rot2 = k2 * cos + k1 * sin

    k_out = k_normed.clone()
    k_out[..., :half_rd] = rot1
    k_out[..., half_rd:rotary_dim] = rot2

    k_out = k_out.permute(1, 0, 2, 3).contiguous().to(kv.dtype)
    v_out = v_all.permute(1, 0, 2, 3).contiguous()
    return k_out, v_out
