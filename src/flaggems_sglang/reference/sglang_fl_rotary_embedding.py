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

"""Reference for SGLang fused in-place RoPE.

Mirrors ``sgl_kernel.apply_rope_with_cos_sin_cache_inplace`` /
``sglang/srt/layers/rotary_embedding.py``:

    positions      [nnz] int64, indices into ``cos_sin_cache``
    query          [nnz, num_q_heads * head_size]
    key            [nnz, num_kv_heads * head_size]
    cos_sin_cache  [max_position, rotary_dim], cos in ``[:, :rd/2]`` and
                   sin in ``[:, rd/2:]``
    head_size      int; rotary_dim = cos_sin_cache.shape[-1]
    is_neox        True  -> rotate halves ``[0, rd/2)`` against ``[rd/2, rd)``
                   False -> rotate even/odd lane pairs

Lanes ``[rotary_dim, head_size)`` of every head pass through untouched. The op
rotates ``query``/``key`` in place and also returns them so the correctness
harness can compare the produced tensors.
"""

import torch


def _rotate(x, cos, sin, head_size, rotary_dim, is_neox):
    nnz = x.shape[0]
    xf = x.float().view(nnz, -1, head_size)
    rot = xf[..., :rotary_dim]
    passthrough = xf[..., rotary_dim:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    if is_neox:
        half = rotary_dim // 2
        x1 = rot[..., :half]
        x2 = rot[..., half:]
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        rotated = torch.cat([o1, o2], dim=-1)
    else:
        x1 = rot[..., 0::2]
        x2 = rot[..., 1::2]
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        rotated = torch.stack([o1, o2], dim=-1).reshape(x1.shape[:-1] + (-1,))
    out = torch.cat([rotated, passthrough], dim=-1).reshape(nnz, -1)
    return out.to(x.dtype)


def reference(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    cs = cos_sin_cache[positions.long()].float()
    cos = cs[:, :half]
    sin = cs[:, half:]
    q_out = _rotate(query, cos, sin, head_size, rotary_dim, is_neox)
    k_out = _rotate(key, cos, sin, head_size, rotary_dim, is_neox)
    return q_out, k_out
