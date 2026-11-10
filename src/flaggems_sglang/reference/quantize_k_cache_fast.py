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

"""Reference for kv_cache/quantize_k_cache_fast.

A packed row is ``[dim_nope fp8 | num_tiles fp32 scales | dim_rope bf16]``
with ``num_tiles = dim_nope // group_size``. Each NoPE tile gets one fp32
scale, ``amax(|tile|) / 448.0``, and is divided by it before the cast; the
RoPE columns are copied through untouched. This is the exact inverse of what
``dequantize_k_cache_fast`` multiplies back out.
"""

import torch

_FP8_MAX = 448.0


def reference(k_nope, k_rope, group_size=128):
    original_ndim = k_nope.ndim
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens, dim_nope = k_nope_2d.shape
    dim_rope = int(k_rope_2d.shape[1])
    num_tiles = dim_nope // group_size

    packed = torch.empty(
        (num_tokens, dim_nope + num_tiles * 4 + 2 * dim_rope),
        dtype=torch.float8_e4m3fn,
        device=k_nope.device,
    )
    nope_q = packed[:, :dim_nope]
    nope_s = packed[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope_out = packed[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    if dim_rope > 0:
        rope_out[:] = k_rope_2d

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        tile = k_nope_2d[:, lo:hi].float()
        # No guard on the amax: a tile of all zeros divides zero by zero, which
        # is what the contract's own reference does too.
        scale = tile.abs().max(dim=-1).values / _FP8_MAX
        nope_s[:, tile_idx] = scale
        nope_q[:, lo:hi] = (tile / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    if original_ndim == 3:
        return packed.view(num_tokens, 1, packed.shape[1])
    return packed
