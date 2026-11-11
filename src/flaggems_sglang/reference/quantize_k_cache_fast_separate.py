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

"""Reference for kv_cache/quantize_k_cache_fast_separate.

The same quantization as ``quantize_k_cache_fast``, returned as two byte
buffers instead of one packed row::

    nonrotary_bytes: [dim_nope fp8 | num_tiles fp32 scales]
    rotary_bytes:    [dim_rope bf16]

with ``num_tiles = dim_nope // group_size``. Each NoPE tile gets one fp32
scale, ``amax(|tile|) / 448.0``, and is divided by it before the cast; the
RoPE columns are copied through untouched.
"""

import torch

_FP8_MAX = 448.0


def reference(k_nope, k_rope, group_size=128):
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens, dim_nope = k_nope_2d.shape
    dim_rope = int(k_rope_2d.shape[1])
    num_tiles = dim_nope // group_size

    nope_part = torch.empty(
        (num_tokens, dim_nope + num_tiles * 4),
        dtype=torch.uint8,
        device=k_nope.device,
    )
    nope_q = nope_part[:, :dim_nope].view(torch.float8_e4m3fn)
    nope_s = nope_part[:, dim_nope:].view(torch.float32)

    if dim_rope > 0:
        rope_part = torch.empty(
            (num_tokens, dim_rope * 2),
            dtype=torch.uint8,
            device=k_rope.device,
        )
        rope_part.view(torch.bfloat16)[:] = k_rope_2d
    else:
        rope_part = torch.empty(
            (num_tokens, 0), dtype=torch.uint8, device=k_rope.device
        )

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        tile = k_nope_2d[:, lo:hi].float()
        # No guard on the amax: a tile of all zeros divides zero by zero, which
        # is what the contract's own reference does too.
        scale = tile.abs().max(dim=-1).values / _FP8_MAX
        nope_s[:, tile_idx] = scale
        nope_q[:, lo:hi] = (tile / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)
