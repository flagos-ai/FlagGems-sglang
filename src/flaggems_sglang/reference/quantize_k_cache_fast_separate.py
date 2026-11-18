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

"""Reference implementation for dsa/quantize_k_cache_fast_separate."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn


def reference(k_nope, k_rope, group_size=128):
    num_tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    num_tiles = dim_nope // group_size
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    # Split buffers rather than one concatenated row: same quantization math,
    # one fewer concat for the caller feeding set_mla_kv_buffer_triton.
    nope_part_bytes = dim_nope + num_tiles * 4
    rope_part_bytes = dim_rope * k_rope.element_size()

    nope_part_u8 = torch.empty(
        num_tokens, nope_part_bytes, dtype=torch.uint8, device=k_nope.device
    )
    rope_part_u8 = torch.empty(
        num_tokens, rope_part_bytes, dtype=torch.uint8, device=k_rope.device
    )

    nope_q = nope_part_u8[:, :dim_nope].view(_FP8_DTYPE)
    nope_s = nope_part_u8[:, dim_nope:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)
    rope_view[:] = k_rope

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        tile = k_nope[:, lo:hi].float()
        scale = tile.abs().amax(dim=-1) / fp8_max
        nope_s[:, tile_idx] = scale
        quant = (
            (tile / scale.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)
        )
        nope_q[:, lo:hi] = quant

    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)
