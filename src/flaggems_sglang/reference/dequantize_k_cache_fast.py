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

"""Reference implementation for dsa/dequantize_k_cache_fast."""

import torch

_DIM_NOPE = 512
_DIM_ROPE = 64


def reference(quant_k_cache, group_size=128):
    num_tokens = quant_k_cache.shape[0]
    num_tiles = _DIM_NOPE // group_size

    output = torch.empty(
        (num_tokens, _DIM_NOPE + _DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    # 656 bytes per row: [512 nope fp8 | 4 fp32 scales | 64 bf16 rope]. The
    # scale and rope fields are bit-cast out of the fp8 storage, not
    # numerically converted.
    input_nope_q = quant_k_cache[:, :_DIM_NOPE]
    input_nope_s = quant_k_cache[
        :, _DIM_NOPE : _DIM_NOPE + num_tiles * 4
    ].view(torch.float32)
    input_rope = quant_k_cache[:, _DIM_NOPE + num_tiles * 4 :].view(
        torch.bfloat16
    )
    output[:, _DIM_NOPE:] = input_rope

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        cur_nope = input_nope_q[:, lo:hi].to(torch.float32)
        cur_scale = input_nope_s[:, tile_idx].unsqueeze(-1)
        output[:, lo:hi] = cur_nope * cur_scale

    return output
