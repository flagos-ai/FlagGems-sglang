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

"""Reference implementation for dsa/dequantize_k_cache_paged."""

import torch

_DIM_NOPE = 512
_DIM_ROPE = 64


def reference(quant_k_cache, page_table_1_flattened, group_size=128):
    quant_k_cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    num_tiles = _DIM_NOPE // group_size

    # The gather is the only thing on top of dequantize_k_cache_fast; several
    # output rows may map to the same slot under prefix sharing.
    idx = page_table_1_flattened.long()
    gathered = quant_k_cache[idx]

    output = torch.empty(
        gathered.shape[0],
        _DIM_NOPE + _DIM_ROPE,
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    input_nope_q = gathered[:, :_DIM_NOPE]
    input_nope_s = gathered[:, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(
        torch.float32
    )
    input_rope = gathered[:, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)
    output[:, _DIM_NOPE:] = input_rope

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        cur_nope = input_nope_q[:, lo:hi].to(torch.float32)
        cur_scale = input_nope_s[:, tile_idx].unsqueeze(-1)
        output[:, lo:hi] = cur_nope * cur_scale

    return output.unsqueeze(1)
