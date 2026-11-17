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

"""Reference implementation for dsa/quantize_k_cache_fast."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn


def reference(k_nope, k_rope, group_size=128):
    num_tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    num_tiles = dim_nope // group_size
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    output = torch.empty(
        num_tokens,
        dim_nope + num_tiles * 4 + k_rope.element_size() * dim_rope,
        dtype=_FP8_DTYPE,
        device=k_nope.device,
    )
    output_nope_q = output[:, :dim_nope]
    output_nope_s = output[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    output_rope = output[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    output_rope[:] = k_rope

    for tile_idx in range(num_tiles):
        lo = tile_idx * group_size
        hi = lo + group_size
        tile = k_nope[:, lo:hi].float()
        # No eps floor here: the fast kernel deliberately omits the one the
        # generic _quantize_k_cache_ref applies.
        scale = tile.abs().amax(dim=-1) / fp8_max
        output_nope_s[:, tile_idx] = scale
        quant = (
            (tile / scale.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)
        )
        output_nope_q[:, lo:hi] = quant

    return output
