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

"""Reference for kv_cache/dequantize_k_cache_paged.

Same packed layout as ``dequantize_k_cache_fast``, but the source rows are
addressed indirectly through ``page_table_1_flattened``. ``num_tokens`` may
exceed the number of cache rows because several sequences can share KV
slots through prefix caching, so the page table is not assumed injective.
"""

import torch


def _infer_dims(dim_quant, group_size=128):
    """Recover ``(dim_nope, dim_rope)`` from a packed row width."""
    for dim_rope in (0, 64):
        scaled_width = (dim_quant - 2 * dim_rope) * group_size
        if scaled_width % (group_size + 4):
            continue
        dim_nope = scaled_width // (group_size + 4)
        if dim_nope > 0 and dim_nope % group_size == 0:
            return dim_nope, dim_rope
    raise ValueError(f"Unsupported packed DSA KV row width: {dim_quant}")


def reference(quant_k_cache, page_table_1_flattened, group_size=128):
    dim_quant = quant_k_cache.shape[-1]
    rows = quant_k_cache.view(-1, dim_quant)
    dim_nope, dim_rope = _infer_dims(dim_quant, group_size)
    num_tiles = dim_nope // group_size
    num_tokens = page_table_1_flattened.shape[0]
    index = page_table_1_flattened.to(torch.long)

    nope_q = rows[:, :dim_nope].to(torch.float32).index_select(0, index)
    nope_s = (
        rows[:, dim_nope : dim_nope + num_tiles * 4]
        .view(torch.float32)
        .index_select(0, index)
    )
    rope = (
        rows[:, dim_nope + num_tiles * 4 :]
        .view(torch.bfloat16)
        .index_select(0, index)
    )

    # Expand along the scale axis (the last one): after ``index_select`` the
    # leading axis is tokens and the trailing axis holds the per-tile scales.
    scales = nope_s.repeat_interleave(group_size, dim=-1)
    nope = (nope_q * scales).to(torch.bfloat16)
    return torch.cat([nope, rope], dim=1).view(num_tokens, 1, -1)
