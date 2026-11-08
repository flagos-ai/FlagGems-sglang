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

"""Reference for kv_cache/dequantize_k_cache_fast.

A packed DSA KV-cache row holds three segments back to back:

    [dim_nope fp8 bytes | num_tiles fp32 scales | dim_rope bf16 elements]

with ``num_tiles = dim_nope // group_size`` and therefore a row width of
``dim_nope + num_tiles * 4 + dim_rope * 2`` bytes. The NoPE segment is
dequantized with one fp32 scale per ``group_size`` columns; the RoPE
segment is stored uncompressed and is copied through.
"""

import torch


def reference(quant_k_cache, dim_nope, dim_rope, group_size=128):
    # A 3-D input ([blocks, block_size, dim_quant]) is flattened to rows and
    # the result is reshaped back, mirroring the op's own handling.
    shape = quant_k_cache.shape
    dim_quant = shape[-1]
    rows = quant_k_cache.reshape(-1, dim_quant)
    num_tiles = dim_nope // group_size
    assert dim_quant == dim_nope + num_tiles * 4 + dim_rope * 2, (
        f"packed row width {dim_quant} does not match dim_nope={dim_nope}, "
        f"dim_rope={dim_rope}, group_size={group_size}"
    )

    nope_q = rows[:, :dim_nope].to(torch.float32)
    nope_s = rows[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope = rows[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    # Expand along the scale axis (the last one), not the token axis: with a
    # 3-D input ``dim=1`` would be the block axis instead.
    scales = nope_s.repeat_interleave(group_size, dim=-1)
    nope = (nope_q * scales).to(torch.bfloat16)
    out = torch.cat([nope, rope], dim=1)
    return out.view(*shape[:-1], dim_nope + dim_rope)
