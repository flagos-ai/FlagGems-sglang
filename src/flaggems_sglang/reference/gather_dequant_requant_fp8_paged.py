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

"""Reference implementation for dsa/gather_dequant_requant_fp8_paged."""

import torch

_DIM_NOPE = 512
_DIM_ROPE = 64


def reference(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    num_tokens = page_table_1_flattened.shape[0]
    num_tiles = _DIM_NOPE // group_size
    total_rows = num_tokens + extra_rows

    idx = page_table_1_flattened.to(torch.int64)
    # 656 bytes per token: [512 nope fp8 | 4 fp32 scales | 64 bf16 rope].
    nope_q = cache[idx, :_DIM_NOPE].to(torch.float32)
    scales = cache[idx, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(
        torch.float32
    )
    rope = cache[idx, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)

    # Dequant per group, then requant to per-tensor fp8 with scale 1.0.
    nope = nope_q * scales.repeat_interleave(group_size, dim=-1)

    output = torch.zeros(
        (total_rows, 1, _DIM_NOPE + _DIM_ROPE),
        dtype=torch.float8_e4m3fn,
        device=cache.device,
    )
    output[:num_tokens, 0, :_DIM_NOPE] = nope.to(torch.float8_e4m3fn)
    output[:num_tokens, 0, _DIM_NOPE:] = rope.to(torch.float8_e4m3fn)
    # Rows [num_tokens, total_rows) stay zero: the SM90 kernel clamps every
    # -1 topk slot into that landing pad and reads it.
    return output
