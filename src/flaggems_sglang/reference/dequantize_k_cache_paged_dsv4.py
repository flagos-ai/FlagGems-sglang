# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""PyTorch reference for ``dequantize_k_cache_paged_dsv4``.

The DeepSeek-V4 paged layout splits data and scales at page granularity,
unlike the DSA layout which packs them inside a single row::

    page = [token 0 .. P-1: 448 fp8 nope + 64 bf16 rope = 576 bytes each]
           [token 0 .. P-1: 7 ue8m0 scales + 1 pad byte = 8 bytes each]
           padded up to a multiple of 576 bytes

so a token's scales are ``P * 576`` bytes away from its data. The ue8m0
scale byte decodes as ``2 ** (byte - 127)``.
"""

import torch

DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1


def _ue8m0_to_fp32(scale_u8):
    """Decode a ue8m0 byte as ``2 ** (byte - 127)``.

    Byte 0 therefore lands on the denormal ``2 ** -127``, matching the
    kernel's ``exp2(byte - 127)``. Byte 255 (NaN in the float8_e8m0fnu
    encoding) is left to the same formula, so the reference and the kernel
    agree bit for bit; callers should keep scales in ``1..254`` when they
    want a defined value.
    """
    return torch.exp2((scale_u8.to(torch.float32) - 127.0))


def reference(quant_k_cache, page_table_1_flattened, page_size):
    """Dequantize paged DeepSeek-V4 k-cache tokens.

    Args:
        quant_k_cache: page-major cache, any dtype whose underlying storage
            is uint8 (often bf16). Shape ``(num_pages, bytes_per_page)``.
        page_table_1_flattened: 1-D integer tensor of token ids, i.e. the
            value indexes the flattened token space of the cache.
        page_size: tokens per page.

    Returns:
        ``(num_tokens, 1, DIM_NOPE + DIM_ROPE)`` bfloat16.
    """
    u8 = quant_k_cache.view(torch.uint8).contiguous()
    num_tokens = page_table_1_flattened.shape[0]
    flat = u8.reshape(-1)
    bytes_per_page = u8.shape[-1]
    s_offset = page_size * NOPE_ROPE_BYTES

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_base = page_idx * bytes_per_page
    data_base = page_base + in_page * NOPE_ROPE_BYTES
    scale_base = page_base + s_offset + in_page * PADDED_SCALE_PER_TOKEN

    nope_off = torch.arange(DIM_NOPE, device=u8.device)
    nope_u8 = flat[data_base[:, None] + nope_off[None, :]]
    nope_fp8 = nope_u8.view(torch.float8_e4m3fn)

    tile_off = torch.arange(NUM_SCALE_TILES, device=u8.device)
    scale_u8 = flat[scale_base[:, None] + tile_off[None, :]]
    scales = _ue8m0_to_fp32(scale_u8).repeat_interleave(TILE_SIZE, dim=-1)

    nope = (nope_fp8.to(torch.float32) * scales).to(torch.bfloat16)

    # The rope tail is stored as bf16 elements, so byte offsets halve.
    bf16_flat = flat.view(torch.bfloat16)
    rope_base = (data_base + DIM_NOPE) // 2
    rope_off = torch.arange(DIM_ROPE, device=u8.device)
    rope = bf16_flat[rope_base[:, None] + rope_off[None, :]]

    out = torch.cat([nope, rope], dim=-1)
    return out.view(num_tokens, 1, DIM_NOPE + DIM_ROPE)
