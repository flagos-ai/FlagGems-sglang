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

"""Dequantize paged DeepSeek-V4 KV-cache tokens back to bf16.

Unlike the DSA layout, which packs ``[nope | scales | rope]`` inside one row,
the V4 layout separates scales at page granularity::

    page = [token 0 .. P-1 data: 448 fp8 nope + 64 bf16 rope = 576 B each]
           [token 0 .. P-1 scales: 7 ue8m0 bytes + 1 pad = 8 B each]
           padded up to a multiple of 576 bytes

One ue8m0 byte covers 64 nope columns, so the 448 NoPE width needs exactly
7 of them. The kernel keeps the scale decode in the exponent domain
(``2 ** (byte - 127)``) and issues a single ``[T, 8, 64]`` load per program,
because the dequant itself is elementwise and only the grid shape changes.
"""

import torch
import triton
import triton.language as tl

DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1


@triton.jit
def _dequantize_k_cache_paged_dsv4_kernel(
    out_ptr,
    q_ptr,
    r_ptr,
    s_ptr,
    page_ptr,
    out_s0,
    num_tokens,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    token_ids = tl.program_id(0).to(tl.int64) * TOKENS_PER_PROG + tl.arange(
        0, TOKENS_PER_PROG
    ).to(tl.int64)
    tmask = token_ids < num_tokens

    loc = tl.load(page_ptr + token_ids, mask=tmask, other=0).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_base = page_idx * BYTES_PER_PAGE
    data_base = page_base + in_page * NOPE_ROPE_BYTES
    scale_base = page_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    out_row = token_ids * out_s0

    # tl.arange needs a power of two, so walk the 8-byte padded scale row and
    # mask the pad slot off (NUM_SCALE_TILES == 7).
    tiles = tl.arange(0, PADDED_SCALE_PER_TOKEN)
    inner = tl.arange(0, TILE_SIZE)
    live = tmask[:, None, None] & (tiles < NUM_SCALE_TILES)[None, :, None]

    q_off = (
        data_base[:, None, None]
        + tiles[None, :, None] * TILE_SIZE
        + inner[None, None, :]
    )
    q = tl.load(q_ptr + q_off, mask=live, other=0.0).to(tl.float32)

    s_u8 = tl.load(
        s_ptr + scale_base[:, None] + tiles[None, :],
        mask=tmask[:, None],
        other=0,
    ).to(tl.int32)
    scale = tl.exp2((s_u8 - 127).to(tl.float32))

    o_off = (
        out_row[:, None, None]
        + tiles[None, :, None] * TILE_SIZE
        + inner[None, None, :]
    )
    tl.store(
        out_ptr + o_off,
        (q * scale[:, :, None]).to(out_ptr.dtype.element_ty),
        mask=live,
    )

    # The rope tail is bf16, so byte offsets halve when indexing it.
    rope_off = tl.arange(0, DIM_ROPE)
    bf16_base = (data_base + DIM_NOPE) // 2
    rope = tl.load(
        r_ptr + bf16_base[:, None] + rope_off[None, :],
        mask=tmask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + out_row[:, None] + DIM_NOPE + rope_off[None, :],
        rope,
        mask=tmask[:, None],
    )


def dequantize_k_cache_paged_dsv4(
    quant_k_cache, page_table_1_flattened, page_size
):
    """Dequantize paged DeepSeek-V4 k-cache tokens.

    Args:
        quant_k_cache: page-major cache whose underlying storage is uint8.
            Shape ``(num_pages, bytes_per_page)``; the dtype of the pool is
            irrelevant and is reinterpreted as bytes.
        page_table_1_flattened: 1-D integer tensor of token ids into the
            flattened token space of the cache. It is not assumed to be
            injective: prefix caching lets several tokens share one slot.
        page_size: tokens per page.

    Returns:
        ``(num_tokens, 1, DIM_NOPE + DIM_ROPE)`` bfloat16.
    """
    if page_table_1_flattened.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "page_table_1_flattened must be int32 or int64, got "
            f"{page_table_1_flattened.dtype}"
        )

    u8 = quant_k_cache.view(torch.uint8)
    if not u8.is_contiguous():
        raise ValueError("quant_k_cache must be contiguous")

    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = u8.shape[-1]
    required = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    if bytes_per_page < required:
        raise ValueError(
            f"bytes_per_page ({bytes_per_page}) cannot hold {page_size} "
            f"tokens of the V4 layout (needs {required})"
        )
    if bytes_per_page % 2:
        raise ValueError(
            "bytes_per_page must be even to view the rope tail as bfloat16, "
            f"got {bytes_per_page}"
        )

    out = torch.empty(
        (num_tokens, 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if num_tokens == 0:
        return out

    flat_u8 = u8.reshape(-1)
    tokens_per_prog = 4
    _dequantize_k_cache_paged_dsv4_kernel[
        (triton.cdiv(num_tokens, tokens_per_prog),)
    ](
        out,
        flat_u8.view(torch.float8_e4m3fn),
        flat_u8.view(torch.bfloat16),
        flat_u8,
        page_table_1_flattened,
        out.stride(0),
        num_tokens,
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=page_size * NOPE_ROPE_BYTES,
        TOKENS_PER_PROG=tokens_per_prog,
    )
    return out


__all__ = ["dequantize_k_cache_paged_dsv4"]
