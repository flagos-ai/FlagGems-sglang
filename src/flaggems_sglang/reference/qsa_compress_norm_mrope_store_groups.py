# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(
    raw_cache,
    raw_block_table,
    token_to_req,
    logical_positions,
    compressed_slots,
    compressed_cache,
    norm_weight,
    cos_sin_cache,
    compress_ratio,
    norm_eps,
    rotary_dim,
    mrope_section,
    mrope_interleaved,
    rope_cache=None,
):
    half = rotary_dim // 2
    frequencies = torch.arange(half, device=raw_cache.device)
    section_t, section_h, section_w = mrope_section
    for row in range(token_to_req.numel()):
        request = int(token_to_req[row])
        end = int(logical_positions[row])
        slot = int(compressed_slots[row])
        if not (
            0 <= request < raw_block_table.shape[0]
            and end >= compress_ratio - 1
            and 0
            <= slot
            < compressed_cache.shape[0] * compressed_cache.shape[1]
        ):
            continue
        members = []
        for position in range(end - compress_ratio + 1, end + 1):
            logical_page, offset = divmod(position, raw_cache.shape[1])
            page = (
                int(raw_block_table[request, logical_page])
                if logical_page < raw_block_table.shape[1]
                else -1
            )
            members.append(
                raw_cache[page, offset, 0].float()
                if 0 <= page < raw_cache.shape[0]
                else torch.zeros_like(norm_weight, dtype=torch.float32)
            )
        pooled = (
            torch.stack(members).sum(0).div(compress_ratio).to(raw_cache.dtype)
        )
        values = pooled.float()
        normed = (
            values
            * torch.rsqrt(values.square().mean() + norm_eps)
            * (1 + norm_weight.float())
        ).to(raw_cache.dtype)
        first = end - compress_ratio + 1
        if rope_cache is None:
            axes = torch.full(
                (3,), first, dtype=torch.int64, device=raw_cache.device
            )
        else:
            logical_page, offset = divmod(first, rope_cache.shape[1])
            page = int(raw_block_table[request, logical_page])
            axes = rope_cache[page, offset, 0]
        if mrope_interleaved:
            use_h = (frequencies % 3 == 1) & (frequencies < section_h * 3)
            use_w = (frequencies % 3 == 2) & (frequencies < section_w * 3)
        else:
            use_h = (frequencies >= section_t) & (
                frequencies < section_t + section_h
            )
            use_w = (frequencies >= section_t + section_h) & (
                frequencies < sum(mrope_section)
            )
        positions = torch.where(
            use_h, axes[1], torch.where(use_w, axes[2], axes[0])
        )
        cos = cos_sin_cache[positions, frequencies].to(raw_cache.dtype)
        sin = cos_sin_cache[positions, half + frequencies].to(raw_cache.dtype)
        left, right = normed[:half], normed[half:rotary_dim]
        rotated = torch.cat(
            (
                (left * cos - right * sin).to(raw_cache.dtype),
                (right * cos + left * sin).to(raw_cache.dtype),
                normed[rotary_dim:],
            )
        )
        page, offset = divmod(slot, compressed_cache.shape[1])
        compressed_cache[page, offset, 0].copy_(rotated)
