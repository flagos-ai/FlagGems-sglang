# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def reference(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
    mrope_interleaved_glm,
    is_neox_style,
    axis_map,
):
    """Original in-place, None-returning ABI for the documented workload.

    Unsupported modes fail explicitly instead of computing another RoPE.
    Arithmetic uses FP32 intermediates and rounds once to the input dtype.
    """
    if (
        not mrope_interleaved
        or mrope_interleaved_glm
        or not is_neox_style
        or axis_map is not None
    ):
        raise NotImplementedError(
            "mrotary_embedding Torch reference requires THW interleaved Neox "
            "without axis_map"
        )
    if (
        head_size != 256
        or rotary_dim != 64
        or list(mrope_section) != [11, 11, 10]
    ):
        raise NotImplementedError(
            "mrotary_embedding Torch reference supports "
            "head256/rotary64/sections11,11,10"
        )
    half = rotary_dim // 2
    idx = torch.arange(half, device=q.device)
    h_mask = (idx % 3 == 1) & (idx <= 3 * mrope_section[1])
    w_mask = (idx % 3 == 2) & (idx <= 3 * mrope_section[2])
    axes = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    row = positions.index_select(0, axes).transpose(0, 1).long()
    cos = cos_sin_cache[row, idx].float().unsqueeze(1)
    sin = cos_sin_cache[row, idx + half].float().unsqueeze(1)
    for value in (q, k):
        heads = value.view(value.shape[0], -1, head_size)
        # Read both halves before writing; leave trailing channels untouched.
        x1 = heads[..., :half].float()
        x2 = heads[..., half:rotary_dim].float()
        first = x1 * cos - x2 * sin
        second = x2 * cos + x1 * sin
        heads[..., :half].copy_(first)
        heads[..., half:rotary_dim].copy_(second)
    return None
