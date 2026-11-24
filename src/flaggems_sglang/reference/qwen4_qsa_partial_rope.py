# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(x, positions, cache, axis_map, rotary_dim, is_neox_style=True):
    half = rotary_dim // 2
    frequency = torch.arange(half, device=x.device)
    pos = (
        positions[:, None].expand(-1, half)
        if positions.ndim == 1
        else positions[axis_map.long(), :].T
    )
    cosine = cache[pos, frequency].to(x.dtype)[:, None]
    sine = cache[pos, frequency + half].to(x.dtype)[:, None]
    result = x.clone()
    if is_neox_style:
        left, right = x[..., :half], x[..., half:rotary_dim]
        result[..., :half] = left * cosine - right * sine
        result[..., half:rotary_dim] = right * cosine + left * sine
    else:
        left, right = x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
        result[..., :rotary_dim:2] = left * cosine - right * sine
        result[..., 1:rotary_dim:2] = right * cosine + left * sine
    return result
