# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import math

import torch


def reference(
    q,
    cache,
    table,
    token_to_req,
    positions,
    lengths,
    compress_ratio,
    num_columns=None,
    score_scale=None,
):
    columns = (
        table.shape[1] * cache.shape[1] if num_columns is None else num_columns
    )
    req = token_to_req.to(torch.int64)
    valid_req = (req >= 0) & (req < table.shape[0])
    safe_req = req.clamp(0, table.shape[0] - 1)
    seq = lengths.index_select(0, safe_req)
    seq = torch.where(valid_req, seq, torch.zeros_like(seq))
    visible = torch.minimum(
        (positions.to(torch.int64) + 1) // compress_ratio,
        seq // compress_ratio,
    )
    col = torch.arange(columns, device=q.device, dtype=torch.int64)
    logical_page = col // cache.shape[1]
    offset = col % cache.shape[1]
    safe_logical = logical_page.clamp(0, table.shape[1] - 1)
    physical = table[safe_req[:, None], safe_logical[None, :]].to(torch.int64)
    valid = (
        valid_req[:, None]
        & (col[None, :] < visible[:, None])
        & (physical >= 0)
        & (physical < cache.shape[0])
    )
    safe_physical = physical.clamp(0, cache.shape[0] - 1)
    keys = cache[safe_physical, offset[None, :], 0].float()
    dots = torch.matmul(q.float(), keys.transpose(1, 2))
    logits = torch.relu(dots).sum(1) / (
        math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    )
    logits = logits.masked_fill(~valid, -float("inf"))
    return logits, visible.to(torch.int32)
