# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    softmax_scale=None,
    out=None,
    gate=None,
    split_workspace=None,
):
    rows, heads, width = q.shape
    positions = logical_indices.long()
    request = token_to_req.long()
    valid_req = (request >= 0) & (request < block_table.shape[0])
    request = request.clamp(0, block_table.shape[0] - 1)
    logical_page = positions.div(k_cache.shape[1], rounding_mode="floor")
    valid = (
        valid_req[:, None]
        & (positions >= 0)
        & (logical_page < block_table.shape[1])
    )
    physical = block_table[
        request[:, None], logical_page.clamp(0, block_table.shape[1] - 1)
    ].long()
    valid &= (physical >= 0) & (physical < k_cache.shape[0])
    physical = physical.clamp(0, k_cache.shape[0] - 1)
    offset = positions.remainder(k_cache.shape[1])
    keys = (
        k_cache[physical, offset]
        .float()
        .repeat_interleave(heads // k_cache.shape[2], dim=2)
    )
    values = (
        v_cache[physical, offset]
        .float()
        .repeat_interleave(heads // v_cache.shape[2], dim=2)
    )
    scale = width**-0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("nhd,nkhd->nhk", q.float(), keys) * scale
    scores = scores.masked_fill(~valid[:, None], -torch.inf)
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    result = torch.einsum("nhk,nkhd->nhd", probs, values).to(q.dtype)
    if gate is not None:
        result = result * torch.sigmoid(gate)
    if out is not None:
        out.copy_(result)
        return out
    return result
