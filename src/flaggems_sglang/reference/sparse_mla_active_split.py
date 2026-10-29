# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Literal selected-key attention for benchmarks; independent of Triton."""

import torch


def reference(q, kv, indices, sm_scale, active_heads, sink, lengths):
    rows = q.shape[0]
    compute_dtype = (
        torch.float64 if q.dtype == torch.float64 else torch.float32
    )
    slots = indices[:, 0].long()
    valid = (
        (slots >= 0)
        & (slots < kv.shape[0])
        & (torch.arange(2048, device=q.device)[None, :] < lengths[:, None])
    )
    selected = kv[:, 0][slots.clamp(0, kv.shape[0] - 1)].to(compute_dtype)
    scores = (
        torch.bmm(
            q[:, :active_heads].to(compute_dtype), selected.transpose(1, 2)
        )
        * sm_scale
    )
    scores = scores.masked_fill(~valid[:, None], -torch.inf)
    maximum = scores.max(dim=-1).values
    lse = torch.logsumexp(scores, dim=-1)
    denominator = torch.logaddexp(lse, sink[:active_heads][None])
    probability = torch.exp(scores - denominator[:, :, None])
    output = torch.bmm(probability, selected[:, :, :512])
    have_keys = valid.any(dim=-1)[:, None]
    output = torch.where(have_keys[:, :, None], output, 0.0)
    output = torch.where(
        torch.isposinf(sink[:active_heads])[None, :, None], 0.0, output
    )
    lse = torch.where(have_keys.expand(rows, active_heads), lse, torch.inf)
    return output.to(q.dtype), maximum, lse
