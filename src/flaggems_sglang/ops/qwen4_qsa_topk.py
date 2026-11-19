# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""qwen4_qsa_topk; SGLang layout and rounding contract.

Extracted from the validated Qwen3.8 SGLang baseline; this is not a vLLM kernel copy.
"""

import torch


def qwen4_qsa_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Vectorized fallback for arbitrary row lengths; no per-row device sync."""
    rows, cols = logits.shape
    starts = row_starts.to(device=logits.device, dtype=torch.long).reshape(
        -1, 1
    )
    ends = row_ends.to(device=logits.device, dtype=torch.long).reshape(-1, 1)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    k = min(topk, cols)
    if not rows or not k:
        return out
    columns = torch.arange(cols, device=logits.device).reshape(1, -1)
    masked = torch.where(
        (columns >= starts) & (columns < ends), logits, -float("inf")
    )
    indices = torch.topk(masked, k, dim=1).indices
    valid = (indices >= starts) & (indices < ends)
    out[:, :k] = torch.where(valid, indices - starts, -1).to(torch.int32)
    return out


__all__ = ["qwen4_qsa_topk"]
