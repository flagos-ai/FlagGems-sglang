# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""qwen4_qsa_all_blocks; SGLang layout and rounding contract.

Extracted from the validated Qwen3.8 SGLang baseline; this is not a vLLM kernel copy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _all_blocks(lengths, out, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    n = tl.load(lengths + row)
    tl.store(out + row * K + col, tl.where(col < n, col, -1), col < K)


def qwen4_qsa_all_blocks(lengths: torch.Tensor, topk: int) -> torch.Tensor:
    """Caller guarantees every row fits topk using a host-side length bound."""
    if topk <= 0 or lengths.ndim != 1:
        raise ValueError("Expected positive topk and 1-D compressed lengths")
    lengths = lengths.to(dtype=torch.int32).contiguous()
    out = torch.empty(
        (lengths.numel(), topk), dtype=torch.int32, device=lengths.device
    )
    if lengths.numel():
        _all_blocks[(lengths.numel(),)](
            lengths, out, topk, triton.next_power_of_2(topk)
        )
    return out


__all__ = ["qwen4_qsa_all_blocks"]
