# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""qwen4_grouped_gemma_rmsnorm; adapted from FlagGems-vllm PR #455 dependencies.

Source: flagos-ai/FlagGems-vllm@f775b37f3066e2935742d26c7592a5dd55499159.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HAS_TRITON = True


def can_use_hc_triton(*tensors: torch.Tensor) -> bool:
    """Return whether the specialized contiguous inference path is usable."""

    return bool(
        HAS_TRITON
        and tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    group_row = tl.program_id(0)
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    input_base = group_row * hidden_size
    weight_base = (group_row % hc_count) * hidden_size
    values = tl.load(
        input_ptr + input_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + weight_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    tl.store(
        output_ptr + input_base + offsets,
        values * inv_rms * (1.0 + weight),
        mask=mask,
    )


def qwen4_grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    if hidden_states.ndim < 2 or hidden_states.shape[-1] != weight.numel():
        raise ValueError(
            "Qwen4 HC RMSNorm received incompatible input and weight"
        )
    if hc_count <= 0 or weight.numel() % hc_count or eps <= 0:
        raise ValueError("Qwen4 HC RMSNorm requires a valid HC count")
    if not can_use_hc_triton(hidden_states, weight):
        raise RuntimeError(
            "Qwen4 HC RMSNorm requires contiguous accelerator tensors"
        )
    output = torch.empty_like(hidden_states)
    if not hidden_states.numel():
        return output
    hidden_size = weight.numel() // hc_count
    block_h = triton.next_power_of_2(hidden_size)
    rows = hidden_states.numel() // weight.numel()
    _grouped_gemma_rmsnorm_kernel[(rows * hc_count,)](
        hidden_states,
        weight,
        output,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["qwen4_grouped_gemma_rmsnorm"]
