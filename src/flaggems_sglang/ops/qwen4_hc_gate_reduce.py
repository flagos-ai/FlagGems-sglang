# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""qwen4_hc_gate_reduce; adapted from FlagGems-vllm PR #455 dependencies.

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
def _hc_gate_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    stride_logits_row,
    stride_normed_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    dtype: tl.constexpr = normed_ptr.dtype.element_ty
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for branch in tl.static_range(0, hc_count):
        branch_offsets = branch * hidden_size + offsets
        logits = tl.load(
            logits_ptr + row * stride_logits_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normed = tl.load(
            normed_ptr + row * stride_normed_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(logits).to(dtype).to(tl.float32)
        product = (gate * normed).to(dtype).to(tl.float32)
        accumulator += product
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        accumulator / hc_count,
        mask=mask,
    )


def qwen4_hc_gate_reduce(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if logits.shape != normed.shape or logits.ndim < 2:
        raise ValueError(
            "Qwen4 HC gate logits and normalized input must match"
        )
    if hc_count <= 0 or logits.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC gate reduction requires a valid HC count")
    if not can_use_hc_triton(logits, normed):
        raise RuntimeError(
            "Qwen4 HC gate reduction requires contiguous tensors"
        )
    hidden_size = logits.shape[-1] // hc_count
    output = torch.empty(
        (*logits.shape[:-1], hidden_size),
        dtype=normed.dtype,
        device=normed.device,
    )
    if not logits.numel():
        return output
    rows = logits.numel() // logits.shape[-1]
    block_h = 256
    _hc_gate_reduce_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        logits,
        normed,
        output,
        logits.stride(-2),
        normed.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["qwen4_hc_gate_reduce"]
