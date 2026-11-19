# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""qwen4_hc_inject_combine; adapted from FlagGems-vllm PR #455 dependencies.

Source: flagos-ai/FlagGems-vllm@f775b37f3066e2935742d26c7592a5dd55499159.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HAS_TRITON = True


def _has_flat_row_layout(tensor: torch.Tensor) -> bool:
    """Return whether leading dims can be flattened using ``stride(-2)``.

    Packed HC projection output is split into ``[..., lowrank]`` and
    ``[..., hc_count]`` views.  The latter is contiguous within each row but
    retains the packed projection's wider row stride.  The injection kernel
    accepts that layout explicitly, while still rejecting transposes and
    layouts whose leading dimensions cannot be flattened into rows.
    """

    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    if tensor.stride(-2) < tensor.shape[-1]:
        return False
    return all(
        tensor.stride(dim) == tensor.shape[dim + 1] * tensor.stride(dim + 1)
        for dim in range(tensor.ndim - 2)
    )


def can_use_hc_inject_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    """Allow packed-projection logits with a wider row stride.

    ``_hc_inject_combine_kernel`` consumes ``stride_logits_row`` and only
    requires the branch dimension to be contiguous.  The other two inputs
    retain the stricter contiguous inference contract.
    """

    tensors = (injection_logits, block_output, residual)
    return bool(
        HAS_TRITON
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            for tensor in tensors
        )
        and len({tensor.device for tensor in tensors}) == 1
        and _has_flat_row_layout(injection_logits)
        and block_output.is_contiguous()
        and residual.is_contiguous()
    )


@triton.jit
def _hc_inject_combine_kernel(
    injection_logits_ptr,
    block_output_ptr,
    residual_ptr,
    output_ptr,
    stride_logits_row,
    stride_block_row,
    stride_residual_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    branch = tl.program_id(1)
    offsets = tl.program_id(2) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    logits = tl.load(
        injection_logits_ptr + row * stride_logits_row + branch
    ).to(tl.float32)
    dtype: tl.constexpr = residual_ptr.dtype.element_ty
    scaled = (logits / hc_count).to(dtype).to(tl.float32)
    gate = tl.sigmoid(scaled).to(dtype).to(tl.float32)
    injection_weight = (2.0 * gate).to(dtype).to(tl.float32)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        residual + (block_output * injection_weight).to(dtype).to(tl.float32),
        mask=mask,
    )


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if injection_logits.shape != (*block_output.shape[:-1], hc_count):
        raise ValueError("Qwen4 HC injection logits have an invalid shape")
    if hc_count <= 0:
        raise ValueError("Qwen4 HC injection requires a positive HC count")
    if residual.shape != (
        *block_output.shape[:-1],
        hc_count * block_output.shape[-1],
    ):
        raise ValueError(
            "Qwen4 HC residual and block output shapes are incompatible"
        )
    if not can_use_hc_inject_triton(injection_logits, block_output, residual):
        raise RuntimeError(
            "Qwen4 HC injection received an unsupported accelerator layout"
        )
    output = torch.empty_like(residual)
    if not residual.numel():
        return output
    hidden_size = block_output.shape[-1]
    rows = block_output.numel() // hidden_size
    block_h = 256
    _hc_inject_combine_kernel[
        (rows, hc_count, triton.cdiv(hidden_size, block_h))
    ](
        injection_logits,
        block_output,
        residual,
        output,
        injection_logits.stride(-2),
        block_output.stride(-2),
        residual.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["qwen4_hc_inject_combine"]
