# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""qwen4_hc_combine_norm; adapted from FlagGems-vllm PR #455 dependencies.

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


def can_use_hc_combine_norm_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    """Return whether delayed combine+grouped-RMSNorm can use Triton.

    The injection view may retain the wider row stride of the packed
    down+inject projection.  The residual and block output are required to
    remain contiguous because the fused kernel writes a fresh contiguous
    multi-stream state and normalizes each branch in place conceptually.
    """

    return bool(
        can_use_hc_inject_triton(injection_logits, block_output, residual)
        and norm_weight.device == residual.device
        and norm_weight.dtype in (torch.bfloat16, torch.float16)
        and norm_weight.is_contiguous()
    )


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_block_row,
    stride_residual_row,
    stride_injection_row,
    stride_output_row,
    stride_normed_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    weight_shared: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
    num_tiles_padded: tl.constexpr,
) -> None:
    """Fuse pending HC injection, materialization, and grouped RMSNorm.

    One program owns one token/branch pair.  All hidden-size tiles are kept
    in the program so the RMS reduction sees the same rounded-to-residual-dtype
    values that the unfused combine -> norm sequence would consume.
    """

    row = tl.program_id(0)
    branch = tl.program_id(1)
    tile_ids = tl.arange(0, num_tiles_padded)
    offsets = tile_ids[:, None] * block_h + tl.arange(0, block_h)[None, :]
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    weight_offsets = offsets if weight_shared else branch_offsets

    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block = tl.load(
        block_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    injection = tl.load(
        injection_ptr + row * stride_injection_row + branch,
    ).to(tl.float32)
    dtype: tl.constexpr = residual_ptr.dtype.element_ty
    scaled = (injection / hc_count).to(dtype).to(tl.float32)
    gate = tl.sigmoid(scaled).to(dtype).to(tl.float32)
    injection = (2.0 * gate).to(dtype).to(tl.float32)

    # Preserve both eager multiplication and residual-add rounding points.
    delta = (block * injection).to(dtype).to(tl.float32)
    combined = (residual + delta).to(dtype)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        combined,
        mask=mask,
    )

    combined_f32 = combined.to(tl.float32)
    sum_sq = tl.sum(tl.sum(combined_f32 * combined_f32, axis=1), axis=0)
    inv_rms = tl.rsqrt(sum_sq / hidden_size + eps)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    normalized = combined_f32 * inv_rms
    normalized = normalized * (1.0 + weight.to(tl.float32))
    tl.store(
        normed_ptr + row * stride_normed_row + branch_offsets,
        normalized,
        mask=mask,
    )


def qwen4_hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused delayed HC combine + grouped Gemma RMSNorm."""

    if (
        residual.ndim != 2
        or block_output.ndim != 2
        or injection_logits.ndim != 2
    ):
        raise ValueError(
            "Qwen4 HC combine_norm currently requires 2-D tensors"
        )
    if hc_count <= 0 or residual.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC combine_norm requires a valid HC count")
    hidden_size = residual.shape[-1] // hc_count
    if block_output.shape != (residual.shape[0], hidden_size):
        raise ValueError("Qwen4 HC combine_norm block output shape is invalid")
    if injection_logits.shape != (residual.shape[0], hc_count):
        raise ValueError("Qwen4 HC combine_norm injection shape is invalid")
    if norm_weight.numel() not in (hidden_size, residual.shape[-1]):
        raise ValueError("Qwen4 HC combine_norm weight shape is invalid")
    if not can_use_hc_combine_norm_triton(
        injection_logits, block_output, residual, norm_weight
    ):
        raise NotImplementedError(
            "Qwen4 HC combine_norm requires contiguous accelerator tensors"
        )

    if eps <= 0:
        raise ValueError("HC normalization epsilon must be positive")
    output = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    if not residual.numel():
        return output, normalized

    block_h = 512
    rows = residual.shape[0]
    _hc_combine_norm_kernel[(rows, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        output,
        normalized,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        output.stride(0),
        normalized.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        weight_shared=norm_weight.numel() == hidden_size,
        eps=eps,
        block_h=block_h,
        num_tiles_padded=triton.next_power_of_2(
            triton.cdiv(hidden_size, block_h)
        ),
        num_warps=8 if hidden_size > 2048 else 4,
        enable_fp_fusion=False,
    )
    return output, normalized


__all__ = ["qwen4_hc_combine_norm"]
