# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""ple_gate_norm; adapted from FlagGems-vllm PR #455 dependencies.

Source: flagos-ai/FlagGems-vllm@f775b37f3066e2935742d26c7592a5dd55499159.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

HAS_TRITON = True


@triton.jit
def _ple_gate_norm_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    key_weight_ptr,
    query_weight_ptr,
    conv_weight_ptr,
    gated_ptr,
    normalized_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    inv_sqrt_hidden: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    """Fuse three grouped RMS norms and the complete PLE gate chain.

    A program owns one ``(token, HC branch)`` row.  Reductions accumulate in
    FP32, while explicit casts preserve the reference path's low-precision
    materialization points (normalized key/query, gate, and gated value).
    """

    group_row = tl.program_id(0)
    token = group_row // hc_count
    branch = group_row % hc_count
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    row_offsets = group_row * hidden_size + offsets
    value_offsets = token * hidden_size + offsets
    weight_offsets = branch * hidden_size + offsets

    key = tl.load(key_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    query = tl.load(query_ptr + row_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    key_weight = tl.load(
        key_weight_ptr + weight_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    query_weight = tl.load(
        query_weight_ptr + weight_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    key_inv_rms = tl.rsqrt(tl.sum(key * key, axis=0) / hidden_size + eps)
    query_inv_rms = tl.rsqrt(tl.sum(query * query, axis=0) / hidden_size + eps)

    # The eager reference stores both norm results in the source dtype before
    # multiplying them.  Preserve those rounding points instead of feeding an
    # all-FP32 dot product into the nonlinear gate.
    source_dtype: tl.constexpr = key_ptr.dtype.element_ty
    norm_key = (key * key_inv_rms * (1.0 + key_weight)).to(source_dtype)
    norm_query = (query * query_inv_rms * (1.0 + query_weight)).to(
        source_dtype
    )
    products = (norm_key * norm_query).to(source_dtype)
    gate_linear = (
        tl.sum(products.to(tl.float32), axis=0) * inv_sqrt_hidden
    ).to(source_dtype)
    gate_linear_fp32 = gate_linear.to(tl.float32)
    gate_sign = tl.where(
        gate_linear_fp32 > 0.0,
        1.0,
        tl.where(gate_linear_fp32 < 0.0, -1.0, 0.0),
    )
    signed_root = (
        gate_sign * tl.sqrt(tl.maximum(tl.abs(gate_linear_fp32), 1.0e-6))
    ).to(source_dtype)
    gate = tl.sigmoid(signed_root.to(tl.float32)).to(source_dtype)

    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    gated = (gate * value).to(source_dtype)
    gated_fp32 = gated.to(tl.float32)
    conv_weight = tl.load(
        conv_weight_ptr + weight_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    conv_inv_rms = tl.rsqrt(
        tl.sum(gated_fp32 * gated_fp32, axis=0) / hidden_size + eps
    )
    normalized = gated_fp32 * conv_inv_rms * (1.0 + conv_weight)

    tl.store(gated_ptr + row_offsets, gated, mask=mask)
    tl.store(normalized_ptr + row_offsets, normalized, mask=mask)


def can_use_ple_gate_norm_triton(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    hc_count: int,
    *,
    min_tokens: int = 0,
) -> bool:
    """Return whether the contiguous inference gate/norm path is supported."""

    tensors = (key, query, value, key_weight, query_weight, conv_weight)
    if not HAS_TRITON or hc_count <= 0:
        return False
    if key.ndim != 3 or query.shape != key.shape:
        return False
    tokens, branches, hidden_size = key.shape
    if branches != hc_count or tokens < min_tokens or hidden_size <= 0:
        return False
    if value.shape != (tokens, hidden_size):
        return False
    if any(
        weight.shape != (hc_count * hidden_size,) for weight in tensors[3:]
    ):
        return False
    device = key.device
    return bool(
        key.dtype in _SUPPORTED_DTYPES
        and query.dtype == key.dtype
        and value.dtype == key.dtype
        and device.type not in ("cpu", "meta")
        and all(tensor.device == device for tensor in tensors)
        and all(tensor.dtype in _SUPPORTED_DTYPES for tensor in tensors[3:])
        and all(tensor.is_contiguous() for tensor in tensors)
    )


def ple_gate_norm_(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    gated: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    eps: float,
) -> None:
    """Mutating custom-op implementation used by compiled model forward."""

    if not can_use_ple_gate_norm_triton(
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        hc_count,
    ):
        raise NotImplementedError("PLE gate/norm received unsupported tensors")
    if eps <= 0:
        raise ValueError("PLE gate/norm epsilon must be positive")
    if gated.shape != key.shape or normalized.shape != key.shape:
        raise ValueError("PLE gate/norm output buffers must match key shape")
    if gated.dtype != key.dtype or normalized.dtype != key.dtype:
        raise TypeError("PLE gate/norm output buffers must match key dtype")
    if not gated.is_contiguous() or not normalized.is_contiguous():
        raise ValueError("PLE gate/norm output buffers must be contiguous")
    if gated.device != key.device or normalized.device != key.device:
        raise ValueError("PLE gate/norm output buffers must match key device")
    if not key.numel():
        return

    hidden_size = key.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    rows = key.shape[0] * hc_count
    _ple_gate_norm_kernel[(rows,)](
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        inv_sqrt_hidden=1.0 / math.sqrt(hidden_size),
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
    )


def ple_gate_norm(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the two required results and launch the fused custom op."""

    gated = torch.empty_like(key)
    normalized = torch.empty_like(key)
    ple_gate_norm_(
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hc_count,
        eps,
    )
    return gated, normalized


__all__ = ["ple_gate_norm"]
