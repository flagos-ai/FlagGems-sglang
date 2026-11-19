# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""qwen4_hc_scale_silu; adapted from FlagGems-vllm PR #455 dependencies.

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
def _hc_scale_silu_kernel(
    x_ptr,
    out_ptr,
    n: tl.constexpr,
    width: tl.constexpr,
    stride: tl.constexpr,
    hc: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    dtype: tl.constexpr = x_ptr.dtype.element_ty
    x = tl.load(
        x_ptr + (offsets // width) * stride + offsets % width, offsets < n, 0
    ).to(tl.float32)
    scaled = (x / hc).to(dtype).to(tl.float32)
    out = scaled * tl.sigmoid(scaled)
    tl.store(out_ptr + offsets, out, offsets < n)


def qwen4_hc_scale_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Fuse division and SiLU while retaining the division's source dtype."""
    if hc_count <= 0 or not (
        x.ndim == 2 and x.stride(1) == 1 and x.device.type != "cpu"
    ):
        raise ValueError(
            "HC scale SiLU requires a positive HC count and contiguous accelerator input"
        )
    out = torch.empty_like(x)
    if x.numel():
        _hc_scale_silu_kernel[(triton.cdiv(x.numel(), 256),)](
            x,
            out,
            n=x.numel(),
            width=x.shape[1],
            stride=x.stride(0),
            hc=hc_count,
            block=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["qwen4_hc_scale_silu"]
