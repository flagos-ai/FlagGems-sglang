# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-performance Triton kernel for moe/silu_and_mul_routed."""

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


def _device_ctx(device: torch.device):
    if device.type == "cpu":
        return contextlib.nullcontext()
    return torch.get_device_module(device).device(device)


def _select_tile_config(m: int, n: int) -> tuple[int, int, int]:
    """Deterministic 2D tile & warp heuristic (TRITON_DISABLE_AUTOTUNE safe)."""
    if n <= 128:
        block_n = triton.next_power_of_2(max(n, 32))
    elif m <= 16:
        block_n = 128 if n <= 2048 else 256
    elif m <= 64:
        block_n = 256
    else:
        block_n = 256 if n < 2048 else 512

    if m <= 64:
        block_m = 1
    elif m <= 512:
        block_m = 2
    else:
        block_m = 4

    elems = block_m * block_n
    if elems <= 256:
        num_warps = 2
    elif elems <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    return block_m, block_n, num_warps


@triton.jit
def _silu_and_mul_routed_kernel(
    gateup_ptr,
    weights_ptr,
    out_ptr,
    m,
    n,
    stride_gm,
    stride_gk,
    stride_w,
    stride_om,
    HAS_WEIGHTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    STREAM_LARGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask_n = offs_n < n

    stride_gm64 = tl.full((), stride_gm, tl.int64)
    stride_gk64 = tl.full((), stride_gk, tl.int64)
    stride_om64 = tl.full((), stride_om, tl.int64)

    if BLOCK_M == 1:
        row = tl.full((), pid_m, tl.int64)
        if INTERLEAVED:
            gate_ptrs = (
                gateup_ptr + row * stride_gm64 + (2 * offs_n) * stride_gk64
            )
            up_ptrs = (
                gateup_ptr + row * stride_gm64 + (2 * offs_n + 1) * stride_gk64
            )
        else:
            gate_ptrs = gateup_ptr + row * stride_gm64 + offs_n * stride_gk64
            up_ptrs = (
                gateup_ptr + row * stride_gm64 + (offs_n + n) * stride_gk64
            )

        if STREAM_LARGE:
            gate = tl.load(
                gate_ptrs,
                mask=mask_n,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            up = tl.load(
                up_ptrs, mask=mask_n, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
        else:
            gate = tl.load(gate_ptrs, mask=mask_n, other=0.0).to(tl.float32)
            up = tl.load(up_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        acc = gate * tl.sigmoid(gate) * up
        if HAS_WEIGHTS:
            w = tl.load(
                weights_ptr + row * stride_w, eviction_policy="evict_last"
            ).to(tl.float32)
            acc = acc * w

        out_ptrs = out_ptr + row * stride_om64 + offs_n
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_n)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask_m = offs_m < m
        mask_mn = mask_m[:, None] & mask_n[None, :]

        if INTERLEAVED:
            gate_ptrs = (
                gateup_ptr
                + offs_m[:, None] * stride_gm64
                + (2 * offs_n[None, :]) * stride_gk64
            )
            up_ptrs = (
                gateup_ptr
                + offs_m[:, None] * stride_gm64
                + (2 * offs_n[None, :] + 1) * stride_gk64
            )
        else:
            gate_ptrs = (
                gateup_ptr
                + offs_m[:, None] * stride_gm64
                + offs_n[None, :] * stride_gk64
            )
            up_ptrs = (
                gateup_ptr
                + offs_m[:, None] * stride_gm64
                + (offs_n[None, :] + n) * stride_gk64
            )

        if STREAM_LARGE:
            gate = tl.load(
                gate_ptrs,
                mask=mask_mn,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            up = tl.load(
                up_ptrs, mask=mask_mn, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
        else:
            gate = tl.load(gate_ptrs, mask=mask_mn, other=0.0).to(tl.float32)
            up = tl.load(up_ptrs, mask=mask_mn, other=0.0).to(tl.float32)

        acc = gate * tl.sigmoid(gate) * up
        if HAS_WEIGHTS:
            w = tl.load(
                weights_ptr + offs_m * stride_w,
                mask=mask_m,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            acc = acc * w[:, None]

        out_ptrs = out_ptr + offs_m[:, None] * stride_om64 + offs_n[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_mn)


def silu_and_mul_routed(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_interleaved: bool = True,
) -> torch.Tensor:
    if gateup_output.ndim != 2 or gateup_output.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected 2D gateup_output with even last dim, got {tuple(gateup_output.shape)}"
        )

    m, hidden = gateup_output.shape
    n = hidden // 2
    target_dtype = gateup_output.dtype if out_dtype is None else out_dtype
    out = torch.empty((m, n), dtype=target_dtype, device=gateup_output.device)

    if m == 0 or n == 0:
        return out

    has_weights = topk_weights is not None
    if has_weights:
        if topk_weights.ndim != 1 or topk_weights.shape[0] != m:
            raise ValueError(
                f"Expected 1D topk_weights of length {m}, got {tuple(topk_weights.shape)}"
            )
        w_ptr = topk_weights
        stride_w = topk_weights.stride(0)
    else:
        w_ptr = gateup_output
        stride_w = 0

    block_m, block_n, num_warps = _select_tile_config(m, n)
    stream_large = (
        gateup_output.numel() * gateup_output.element_size()
        >= 32 * 1024 * 1024
    )
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))

    with _device_ctx(gateup_output.device):
        _silu_and_mul_routed_kernel[grid](
            gateup_output,
            w_ptr,
            out,
            m,
            n,
            gateup_output.stride(0),
            gateup_output.stride(1),
            stride_w,
            out.stride(0),
            HAS_WEIGHTS=has_weights,
            INTERLEAVED=bool(use_interleaved),
            STREAM_LARGE=stream_large,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    return out


__all__ = ["silu_and_mul_routed"]
