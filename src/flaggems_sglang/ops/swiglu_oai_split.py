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

"""High-performance 2D strided Triton kernel for moe/swiglu_oai_split."""

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
    """Deterministic 2D tile & warp heuristic (eliminates 1D div/mod & constexpr recompile)."""
    if n <= 128:
        block_n = triton.next_power_of_2(max(n, 32))
    elif m <= 16:
        block_n = 128 if n <= 2048 else 256
    elif m <= 64:
        block_n = 256 if n <= 4096 else 512
    else:
        block_n = 512 if n >= 2048 else 256

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
def _swiglu_oai_split_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    stride_xm,
    stride_xk,
    stride_om,
    alpha,
    beta,
    limit_val,
    CONTIGUOUS_INNER: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    STREAM_LARGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask_n = offs_n < n

    stride_xm64 = tl.full((), stride_xm, tl.int64)
    stride_xk64 = tl.full((), stride_xk, tl.int64)
    stride_om64 = tl.full((), stride_om, tl.int64)

    alpha_f = tl.full((), alpha, tl.float32)
    beta_f = tl.full((), beta, tl.float32)

    if BLOCK_M == 1:
        row = tl.full((), pid_m, tl.int64)
        if CONTIGUOUS_INNER:
            gate_ptrs = x_ptr + row * stride_xm64 + offs_n
            up_ptrs = x_ptr + row * stride_xm64 + offs_n + n
        else:
            gate_ptrs = x_ptr + row * stride_xm64 + offs_n * stride_xk64
            up_ptrs = x_ptr + row * stride_xm64 + (offs_n + n) * stride_xk64

        if STREAM_LARGE:
            gate = tl.load(
                gate_ptrs, mask=mask_n, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            up = tl.load(
                up_ptrs, mask=mask_n, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
        else:
            gate = tl.load(gate_ptrs, mask=mask_n, other=0.0).to(tl.float32)
            up = tl.load(up_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        if HAS_LIMIT:
            lim = tl.full((), limit_val, tl.float32)
            gate = tl.minimum(gate, lim)
            up = tl.minimum(tl.maximum(up, -lim), lim)

        out = gate * tl.sigmoid(alpha_f * gate) * (up + beta_f)
        tl.store(
            out_ptr + row * stride_om64 + offs_n,
            out.to(out_ptr.dtype.element_ty),
            mask=mask_n,
        )
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask_m = offs_m < m
        mask_mn = mask_m[:, None] & mask_n[None, :]

        if CONTIGUOUS_INNER:
            gate_ptrs = x_ptr + offs_m[:, None] * stride_xm64 + offs_n[None, :]
            up_ptrs = x_ptr + offs_m[:, None] * stride_xm64 + offs_n[None, :] + n
        else:
            gate_ptrs = (
                x_ptr + offs_m[:, None] * stride_xm64 + offs_n[None, :] * stride_xk64
            )
            up_ptrs = (
                x_ptr
                + offs_m[:, None] * stride_xm64
                + (offs_n[None, :] + n) * stride_xk64
            )

        if STREAM_LARGE:
            gate = tl.load(
                gate_ptrs, mask=mask_mn, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            up = tl.load(
                up_ptrs, mask=mask_mn, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
        else:
            gate = tl.load(gate_ptrs, mask=mask_mn, other=0.0).to(tl.float32)
            up = tl.load(up_ptrs, mask=mask_mn, other=0.0).to(tl.float32)

        if HAS_LIMIT:
            lim = tl.full((), limit_val, tl.float32)
            gate = tl.minimum(gate, lim)
            up = tl.minimum(tl.maximum(up, -lim), lim)

        out = gate * tl.sigmoid(alpha_f * gate) * (up + beta_f)
        out_ptrs = out_ptr + offs_m[:, None] * stride_om64 + offs_n[None, :]
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_mn)


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected gate_up with ndim>=2 and even last dim, got {tuple(gate_up.shape)}"
        )

    width = gate_up.shape[-1]
    n = width // 2
    if gate_up.ndim == 2:
        x = gate_up
    else:
        try:
            x = gate_up.view(-1, width)
        except RuntimeError:
            x = gate_up.reshape(-1, width)

    m = x.shape[0]
    target_dtype = gate_up.dtype if out_dtype is None else out_dtype
    out = torch.empty((m, n), dtype=target_dtype, device=gate_up.device)

    if m == 0 or n == 0:
        return out

    has_limit = limit is not None
    limit_val = float(limit) if has_limit else 0.0
    block_m, block_n, num_warps = _select_tile_config(m, n)
    stream_large = x.numel() * x.element_size() >= 32 * 1024 * 1024
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))

    with _device_ctx(x.device):
        _swiglu_oai_split_kernel[grid](
            x,
            out,
            m,
            n,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            float(alpha),
            float(beta),
            limit_val,
            CONTIGUOUS_INNER=(x.stride(1) == 1),
            HAS_LIMIT=has_limit,
            STREAM_LARGE=stream_large,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    return out


__all__ = ["swiglu_oai_split"]
