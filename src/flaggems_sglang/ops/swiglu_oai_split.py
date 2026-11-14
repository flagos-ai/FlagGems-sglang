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


def _maybe_device_ctx(device: torch.device):
    if device.type == "cuda":
        if device.index is None or device.index == torch.cuda.current_device():
            return contextlib.nullcontext()
        return torch.cuda.device(device)
    if device.type == "cpu":
        return contextlib.nullcontext()
    return torch.get_device_module(device).device(device)


def _select_tile_config(m: int, n: int) -> tuple[int, int]:
    """Deterministic 1D-per-row vectorized tile & warp heuristic."""
    if n <= 256:
        block_n = 1 << (max(n, 32) - 1).bit_length()
    elif m <= 8:
        block_n = 256 if n <= 1024 else (512 if n <= 4096 else 1024)
    else:
        block_n = 512 if n <= 1024 else 1024

    num_warps = 2 if block_n <= 256 else 4
    return block_n, num_warps


@triton.jit
def _swiglu_oai_split_contig_kernel(
    x_ptr,
    out_ptr,
    n,
    alpha,
    beta,
    limit_val,
    EVEN_N: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    USE_CG: tl.constexpr,
    USE_CS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    in_base = x_ptr + row * n * 2
    out_base = out_ptr + row * n

    if EVEN_N:
        if USE_CG:
            gate = tl.load(in_base + offs_n, cache_modifier=".cg").to(
                tl.float32
            )
            up = tl.load(in_base + n + offs_n, cache_modifier=".cg").to(
                tl.float32
            )
        else:
            gate = tl.load(in_base + offs_n).to(tl.float32)
            up = tl.load(in_base + n + offs_n).to(tl.float32)
    else:
        mask = offs_n < n
        if USE_CG:
            gate = tl.load(
                in_base + offs_n, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            up = tl.load(
                in_base + n + offs_n,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            gate = tl.load(in_base + offs_n, mask=mask, other=0.0).to(
                tl.float32
            )
            up = tl.load(in_base + n + offs_n, mask=mask, other=0.0).to(
                tl.float32
            )

    if HAS_LIMIT:
        gate = tl.minimum(gate, limit_val)
        up = tl.minimum(tl.maximum(up, -limit_val), limit_val)

    out_val = (gate * tl.sigmoid(alpha * gate) * (up + beta)).to(
        out_ptr.dtype.element_ty
    )
    if EVEN_N:
        if USE_CS:
            tl.store(out_base + offs_n, out_val, cache_modifier=".cs")
        else:
            tl.store(out_base + offs_n, out_val)
    else:
        tl.store(out_base + offs_n, out_val, mask=offs_n < n)


@triton.jit
def _swiglu_oai_split_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    stride_xm,
    stride_xk,
    stride_om,
    alpha,
    beta,
    limit_val,
    HAS_LIMIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offs_n < n

    in_base = x_ptr + row * stride_xm
    gate = tl.load(in_base + offs_n * stride_xk, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(in_base + (offs_n + n) * stride_xk, mask=mask, other=0.0).to(
        tl.float32
    )

    if HAS_LIMIT:
        gate = tl.minimum(gate, limit_val)
        up = tl.minimum(tl.maximum(up, -limit_val), limit_val)

    out_val = (gate * tl.sigmoid(alpha * gate) * (up + beta)).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + row * stride_om + offs_n, out_val, mask=mask)


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected gate_up with ndim>=2 and even last dim, "
            f"got {tuple(gate_up.shape)}"
        )

    width = gate_up.shape[-1]
    n = width // 2
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    m = x.shape[0]
    target_dtype = gate_up.dtype if out_dtype is None else out_dtype
    out = gate_up.new_empty((m, n), dtype=target_dtype)

    if m == 0 or n == 0:
        return out

    has_limit = limit is not None
    limit_val = float(limit) if has_limit else 0.0
    block_n, num_warps = _select_tile_config(m, n)
    stride_xm = x.stride(0)
    stride_xk = x.stride(1)
    grid = (m, (n + block_n - 1) // block_n)

    with _maybe_device_ctx(x.device):
        if stride_xk == 1 and stride_xm == width:
            _swiglu_oai_split_contig_kernel[grid](
                x,
                out,
                n,
                float(alpha),
                float(beta),
                limit_val,
                EVEN_N=(n % block_n == 0),
                HAS_LIMIT=has_limit,
                USE_CG=(m >= 4),
                USE_CS=(m * width * x.element_size() >= 16777216),
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        else:
            _swiglu_oai_split_strided_kernel[grid](
                x,
                out,
                n,
                stride_xm,
                stride_xk,
                out.stride(0),
                float(alpha),
                float(beta),
                limit_val,
                HAS_LIMIT=has_limit,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
    return out


__all__ = ["swiglu_oai_split"]
