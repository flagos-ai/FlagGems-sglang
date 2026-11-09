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


def _maybe_device_ctx(device: torch.device):
    if device.type == "cuda":
        if device.index is None or device.index == torch.cuda.current_device():
            return contextlib.nullcontext()
        return torch.cuda.device(device)
    if device.type == "cpu":
        return contextlib.nullcontext()
    return torch.get_device_module(device).device(device)


def _select_tile_config(m: int, n: int, interleaved: bool) -> tuple[int, int]:
    """Deterministic 1D-per-row vectorized tile & warp heuristic."""
    if n <= 256:
        block_n = 1 << (max(n, 32) - 1).bit_length()
    elif interleaved:
        if m <= 8:
            block_n = 256 if n <= 1024 else 512
        else:
            block_n = 512
    else:
        if m <= 8:
            block_n = 256 if n <= 1024 else 512
        else:
            block_n = 512 if n <= 1024 else 1024

    elems_loaded = (2 * block_n) if interleaved else block_n
    num_warps = 2 if elems_loaded <= 256 else 4
    return block_n, num_warps


@triton.jit
def _silu_and_mul_routed_contig_kernel(
    gateup_ptr,
    weights_ptr,
    out_ptr,
    n,
    stride_w,
    EVEN_N: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    USE_CG: tl.constexpr,
    USE_CS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_base = gateup_ptr + row * n * 2
    out_base = out_ptr + row * n

    if INTERLEAVED:
        offs_2n = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
        if EVEN_N:
            if USE_CG:
                pair = tl.load(in_base + offs_2n, cache_modifier=".cg").to(
                    tl.float32
                )
            else:
                pair = tl.load(in_base + offs_2n).to(tl.float32)
        else:
            pair = tl.load(
                in_base + offs_2n, mask=offs_2n < (2 * n), other=0.0
            ).to(tl.float32)
        gate, up = tl.split(tl.reshape(pair, (BLOCK_N, 2)))
    else:
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
            gate = tl.load(in_base + offs_n, mask=mask, other=0.0).to(
                tl.float32
            )
            up = tl.load(in_base + n + offs_n, mask=mask, other=0.0).to(
                tl.float32
            )

    acc = gate * tl.sigmoid(gate) * up
    if HAS_WEIGHTS:
        w = tl.load(weights_ptr + row * stride_w, cache_modifier=".ca").to(
            tl.float32
        )
        acc = acc * w

    out_val = acc.to(out_ptr.dtype.element_ty)
    if EVEN_N:
        if USE_CS:
            tl.store(out_base + offs_n, out_val, cache_modifier=".cs")
        else:
            tl.store(out_base + offs_n, out_val)
    else:
        tl.store(out_base + offs_n, out_val, mask=offs_n < n)


@triton.jit
def _silu_and_mul_routed_strided_kernel(
    gateup_ptr,
    weights_ptr,
    out_ptr,
    n,
    stride_gm,
    stride_gk,
    stride_w,
    stride_om,
    HAS_WEIGHTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offs_n < n
    in_base = gateup_ptr + row * stride_gm

    if INTERLEAVED:
        gate = tl.load(
            in_base + (2 * offs_n) * stride_gk, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            in_base + (2 * offs_n + 1) * stride_gk, mask=mask, other=0.0
        ).to(tl.float32)
    else:
        gate = tl.load(in_base + offs_n * stride_gk, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            in_base + (offs_n + n) * stride_gk, mask=mask, other=0.0
        ).to(tl.float32)

    acc = gate * tl.sigmoid(gate) * up
    if HAS_WEIGHTS:
        w = tl.load(weights_ptr + row * stride_w, cache_modifier=".ca").to(
            tl.float32
        )
        acc = acc * w

    out_val = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_om + offs_n, out_val, mask=mask)


def silu_and_mul_routed(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_interleaved: bool = True,
) -> torch.Tensor:
    if gateup_output.ndim != 2 or gateup_output.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected 2D gateup_output with even last dim, "
            f"got {tuple(gateup_output.shape)}"
        )

    m, hidden = gateup_output.shape
    n = hidden // 2
    target_dtype = gateup_output.dtype if out_dtype is None else out_dtype
    out = gateup_output.new_empty((m, n), dtype=target_dtype)

    if m == 0 or n == 0:
        return out

    has_weights = topk_weights is not None
    if has_weights:
        if topk_weights.ndim != 1 or topk_weights.shape[0] != m:
            raise ValueError(
                f"Expected 1D topk_weights of length {m}, "
                f"got {tuple(topk_weights.shape)}"
            )
        w_ptr = topk_weights
        stride_w = topk_weights.stride(0)
    else:
        w_ptr = gateup_output
        stride_w = 0

    interleaved = bool(use_interleaved)
    block_n, num_warps = _select_tile_config(m, n, interleaved)
    stride_gm = gateup_output.stride(0)
    stride_gk = gateup_output.stride(1)
    grid = (m, (n + block_n - 1) // block_n)

    with _maybe_device_ctx(gateup_output.device):
        if stride_gk == 1 and stride_gm == hidden:
            _silu_and_mul_routed_contig_kernel[grid](
                gateup_output,
                w_ptr,
                out,
                n,
                stride_w,
                EVEN_N=(n % block_n == 0),
                HAS_WEIGHTS=has_weights,
                INTERLEAVED=interleaved,
                USE_CG=(m >= 4),
                USE_CS=(m * hidden * gateup_output.element_size() >= 16777216),
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        else:
            _silu_and_mul_routed_strided_kernel[grid](
                gateup_output,
                w_ptr,
                out,
                n,
                stride_gm,
                stride_gk,
                stride_w,
                out.stride(0),
                HAS_WEIGHTS=has_weights,
                INTERLEAVED=interleaved,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
    return out


__all__ = ["silu_and_mul_routed"]
