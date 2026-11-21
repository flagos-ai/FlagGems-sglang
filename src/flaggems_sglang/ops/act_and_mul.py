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

"""High-performance Triton kernel for moe/act_and_mul."""

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
        block_n = 256 if n <= 1024 else 512
    else:
        block_n = 512 if n <= 1024 else 1024

    num_warps = 2 if block_n <= 256 else 4
    return block_n, num_warps


@triton.jit
def _act_and_mul_contig_kernel(
    gateup_ptr,
    out_ptr,
    n,
    limit_val,
    EVEN_N: tl.constexpr,
    IS_GELU: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    USE_CG: tl.constexpr,
    USE_CS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ty = out_ptr.dtype.element_ty
    in_base = gateup_ptr + row * n * 2
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
        gate = tl.load(in_base + offs_n, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(in_base + n + offs_n, mask=mask, other=0.0).to(tl.float32)

    if HAS_LIMIT:
        gate = tl.minimum(gate, limit_val)
        up = tl.minimum(tl.maximum(up, -limit_val), limit_val)

    if IS_GELU:
        u = 1.5957691216057308 * (gate + 0.044715 * gate * gate * gate)
        u = tl.maximum(tl.minimum(u, 88.0), -88.0)
        act = gate * tl.sigmoid(u)
    else:
        act = gate * tl.sigmoid(gate)

    act_q = act.to(out_ty).to(tl.float32)
    up_q = up.to(out_ty).to(tl.float32)
    out = (act_q * up_q).to(out_ty)

    if EVEN_N:
        if USE_CS:
            tl.store(out_base + offs_n, out, cache_modifier=".cs")
        else:
            tl.store(out_base + offs_n, out)
    else:
        tl.store(out_base + offs_n, out, mask=offs_n < n)


@triton.jit
def _act_and_mul_strided_kernel(
    gateup_ptr,
    out_ptr,
    n,
    stride_gm,
    stride_gk,
    stride_om,
    limit_val,
    IS_GELU: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offs_n < n
    out_ty = out_ptr.dtype.element_ty

    in_base = gateup_ptr + row * stride_gm
    gate = tl.load(in_base + offs_n * stride_gk, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(in_base + (offs_n + n) * stride_gk, mask=mask, other=0.0).to(
        tl.float32
    )

    if HAS_LIMIT:
        gate = tl.minimum(gate, limit_val)
        up = tl.minimum(tl.maximum(up, -limit_val), limit_val)

    if IS_GELU:
        u = 1.5957691216057308 * (gate + 0.044715 * gate * gate * gate)
        u = tl.maximum(tl.minimum(u, 88.0), -88.0)
        act = gate * tl.sigmoid(u)
    else:
        act = gate * tl.sigmoid(gate)

    act_q = act.to(out_ty).to(tl.float32)
    up_q = up.to(out_ty).to(tl.float32)
    out = (act_q * up_q).to(out_ty)
    tl.store(out_ptr + row * stride_om + offs_n, out, mask=mask)


def act_and_mul(
    gateup_output: torch.Tensor,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if activation == "silu":
        is_gelu = False
    elif activation == "gelu":
        is_gelu = True
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    ndim = gateup_output.ndim
    if ndim < 2 or gateup_output.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected gateup_output with ndim>=2 and even last dim, "
            f"got {tuple(gateup_output.shape)}"
        )

    two_n = gateup_output.shape[-1]
    n = two_n // 2
    x2d = gateup_output if ndim == 2 else gateup_output.reshape(-1, two_n)

    m = x2d.shape[0]
    out2d = gateup_output.new_empty((m, n))
    if m == 0 or n == 0:
        return out2d if ndim == 2 else out2d.view(*gateup_output.shape[:-1], n)

    has_limit = swiglu_limit is not None
    limit_val = float(swiglu_limit) if has_limit else 0.0
    block_n, num_warps = _select_tile_config(m, n)
    stride_gm = x2d.stride(0)
    stride_gk = x2d.stride(1)
    grid = (m, (n + block_n - 1) // block_n)

    with _maybe_device_ctx(gateup_output.device):
        if stride_gk == 1 and stride_gm == two_n:
            _act_and_mul_contig_kernel[grid](
                x2d,
                out2d,
                n,
                limit_val,
                EVEN_N=(n % block_n == 0),
                IS_GELU=is_gelu,
                HAS_LIMIT=has_limit,
                USE_CG=(m >= 4),
                USE_CS=(m * two_n * gateup_output.element_size() >= 16777216),
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        else:
            _act_and_mul_strided_kernel[grid](
                x2d,
                out2d,
                n,
                stride_gm,
                stride_gk,
                out2d.stride(0),
                limit_val,
                IS_GELU=is_gelu,
                HAS_LIMIT=has_limit,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
    return out2d if ndim == 2 else out2d.view(*gateup_output.shape[:-1], n)


__all__ = ["act_and_mul"]
