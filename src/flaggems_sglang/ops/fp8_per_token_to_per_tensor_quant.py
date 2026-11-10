# Copyright 2026, The FlagOS Contributors.
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

"""Requantize expert activations from per-group scales to one tensor scale."""

import torch
import triton
import triton.language as tl

__all__ = ["fp8_per_token_to_per_tensor_quant"]

# The scale groups are fixed at 128 columns; upstream asserts the hidden size
# is a multiple of it, and the contract keeps that.
_SCALE_GROUP = 128
_BLOCK_M = 16
_BLOCK_K = _SCALE_GROUP


@triton.jit
def _fp8_per_token_to_per_tensor_quant_kernel(
    x_ptr,
    xs_ptr,
    mm_ptr,
    os_ptr,
    out_ptr,
    M,
    K,
    XS_S0,
    XS_S1,
    XS_S2,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """One program per (expert, row tile, scale group).

    ``BK`` equals ``GROUP``, so a tile touches exactly one scale per row: the
    whole 2-D block shares one column of ``x_scale``.
    """
    e = tl.program_id(2)
    mb = tl.program_id(1)
    kb = tl.program_id(0)

    masked = tl.load(mm_ptr + e)
    rows = mb * BM + tl.arange(0, BM)
    cols = kb * BK + tl.arange(0, BK)

    row_ok = rows < M
    col_ok = cols < K
    in_block = row_ok[:, None] & col_ok[None, :]

    base = (e.to(tl.int64) * M + rows[:, None].to(tl.int64)) * K
    x = tl.load(x_ptr + base + cols[None, :], mask=in_block, other=0.0)

    # A row keeps its expert's group scale only while it is a live row; the
    # band past masked_m is zeroed regardless of what the scale says.
    live = rows < masked
    scale = tl.load(
        xs_ptr + e * XS_S0 + rows * XS_S1 + kb * XS_S2,
        mask=row_ok,
        other=0.0,
    )
    out_scale = tl.load(os_ptr)

    value = x.to(tl.float32) * scale[:, None] / out_scale
    value = tl.where(live[:, None], value, 0.0)
    tl.store(
        out_ptr + base + cols[None, :],
        value.to(out_ptr.dtype.element_ty),
        mask=in_block,
    )


def fp8_per_token_to_per_tensor_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    masked_m: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert per-group scaled activations to a single per-tensor scale.

    Args:
        x: ``[num_experts, m, hidden]`` activation payload. ``hidden`` must be a
            multiple of 128 and the row stride must be contiguous in the last
            dimension (the kernel addresses it as ``row * K + arange(BK)``).
        x_scale: ``[num_experts, m, hidden // 128]`` float32 group scales.
            DeepEP hands the last two dims over column-major, so this accepts
            arbitrary strides rather than requiring contiguity.
        masked_m: ``[num_experts]`` int32 live row count per expert.
        output_scale: single-element float32 tensor to quantize against.

    Returns:
        A new ``[num_experts, m, hidden]`` ``float8_e4m3fn`` tensor. Rows at or
        past ``masked_m[e]`` are zero.

    The payload is requantized in one pass: dequantize with the group scale,
    divide by ``output_scale``, cast to fp8.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D, got {tuple(x.shape)}")
    if x.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16):
        raise ValueError(f"x must be fp8/bf16/f16, got {x.dtype}")

    num_experts, m, hidden = x.shape
    if hidden % _SCALE_GROUP != 0:
        raise ValueError(
            f"hidden {hidden} is not a multiple of the {_SCALE_GROUP} scale group"
        )
    if x.stride(-1) != 1:
        raise ValueError("x must have a unit last-dimension stride")

    if x_scale.dim() != 3:
        raise ValueError(f"x_scale must be 3-D, got {tuple(x_scale.shape)}")
    if x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale must be float32, got {x_scale.dtype}")
    groups = hidden // _SCALE_GROUP
    if tuple(x_scale.shape) != (num_experts, m, groups):
        raise ValueError(
            f"x_scale must be {(num_experts, m, groups)}, got {tuple(x_scale.shape)}"
        )

    if masked_m.dim() != 1 or masked_m.shape[0] != num_experts:
        raise ValueError(
            f"masked_m must be [{num_experts}], got {tuple(masked_m.shape)}"
        )
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be int32, got {masked_m.dtype}")

    if output_scale.numel() != 1:
        raise ValueError(
            f"output_scale must hold one element, got {output_scale.numel()}"
        )
    if output_scale.dtype != torch.float32:
        raise ValueError(
            f"output_scale must be float32, got {output_scale.dtype}"
        )

    out = torch.empty(
        (num_experts, m, hidden), dtype=torch.float8_e4m3fn, device=x.device
    )
    if hidden == 0 or m == 0 or num_experts == 0:
        return out

    grid = (
        triton.cdiv(hidden, _BLOCK_K),
        triton.cdiv(m, _BLOCK_M),
        num_experts,
    )
    _fp8_per_token_to_per_tensor_quant_kernel[grid](
        x,
        x_scale,
        masked_m,
        output_scale,
        out,
        m,
        hidden,
        x_scale.stride(0),
        x_scale.stride(1),
        x_scale.stride(2),
        BM=_BLOCK_M,
        BK=_BLOCK_K,
        GROUP=_SCALE_GROUP,
        num_warps=4,
        num_stages=1,
    )
    return out
