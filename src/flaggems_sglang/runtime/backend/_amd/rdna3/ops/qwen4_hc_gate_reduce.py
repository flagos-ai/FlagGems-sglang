# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""qwen4_hc_gate_reduce for RDNA3 (gfx1100).

Same kernel as the generic op. From 256 rows each program covers a wider
hidden slice, so the grid has fewer, larger programs; fewer rows keep the
generic launch shape.
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
def _hc_gate_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    stride_logits_row,
    stride_normed_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    dtype: tl.constexpr = normed_ptr.dtype.element_ty
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for branch in tl.static_range(0, hc_count):
        branch_offsets = branch * hidden_size + offsets
        logits = tl.load(
            logits_ptr + row * stride_logits_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normed = tl.load(
            normed_ptr + row * stride_normed_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(logits).to(dtype).to(tl.float32)
        product = (gate * normed).to(dtype).to(tl.float32)
        accumulator += product
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        accumulator / hc_count,
        mask=mask,
    )


def _config(rows, hidden_size):
    """block_h and num_warps."""
    if rows < 256:
        return 256, 4
    return min(4096, triton.next_power_of_2(hidden_size)), 8


def qwen4_hc_gate_reduce(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if logits.shape != normed.shape or logits.ndim < 2:
        raise ValueError(
            "Qwen4 HC gate logits and normalized input must match"
        )
    if hc_count <= 0 or logits.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC gate reduction requires a valid HC count")
    if not can_use_hc_triton(logits, normed):
        raise RuntimeError(
            "Qwen4 HC gate reduction requires contiguous tensors"
        )
    hidden_size = logits.shape[-1] // hc_count
    output = torch.empty(
        (*logits.shape[:-1], hidden_size),
        dtype=normed.dtype,
        device=normed.device,
    )
    if not logits.numel():
        return output
    rows = logits.numel() // logits.shape[-1]
    block_h, num_warps = _config(rows, hidden_size)
    _hc_gate_reduce_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        logits,
        normed,
        output,
        logits.stride(-2),
        normed.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["qwen4_hc_gate_reduce"]
