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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl

__all__ = ["per_tensor_quant_mla_fp8"]

_HEAD_BLOCK = 1024
_MAX_HEAD_SIZE = 32768


@triton.jit
def _absmax_bits(
    X,
    BITS,
    HEAD_SIZE: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold one (head, sequence) row into the running per-tensor absmax.

    The absolute value of a float is that float with its sign bit cleared, and
    for non-negative values the bit pattern orders exactly like the number, so
    the reduction runs on int32 bit patterns and the accumulator is an int32
    slot. That keeps the fold exact and avoids depending on a float atomic
    maximum.
    """
    seq = tl.program_id(0)
    head = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_SIZE
    row = tl.load(
        X + head * STRIDE_H + seq * STRIDE_S + lane,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    magnitude = tl.abs(row).to(tl.int32, bitcast=True)
    tl.atomic_max(BITS, tl.max(magnitude))


@triton.jit
def _quantize_rows(
    X,
    BITS,
    Q,
    S,
    NUM_SEQ: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_S: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_SIZE

    bits = tl.load(BITS)
    amax = tl.maximum(bits.to(tl.float32, bitcast=True), EPS)
    # A precise division rather than a multiply by the reciprocal: the
    # reference divides too, and matching the two bit for bit is what lets the
    # test compare encoded bytes with no tolerance at all.
    scale = tl.div_rn(amax, 448.0)
    if (seq == 0) & (head == 0):
        tl.store(S, scale)

    row = tl.load(
        X + head * STRIDE_H + seq * STRIDE_S + lane,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    quantized = tl.clamp(tl.div_rn(row, scale), -448.0, 448.0).to(tl.float8e4nv)
    # The payload is head-major and contiguous whatever the input strides
    # were, which is the layout the absorbed MLA path consumes.
    tl.store(
        Q + head * NUM_SEQ * HEAD_SIZE + seq * HEAD_SIZE + lane,
        quantized.to(tl.uint8, bitcast=True),
        mask=mask,
    )


def _check(x, eps):
    if x.dim() != 3:
        raise ValueError(
            "per_tensor_quant_mla_fp8 expects a 3-D (num_head, num_seq, "
            f"head_size) tensor, got {x.dim()}-D"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "per_tensor_quant_mla_fp8 supports bfloat16, float16 and "
            f"float32 inputs, got {x.dtype}"
        )
    if x.device.type != "cuda":
        raise ValueError("per_tensor_quant_mla_fp8 needs a CUDA tensor")
    if float(eps) <= 0.0:
        raise ValueError(f"per_tensor_quant_mla_fp8 needs eps > 0, got {eps}")


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    """Quantize the absorbed MLA tensor to FP8 with a single scale.

    ``x`` is the absorbed MLA operand in ``(num_head, num_seq, head_size)``
    order. One scale covers the whole tensor: it is the largest magnitude
    anywhere in ``x``, floored at ``eps``, divided by the finite FP8 maximum.
    Every element is then divided by that scale, saturated to the FP8 range and
    encoded. The contract declares no mutation, so the quantized tensor is a
    fresh contiguous allocation in head-major order.

    Two passes are unavoidable, because the scale of the whole tensor has to be
    known before the first byte can be encoded: the first pass folds every
    row's maximum into one accumulator, the second reads it and writes the
    payload. Nothing else is shared, so a wider grid only helps.

    Returns ``(quantized, scale)``, where ``scale`` is a one-element float32
    tensor as in the upstream entry point.
    """
    _check(x, eps)

    num_head, num_seq, head_size = x.shape
    shape = (num_head, num_seq, head_size)
    if head_size > _MAX_HEAD_SIZE:
        raise ValueError(
            "per_tensor_quant_mla_fp8 handles head_size up to "
            f"{_MAX_HEAD_SIZE}, got {head_size}"
        )

    quantized = torch.empty(
        shape,
        dtype=torch.float8_e4m3fn,
        device=x.device,
        memory_format=torch.contiguous_format,
    )
    scale = torch.empty((1,), dtype=torch.float32, device=x.device)

    if num_head == 0 or num_seq == 0 or head_size == 0:
        # No row to fold, so the floor is the whole scale. The caller still
        # gets two well-formed outputs instead of an uninitialised slot.
        scale.fill_(float(eps) / 448.0)
        return quantized, scale

    block = min(_HEAD_BLOCK, triton.next_power_of_2(head_size))
    warps = min(16, max(4, block // 128))
    bits = torch.zeros((1,), dtype=torch.int32, device=x.device)
    grid = (num_seq, num_head)

    _absmax_bits[grid](
        x,
        bits,
        head_size,
        x.stride(0),
        x.stride(1),
        block,
        num_warps=warps,
        num_stages=1,
    )
    _quantize_rows[grid](
        x,
        bits,
        quantized.view(torch.uint8),
        scale,
        num_seq,
        head_size,
        x.stride(0),
        x.stride(1),
        float(eps),
        block,
        num_warps=warps,
        num_stages=1,
    )
    return quantized, scale
