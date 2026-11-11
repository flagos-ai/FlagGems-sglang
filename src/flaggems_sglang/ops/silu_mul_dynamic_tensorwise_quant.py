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

"""Fused SiLU-multiply plus dynamic tensorwise FP8 quantisation.

Computes ``out[r, c] = silu(x[r, c]) * x[r, d + c] / scale`` for the active
rows selected by ``num_tokens_tensor``, where ``scale`` is not supplied by the
caller but derived from the data: ``max(absmax(activation), 1e-10)`` divided
by the finite FP8 maximum.

Dynamic scaling is the path that needs no calibration, which is why the MoE
quantisation path reaches for it, but it costs one extra sweep: the scale has
to be known before the first FP8 byte can be written, and a single Triton
launch has no device-wide barrier, so the traffic floor is "read twice, write
once". The two kernels here are that floor and nothing more.

Four things are deliberately different from the upstream implementation:

1. **Precise division instead of a reciprocal multiply.** Upstream computes
   ``scale = 1.0 / fp8_scale`` in the store kernel and multiplies. ``v * (1/s)``
   and ``v / s`` differ by up to one ulp in fp32, and one ulp is enough to
   move a value across an FP8 rounding boundary and change the encoded byte.
   Both the payload division and the reduction of the absmax into a scale use
   ``tl.div_rn`` here, so the bytes can be compared with no tolerance at all.
2. **The absmax slot holds the absmax, not the scaled absmax.** Taking the
   maximum and then dividing is the same answer as dividing and then taking the
   maximum, because correctly-rounded division is monotone, but it keeps the
   reduction in the integer bit domain where a plain int32 ``atomic_max`` is
   exact. Upstream reduces the quotient through a float CAS loop instead.
3. **The column split runs on a compile-time constant** (``ids //
   INTERMEDIATE_SIZE`` with a ``tl.constexpr`` divisor folds into a multiply
   and a shift), so recovering ``(row, column)`` from a flat lane index costs
   no integer division. That is what makes one flat grid-stride loop per pass
   workable.
4. **The count is clamped to the buffer capacity.** Upstream writes up to
   ``num_tokens * intermediate_size``; a caller that overstates the count
   writes past the end of its own buffer. Clamping makes the worst case a fully
   written buffer rather than a corrupt allocation.

The FP8 conversion is the implicit one performed by the store, so values
beyond the finite range saturate instead of wrapping.
"""

import torch
import triton
import triton.language as tl

__all__ = ["silu_mul_dynamic_tensorwise_quant"]

# Flat tile per program iteration, matching the static sibling so the two
# passes and the static path tile the buffer identically.
_BLOCK = 4096
_MAX_PROGRAMS = 1024
# Software pipelining across the grid-stride iterations of the reduce pass: the
# loads of one iteration overlap the reduction of the previous one.
_NUM_STAGES = 3
# The floor under the derived scale, so an all-zero prefix does not divide by
# zero and turn every byte into a NaN payload.
_EPS = 1e-10
# The finite maximum of e4m3fn, as the quantisation grid the scale is built on.
_FP8_MAX = 448.0

_INPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_OUTPUT_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


@triton.jit
def _silu_mul_absmax_kernel(
    IN,
    SLOT,
    NUM_TOKENS,
    STRIDE_IN,
    INTERMEDIATE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """First pass: the absmax of the activation over the active prefix."""
    num_tokens = tl.load(NUM_TOKENS)
    limit = tl.minimum(num_tokens * INTERMEDIATE_SIZE, CAPACITY)

    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK

    absmax = 0.0
    for base in tl.range(start, limit, step, num_stages=NUM_STAGES):
        ids = base + tl.arange(0, BLOCK)
        mask = ids < limit

        token = ids // INTERMEDIATE_SIZE
        column = ids - token * INTERMEDIATE_SIZE
        offset = token * STRIDE_IN + column

        gate = tl.load(IN + offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(IN + offset + INTERMEDIATE_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )
        activation = tl.div_rn(gate, 1.0 + tl.exp(-gate)) * up
        # Masked lanes loaded zero, so they cannot win this reduction.
        absmax = tl.maximum(absmax, tl.max(tl.abs(activation)))

    # abs() is non-negative, so its bit pattern orders like the value and an
    # integer max is an exact float max here. This sidesteps float atomics,
    # which not every backend implements the same way.
    tl.atomic_max(SLOT, (absmax.to(tl.int32, bitcast=True)))


@triton.jit
def _silu_mul_quantize_kernel(
    IN,
    OUT,
    SLOT,
    SCALE_OUT,
    NUM_TOKENS,
    STRIDE_IN,
    FP8_MAX,
    INTERMEDIATE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Second pass: the same activation, divided by the derived scale."""
    out_dtype = OUT.dtype.element_ty

    num_tokens = tl.load(NUM_TOKENS)
    limit = tl.minimum(num_tokens * INTERMEDIATE_SIZE, CAPACITY)

    # One scale for the whole tensor, so it is recovered once per program
    # rather than once per lane. The divide is precise so that this matches the
    # reference's division of the same absmax by the same constant.
    absmax = tl.load(SLOT).to(tl.float32, bitcast=True)
    absmax = tl.maximum(absmax, 1e-10)
    # The constant is forced to fp32 first: a Python float arriving as a kernel
    # argument is not guaranteed to be one, and dividing in a wider type and
    # rounding the quotient back to fp32 costs the odd ulp.
    fp8_max = FP8_MAX.to(tl.float32)
    scale = tl.div_rn(absmax, fp8_max)
    if tl.program_id(0) == 0:
        # Every program computes the same value, so publishing it costs one
        # store and saves the host a separate pass over one element.
        tl.store(SCALE_OUT, scale)

    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK

    for base in tl.range(start, limit, step, num_stages=NUM_STAGES):
        ids = base + tl.arange(0, BLOCK)
        mask = ids < limit

        token = ids // INTERMEDIATE_SIZE
        column = ids - token * INTERMEDIATE_SIZE
        offset = token * STRIDE_IN + column

        gate = tl.load(IN + offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(IN + offset + INTERMEDIATE_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )
        activation = tl.div_rn(gate, 1.0 + tl.exp(-gate)) * up
        tl.store(
            OUT + ids,
            tl.div_rn(activation, scale).to(out_dtype),
            mask=mask,
        )


def _check(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    if input.dim() != 2:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects a 2-D "
            f"(rows, 2 * intermediate_size) input, got {input.dim()}-D"
        )
    if input.dtype not in _INPUT_DTYPES:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant supports bfloat16, float16 and "
            f"float32 inputs, got {input.dtype}"
        )
    if input.stride(1) != 1:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs a last-dim contiguous "
            "input; a padded row stride is fine, interleaved columns are not"
        )
    if input.device.type != "cuda":
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs a CUDA tensor"
        )
    if int(intermediate_size) <= 0:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs a positive "
            f"intermediate_size, got {intermediate_size}"
        )
    if input.shape[1] != 2 * int(intermediate_size):
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects the last dim to hold "
            f"both halves, {2 * int(intermediate_size)}, got {input.shape[1]}"
        )
    if input.shape[0] < int(expected_num_tokens):
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects at least "
            f"{expected_num_tokens} input rows, got {input.shape[0]}"
        )
    if output.dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant writes an FP8 buffer, got "
            f"{output.dtype}"
        )
    if output.shape != (int(expected_num_tokens), int(intermediate_size)):
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects an output of shape "
            f"({int(expected_num_tokens)}, {int(intermediate_size)}), got "
            f"{tuple(output.shape)}"
        )
    if output.device != input.device:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs the output on the same "
            "device as the input"
        )
    if scale.numel() != 1:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects a one-element scale "
            f"slot, got {scale.numel()} elements"
        )
    if scale.dtype != torch.float32:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects a float32 scale slot, "
            f"got {scale.dtype}"
        )
    if scale.device != input.device:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs the scale on the same "
            "device as the input"
        )
    if num_tokens_tensor.numel() != 1:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant expects a one-element token "
            f"count, got {num_tokens_tensor.numel()} elements"
        )
    if num_tokens_tensor.dtype != torch.int32:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant reads the token count as int32 "
            f"inside the kernel, got {num_tokens_tensor.dtype}"
        )
    if num_tokens_tensor.device != input.device:
        raise ValueError(
            "silu_mul_dynamic_tensorwise_quant needs the token count on the "
            "device, not on the host"
        )


def silu_mul_dynamic_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    """Fused gated SiLU plus dynamic tensorwise FP8 quantisation.

    Args:
        input: ``(rows, 2 * intermediate_size)`` activation buffer. The first
            ``intermediate_size`` columns of each row are the gate half and the
            next ``intermediate_size`` are the up half.
        output: ``(expected_num_tokens, intermediate_size)`` FP8 buffer whose
            current contents are the tail that must survive.
        scale: one-element float32 scale slot. The contract declares no
            mutation, so it is read-only here and the derived scale is returned
            separately instead of being written into it.
        num_tokens_tensor: one-element int32 device tensor holding the number
            of active rows.
        expected_num_tokens: row capacity of ``output``.
        intermediate_size: width of each half.

    Returns:
        ``(quantized, scale_out)``. ``quantized`` is a copy of ``output`` whose
        first ``num_tokens`` rows hold ``silu(gate) * up / scale`` with the
        derived scale, cast to the output dtype; rows at and beyond the token
        count are left exactly as the caller had them, and the caller buffer
        itself is never written. ``scale_out`` is a one-element float32 tensor
        holding ``max(absmax, 1e-10) / 448``.
    """
    _check(
        input,
        output,
        scale,
        num_tokens_tensor,
        expected_num_tokens,
        intermediate_size,
    )

    # The contract declares no mutation, so the tail comes from a copy rather
    # than from the buffer the caller passed in.
    quantized = output.clone(memory_format=torch.contiguous_format)

    width = int(intermediate_size)
    capacity = int(expected_num_tokens) * width
    if capacity == 0:
        # Nothing to average over: the same floor an empty prefix gets inside
        # the kernel, divided on the device so both paths round identically.
        # The divisor is a tensor because torch's tensor/scalar division goes
        # through a reciprocal and is not the IEEE quotient.
        empty = torch.tensor([_EPS], device=input.device, dtype=torch.float32)
        divisor = torch.tensor(
            [_FP8_MAX], device=input.device, dtype=torch.float32
        )
        return quantized, empty / divisor

    # Integer slot: zero initialises to +0.0, and every value reduced into it
    # is the non-negative bit pattern of an abs().
    slot = torch.zeros(1, device=input.device, dtype=torch.int32)
    scale_out = torch.empty(1, device=input.device, dtype=torch.float32)

    programs = min(triton.cdiv(capacity, _BLOCK), _MAX_PROGRAMS)
    _silu_mul_absmax_kernel[(programs,)](
        input,
        slot,
        num_tokens_tensor,
        input.stride(0),
        width,
        capacity,
        _BLOCK,
        _NUM_STAGES,
        num_warps=8,
        num_stages=1,
    )
    _silu_mul_quantize_kernel[(programs,)](
        input,
        quantized,
        slot,
        scale_out,
        num_tokens_tensor,
        input.stride(0),
        _FP8_MAX,
        width,
        capacity,
        _BLOCK,
        _NUM_STAGES,
        num_warps=8,
        num_stages=1,
    )

    return quantized, scale_out
