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

"""Fused SiLU-multiply plus static tensorwise FP8 quantisation.

Computes ``out[r, c] = silu(x[r, c]) * x[r, d + c] / scale`` for the active
rows selected by ``num_tokens_tensor``, casting to the output's FP8 dtype on
store, and returns a copy of the caller's buffer with that prefix replaced.

The whole point of the fused form is that the intermediate activation never
exists: the unfused chain materialises a bf16 ``silu(gate) * up`` tensor and
then reads it back through a cast, so it pays three passes over the data plus
an allocation. This kernel reads the two input halves once, writes the FP8
result once, and allocates nothing but the copy the contract requires.

Two design choices follow from how the MoE path calls this:

1. **``num_tokens`` is read inside the kernel.** The token count lives in a
   device tensor, so a host read would serialise the launch behind a copy.
   The grid is sized for the buffer's full capacity instead and the kernel
   masks the tail itself, which means the host never waits for the count.
2. **The flat index is int32 and the division is by ``tl.constexpr``.**
   ``ids // INTERMEDIATE_SIZE`` with a compile-time divisor folds into a
   multiply and a shift, so recovering the ``(row, column)`` pair for each
   lane costs no integer division. That is what allows one flat grid-stride
   loop to cover the buffer without a 2-D grid whose row count is unknown
   until the kernel runs.

The FP8 conversion is the implicit one performed by the store: a destination
that is an FP8 pointer saturates on downcast, so no explicit clamp is needed
and values beyond the finite range land on the largest representable
magnitude rather than wrapping.
"""

import torch
import triton
import triton.language as tl

__all__ = ["silu_mul_static_tensorwise_quant"]

# Flat tile per program iteration. 4096 keeps the loop bodies long enough to
# amortise the loop overhead without shrinking the live tile below a few
# vectorised loads per thread.
_BLOCK = 4096
# Enough programs to fill the device with the loop still covering the buffer
# in a handful of iterations. Beyond this the grid-stride loop is the cheaper
# side of the trade.
_MAX_PROGRAMS = 1024
# Software pipelining across the grid-stride iterations: the loads of one
# iteration overlap the store of the previous one.
_NUM_STAGES = 3

_INPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_OUTPUT_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


@triton.jit
def _silu_mul_static_tensorwise_quant_kernel(
    IN,
    OUT,
    SCALE,
    NUM_TOKENS,
    STRIDE_IN,
    INTERMEDIATE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Grid-stride flat loop over the active prefix of the output buffer."""
    out_dtype = OUT.dtype.element_ty

    # A device value, read once per program: no host round trip anywhere.
    num_tokens = tl.load(NUM_TOKENS)
    # The contract calls for the active prefix. Clamping to the buffer keeps a
    # caller that overstates the count from writing past the allocation.
    limit = tl.minimum(num_tokens * INTERMEDIATE_SIZE, CAPACITY)
    scale = tl.load(SCALE).to(tl.float32)

    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK

    for base in tl.range(start, limit, step, num_stages=NUM_STAGES):
        ids = base + tl.arange(0, BLOCK)
        mask = ids < limit

        # Constant divisor: a multiply and a shift, not a division.
        token = ids // INTERMEDIATE_SIZE
        column = ids - token * INTERMEDIATE_SIZE
        # The gate and up halves of a row are adjacent, so both loads share
        # one address computation.
        offset = token * STRIDE_IN + column

        gate = tl.load(IN + offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(IN + offset + INTERMEDIATE_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )

        # A precise division rather than a reciprocal multiply: the reference
        # divides too, and matching it bit for bit is what lets the test
        # compare encoded bytes with no tolerance at all.
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
            "silu_mul_static_tensorwise_quant expects a 2-D "
            f"(rows, 2 * intermediate_size) input, got {input.dim()}-D"
        )
    if input.dtype not in _INPUT_DTYPES:
        raise ValueError(
            "silu_mul_static_tensorwise_quant supports bfloat16, float16 and "
            f"float32 inputs, got {input.dtype}"
        )
    if input.stride(1) != 1:
        raise ValueError(
            "silu_mul_static_tensorwise_quant needs a last-dim contiguous "
            "input; a padded row stride is fine, interleaved columns are not"
        )
    if input.device.type != "cuda":
        raise ValueError("silu_mul_static_tensorwise_quant needs a CUDA tensor")
    if int(intermediate_size) <= 0:
        raise ValueError(
            "silu_mul_static_tensorwise_quant needs a positive "
            f"intermediate_size, got {intermediate_size}"
        )
    if input.shape[1] != 2 * int(intermediate_size):
        raise ValueError(
            "silu_mul_static_tensorwise_quant expects the last dim to hold "
            f"both halves, {2 * int(intermediate_size)}, got {input.shape[1]}"
        )
    if input.shape[0] < int(expected_num_tokens):
        raise ValueError(
            "silu_mul_static_tensorwise_quant expects at least "
            f"{expected_num_tokens} input rows, got {input.shape[0]}"
        )
    if output.dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            "silu_mul_static_tensorwise_quant writes an FP8 buffer, got "
            f"{output.dtype}"
        )
    if output.shape != (int(expected_num_tokens), int(intermediate_size)):
        raise ValueError(
            "silu_mul_static_tensorwise_quant expects an output of shape "
            f"({int(expected_num_tokens)}, {int(intermediate_size)}), got "
            f"{tuple(output.shape)}"
        )
    if output.device != input.device:
        raise ValueError(
            "silu_mul_static_tensorwise_quant needs the output on the same "
            "device as the input"
        )
    if scale.numel() != 1:
        raise ValueError(
            "silu_mul_static_tensorwise_quant expects a single static scale, "
            f"got {scale.numel()} elements"
        )
    if scale.device != input.device:
        raise ValueError(
            "silu_mul_static_tensorwise_quant needs the scale on the same "
            "device as the input"
        )
    if num_tokens_tensor.numel() != 1:
        raise ValueError(
            "silu_mul_static_tensorwise_quant expects a one-element token "
            f"count, got {num_tokens_tensor.numel()} elements"
        )
    if num_tokens_tensor.dtype != torch.int32:
        raise ValueError(
            "silu_mul_static_tensorwise_quant reads the token count as int32 "
            f"inside the kernel, got {num_tokens_tensor.dtype}"
        )
    if num_tokens_tensor.device != input.device:
        raise ValueError(
            "silu_mul_static_tensorwise_quant needs the token count on the "
            "device, not on the host"
        )


def silu_mul_static_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    """Fused gated SiLU plus static tensorwise FP8 quantisation.

    Args:
        input: ``(rows, 2 * intermediate_size)`` activation buffer. The first
            ``intermediate_size`` columns of each row are the gate half and the
            next ``intermediate_size`` are the up half.
        output: ``(expected_num_tokens, intermediate_size)`` FP8 buffer whose
            current contents are the tail that must survive.
        scale: one-element static scale to divide by before the cast.
        num_tokens_tensor: one-element int32 device tensor holding the number
            of active rows.
        expected_num_tokens: row capacity of ``output``.
        intermediate_size: width of each half.

    Returns:
        A copy of ``output`` whose first ``num_tokens`` rows hold
        ``silu(gate) * up / scale`` cast to the output dtype. Rows at or beyond
        the token count are left exactly as the caller had them, and the caller
        buffer itself is never written.
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

    capacity = int(expected_num_tokens) * int(intermediate_size)
    if capacity == 0:
        return quantized

    programs = min(
        triton.cdiv(capacity, _BLOCK),
        _MAX_PROGRAMS,
    )
    _silu_mul_static_tensorwise_quant_kernel[(programs,)](
        input,
        quantized,
        scale,
        num_tokens_tensor,
        input.stride(0),
        int(intermediate_size),
        capacity,
        _BLOCK,
        _NUM_STAGES,
        num_warps=8,
        num_stages=1,
    )
    return quantized
