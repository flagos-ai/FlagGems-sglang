# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Fused MoE gated activation: ``act(gate) * up`` over a packed 2-D input.

For ``gateup_output`` of shape ``[M, 2H]`` the kernel computes

    gate = gateup_output[:, :H]        up = gateup_output[:, H:]
    out  = act(clamp(gate, max=lim)) * clamp(up, -lim, lim)

with ``act`` either SiLU or tanh-approximated GELU, and the optional SwiGLU
clamp active only when ``swiglu_limit`` is given.

The reference path is a chain of elementwise kernels — two fp32 casts, the
clamp, the activation, a cast back to the input dtype and the multiply —
and each of them streams the whole ``[M, H]`` slab again, so the unfused
form moves roughly 5-7x the minimum traffic. This kernel reads each half
once and writes the result once, which is what the memory-bound prefill
sizes pay for. Activation and clamp are evaluated in fp32, then both
operands are cast back to the input dtype before multiplying, matching the
reference's rounding exactly.
"""

import torch
import triton
import triton.language as tl

# Widest tile per CTA along the halved last dim. Mirrors the tile width
# used by the sibling ``silu_and_mul`` kernel: a wide contiguous tile keeps
# the loads/stores vectorizable, while 16384 overflows the unified buffer
# once multi-buffering is enabled.
_MAX_BLOCK_SIZE = 8192


@triton.jit
def _act_and_mul_kernel(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offs = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offs < d

    # int64 row base keeps the offset product from overflowing on large
    # packed buffers; the lane offsets stay int32.
    base = x_ptr + row_idx.to(tl.int64) * (2 * d)
    gate_raw = tl.load(base + col_offs, mask=mask, other=0.0)
    up_raw = tl.load(base + d + col_offs, mask=mask, other=0.0)
    gate = gate_raw.to(tl.float32)
    up = up_raw.to(tl.float32)

    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)

    if ACTIVATION == 0:
        # silu(x) = x / (1 + exp(-x)): one exp and one division.
        act = tl.fdiv(gate, 1.0 + tl.exp(-gate))
    else:
        # gelu_tanh(x) = x * sigmoid(2*z) because 0.5*(1+tanh(z)) ==
        # sigmoid(2z); this keeps the backend-independent exp path.
        # Constants inlined: Triton kernels cannot read module globals.
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        act = tl.fdiv(gate, 1.0 + tl.exp(-2.0 * inner))

    out_base = out_ptr + row_idx.to(tl.int64) * d
    tl.store(out_base + col_offs, tl.cast(act, up_raw.dtype) * up_raw, mask=mask)


def act_and_mul(gateup_output, activation="silu", swiglu_limit=None):
    """Fused gated activation over ``gateup_output`` of shape ``[M, 2H]``.

    Args:
        gateup_output: 2-D tensor ``[M, 2H]``; ``gate`` is the first half
            along the last dim and ``up`` the second half.
        activation: ``"silu"`` or ``"gelu"`` (tanh approximation).
        swiglu_limit: optional float clamp evaluated in fp32 (``gate <=
            limit``, ``|up| <= limit``) before the activation.

    Returns:
        ``[M, H]`` tensor with the same dtype and device as the input.
    """
    if gateup_output.dim() != 2:
        raise ValueError(
            "act_and_mul expects a 2-D [M, 2H] tensor, got shape "
            f"{tuple(gateup_output.shape)}"
        )
    if activation not in ("silu", "gelu"):
        raise ValueError(f"Unsupported activation: {activation}")

    shape = gateup_output.shape
    d = shape[-1] // 2
    if d == 0 or shape[0] == 0:
        return gateup_output.new_empty((shape[0], d))

    x = gateup_output.contiguous()
    out = x.new_empty((shape[0], d))

    block_size = min(_MAX_BLOCK_SIZE, 1 << (d - 1).bit_length())
    n_col_blocks = (d + block_size - 1) // block_size

    _act_and_mul_kernel[(shape[0], n_col_blocks)](
        x,
        out,
        d=d,
        BLOCK_SIZE=block_size,
        ACTIVATION=0 if activation == "silu" else 1,
        HAS_LIMIT=swiglu_limit is not None,
        LIMIT=float(swiglu_limit) if swiglu_limit is not None else 0.0,
    )
    return out


__all__ = ["act_and_mul"]
