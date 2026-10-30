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

"""kda_gate_chunk_cumsum implementation for AMD."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kda_gate_chunk_cumsum_rounded_units(x):
    z = tl.minimum(x, -87.0).to(tl.float64)
    units = libdevice.exp(z) * 7.1362384635298e44
    return libdevice.floor(units + 0.5).to(tl.float32)


@triton.jit
def _kda_gate_chunk_cumsum_kernel(
    gate_ptr, a_log_ptr, bias_ptr, out_ptr, METADATA: tl.constexpr
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    D: tl.constexpr = METADATA[2]
    CHUNK_SIZE: tl.constexpr = METADATA[3]
    SCALE: tl.constexpr = METADATA[4]
    LOWER: tl.constexpr = METADATA[5]
    BLOCK_CHANNELS: tl.constexpr = METADATA[6]
    BATCH_STRIDE: tl.constexpr = NUM_TOKENS * NUM_HEADS * D
    TOKEN_STRIDE: tl.constexpr = NUM_HEADS * D
    HEAD_STRIDE: tl.constexpr = D
    CHANNEL_STRIDE: tl.constexpr = 1
    A_LOG_STRIDE: tl.constexpr = 1
    NUM_CHUNKS: tl.constexpr = (NUM_TOKENS + CHUNK_SIZE - 1) // CHUNK_SIZE
    NUM_CHANNEL_BLOCKS: tl.constexpr = (
        D + BLOCK_CHANNELS - 1
    ) // BLOCK_CHANNELS
    pid = tl.program_id(0)
    channels = pid % NUM_CHANNEL_BLOCKS * BLOCK_CHANNELS + tl.arange(
        0, BLOCK_CHANNELS
    )
    head_idx = pid // NUM_CHANNEL_BLOCKS % NUM_HEADS
    chunk = pid // (NUM_CHANNEL_BLOCKS * NUM_HEADS) % NUM_CHUNKS
    batch_idx = tl.program_id(1)
    tokens = chunk * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    decay = tl.exp(tl.load(a_log_ptr + head_idx * A_LOG_STRIDE).to(tl.float32))
    x = tl.load(
        gate_ptr
        + batch_idx * BATCH_STRIDE
        + tokens[:, None] * TOKEN_STRIDE
        + head_idx * HEAD_STRIDE
        + channels[None, :] * CHANNEL_STRIDE,
        (tokens[:, None] < NUM_TOKENS) & (channels[None, :] < D),
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + (head_idx * D + channels), channels < D, other=0
        ).to(tl.float32)
        x = x + bias[None, :]
    if LOWER is None and decay > 3.4028234663852886e38:
        gate = -decay * (x >= -103.97207641601562).to(tl.float32)
    elif LOWER is None:
        exp_neg_abs = tl.exp(-tl.abs(x))
        poly = 0.005039133597
        poly = poly * exp_neg_abs - 0.02861748822
        poly = poly * exp_neg_abs + 0.07656209171
        poly = poly * exp_neg_abs - 0.1349907219
        poly = poly * exp_neg_abs + 0.190145731
        poly = poly * exp_neg_abs - 0.2481442839
        poly = poly * exp_neg_abs + 0.3331445158
        poly = poly * exp_neg_abs - 0.4999916553
        poly = poly * exp_neg_abs + 0.9999998808
        softplus = tl.maximum(x, 0.0) + exp_neg_abs * poly
        gate = -decay * softplus
        EXACT_TAIL: tl.constexpr = CHUNK_SIZE > 128 or (
            SCALE is not None and (SCALE > 1.0 or SCALE < -1.0)
        )
        if EXACT_TAIL:
            units = _kda_gate_chunk_cumsum_rounded_units(x)
            tail = -(decay * 1.1754943508222875e-38) * (
                units * 1.1920928955078125e-07
            )
            gate = tl.where(x < -87.0, tail, gate)
    else:
        gate = LOWER / (1.0 + tl.exp(-(decay * x)))
    gate = tl.where(tokens[:, None] < NUM_TOKENS, gate, 0.0)
    out_values = tl.cumsum(gate, axis=0)
    if SCALE is not None:
        out_values = out_values * SCALE
    offset = (
        (batch_idx * NUM_TOKENS + tokens[:, None]) * NUM_HEADS + head_idx
    ) * D + channels[None, :]
    tl.store(
        out_ptr + offset,
        out_values,
        (tokens[:, None] < NUM_TOKENS) & (channels[None, :] < D),
    )


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    (batch, num_tokens, num_heads, hidden_size) = g.shape
    out = torch.empty(
        (batch, num_tokens, num_heads, hidden_size),
        device=g.device,
        dtype=torch.float32,
    )
    if batch * num_tokens * num_heads * hidden_size == 0:
        return out
    g = g.contiguous()
    A_log = A_log.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous()
    block_channels = min(
        64 if num_tokens * num_heads * hidden_size >= 4194304 else 32,
        max(1, 8192 // chunk_size),
    )
    grid = (
        num_heads
        * triton.cdiv(num_tokens, chunk_size)
        * triton.cdiv(hidden_size, block_channels),
        batch,
    )
    _kda_gate_chunk_cumsum_kernel[grid](
        g,
        A_log,
        dt_bias,
        out,
        (
            num_tokens,
            num_heads,
            hidden_size,
            chunk_size,
            scale,
            lower_bound,
            block_channels,
        ),
        num_warps=4,
        enable_fp_fusion=True,
    )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
