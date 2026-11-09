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

"""Operator: fla/kda_gate_chunk_cumsum"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    # Matches torch.nn.functional.softplus with default beta=1, threshold=20.
    sp = tl.log(1.0 + tl.exp(x))
    return tl.where(x > 20.0, x, sp)


@triton.jit
def _kda_gate_chunk_cumsum_plain_kernel(
    g_ptr,  # [B, T, H, S]
    a_log_ptr,  # [H]
    out_ptr,  # [B, T, H, S] fp32
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,  # chunk_size (power of 2), == BLOCK_T
    BS: tl.constexpr,  # channel tile (power of 2, >= S)
    NEED_S_MASK: tl.constexpr,
):
    # batch*head-fastest: consecutive programs walk adjacent memory.
    pid = tl.program_id(0)
    bh = pid % (B * H)
    ck = pid // (B * H)
    h = bh % H
    b = bh // H

    ea = tl.exp(tl.load(a_log_ptr + h).to(tl.float32))
    head_off = h * S

    chunk_start = ck * CHUNK
    t_offs = chunk_start + tl.arange(0, CHUNK)
    t_mask = t_offs < T  # also covers the short tail chunk

    s_offs = tl.arange(0, BS)
    if NEED_S_MASK:
        s_mask = s_offs < S
        mask = t_mask[:, None] & s_mask[None, :]
    else:
        mask = t_mask[:, None]

    ptrs = (
        g_ptr
        + b * (T * H * S)
        + t_offs[:, None] * (H * S)
        + head_off
        + s_offs[None, :]
    )
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    gate = -ea * _softplus(x)
    # Zero out-of-range rows/channels so they don't pollute the cumsum.
    gate = tl.where(mask, gate, 0.0)

    # Single 2D cumsum over the whole chunk along time.
    y = tl.cumsum(gate, axis=0)

    out_ptrs = (
        out_ptr
        + b * (T * H * S)
        + t_offs[:, None] * (H * S)
        + head_off
        + s_offs[None, :]
    )
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _kda_gate_chunk_cumsum_kernel(
    g_ptr,  # [B, T, H, S]
    a_log_ptr,  # [H]
    dt_bias_ptr,  # [H*S] or unused
    out_ptr,  # [B, T, H, S] fp32
    scale,
    lower_bound,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,  # chunk_size (power of 2), == BLOCK_T
    BS: tl.constexpr,  # channel tile (power of 2, >= S)
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_LB: tl.constexpr,
    NEED_S_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid % (B * H)
    ck = pid // (B * H)
    h = bh % H
    b = bh // H

    ea = tl.exp(tl.load(a_log_ptr + h).to(tl.float32))
    head_off = h * S

    chunk_start = ck * CHUNK
    t_offs = chunk_start + tl.arange(0, CHUNK)
    t_mask = t_offs < T  # also covers the short tail chunk

    s_offs = tl.arange(0, BS)
    if NEED_S_MASK:
        s_mask = s_offs < S
        mask = t_mask[:, None] & s_mask[None, :]
    else:
        mask = t_mask[:, None]

    ptrs = (
        g_ptr
        + b * (T * H * S)
        + t_offs[:, None] * (H * S)
        + head_off
        + s_offs[None, :]
    )
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(
            dt_bias_ptr + head_off + s_offs,
            mask=(s_mask if NEED_S_MASK else (s_offs >= 0)),
            other=0.0,
        ).to(tl.float32)
        x = x + bias[None, :]

    if HAS_LB:
        gate = lower_bound * tl.sigmoid(ea * x)
    else:
        gate = -ea * _softplus(x)

    if HAS_SCALE:
        gate = gate * scale

    # Zero out-of-range rows/channels so they don't pollute the cumsum.
    gate = tl.where(mask, gate, 0.0)

    # Single 2D cumsum over the whole chunk along time.
    y = tl.cumsum(gate, axis=0)

    out_ptrs = (
        out_ptr
        + b * (T * H * S)
        + t_offs[:, None] * (H * S)
        + head_off
        + s_offs[None, :]
    )
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _kda_gate_chunk_cumsum_tiled_kernel(
    g_ptr,  # [B, T, H, S]
    a_log_ptr,  # [H]
    dt_bias_ptr,  # [H*S] or unused
    out_ptr,  # [B, T, H, S] fp32
    scale,
    lower_bound,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,  # chunk_size (power of 2)
    BT: tl.constexpr,  # time tile, divides CHUNK
    BS: tl.constexpr,  # channel tile (power of 2, >= S)
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_LB: tl.constexpr,
    NEED_S_MASK: tl.constexpr,
):
    # Fallback for chunk sizes larger than one time tile: running sum over
    # sequential time tiles within the chunk.
    pid = tl.program_id(0)
    bh = pid % (B * H)
    ck = pid // (B * H)
    h = bh % H
    b = bh // H

    ea = tl.exp(tl.load(a_log_ptr + h).to(tl.float32))
    head_off = h * S

    chunk_start = ck * CHUNK
    chunk_len = tl.minimum(CHUNK, T - chunk_start)

    s_offs = tl.arange(0, BS)
    if NEED_S_MASK:
        s_mask = s_offs < S
    else:
        s_mask = s_offs >= 0

    carry = tl.zeros([BS], dtype=tl.float32)
    for t0 in tl.static_range(0, CHUNK, BT):
        t_offs = chunk_start + t0 + tl.arange(0, BT)
        t_mask = (t0 + tl.arange(0, BT)) < chunk_len

        if NEED_S_MASK:
            mask = t_mask[:, None] & s_mask[None, :]
        else:
            mask = t_mask[:, None]

        ptrs = (
            g_ptr
            + b * (T * H * S)
            + t_offs[:, None] * (H * S)
            + head_off
            + s_offs[None, :]
        )
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        if HAS_BIAS:
            bias = tl.load(
                dt_bias_ptr + head_off + s_offs, mask=s_mask, other=0.0
            ).to(tl.float32)
            x = x + bias[None, :]

        if HAS_LB:
            gate = lower_bound * tl.sigmoid(ea * x)
        else:
            gate = -ea * _softplus(x)

        if HAS_SCALE:
            gate = gate * scale

        gate = tl.where(mask, gate, 0.0)

        y = tl.cumsum(gate, axis=0) + carry[None, :]
        carry += tl.sum(gate, axis=0)

        out_ptrs = (
            out_ptr
            + b * (T * H * S)
            + t_offs[:, None] * (H * S)
            + head_off
            + s_offs[None, :]
        )
        tl.store(out_ptrs, y, mask=mask)


# Chunk sizes up to this are handled by a single whole-chunk 2D cumsum.
_MAX_SINGLE_PASS_CHUNK = 128


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    out = torch.empty((b, t, h, s), dtype=torch.float32, device=g.device)

    BS = triton.next_power_of_2(s)
    need_s_mask = BS != s
    grid = (b * triton.cdiv(t, chunk_size) * h,)

    geom = dict(
        B=b, T=t, H=h, S=s, CHUNK=chunk_size, BS=BS, NEED_S_MASK=need_s_mask
    )

    if chunk_size <= _MAX_SINGLE_PASS_CHUNK:
        if dt_bias is None and scale is None and lower_bound is None:
            # Leanest path: only three pointers are passed at runtime.
            _kda_gate_chunk_cumsum_plain_kernel[grid](
                g,
                A_log,
                out,
                num_warps=4,
                num_stages=1,
                **geom,
            )
        else:
            _kda_gate_chunk_cumsum_kernel[grid](
                g,
                A_log,
                dt_bias if dt_bias is not None else g,
                out,
                1.0 if scale is None else scale,
                0.0 if lower_bound is None else lower_bound,
                HAS_BIAS=dt_bias is not None,
                HAS_SCALE=scale is not None,
                HAS_LB=lower_bound is not None,
                num_warps=4,
                num_stages=1,
                **geom,
            )
    else:
        _kda_gate_chunk_cumsum_tiled_kernel[grid](
            g,
            A_log,
            dt_bias if dt_bias is not None else g,
            out,
            1.0 if scale is None else scale,
            0.0 if lower_bound is None else lower_bound,
            BT=min(16, chunk_size),
            HAS_BIAS=dt_bias is not None,
            HAS_SCALE=scale is not None,
            HAS_LB=lower_bound is not None,
            num_warps=4,
            num_stages=1,
            **geom,
        )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
