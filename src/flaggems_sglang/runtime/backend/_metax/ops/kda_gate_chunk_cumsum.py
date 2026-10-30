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

_LOG2E = tl.constexpr(1.4426950408889634)
_LN2 = tl.constexpr(0.6931471805599453)

_MIN_ELEMS_WIDE = 200_000
_MIN_ELEMS_MID = 3_000_000


def _pick_config(n_elems):
    """Pick (BLOCK_C, num_warps) for the given element count."""
    if n_elems < _MIN_ELEMS_WIDE:
        return 32, 8
    if n_elems < _MIN_ELEMS_MID:
        return 64, 4
    return 64, 2


@triton.jit
def _kda_gate_chunk_cumsum_kernel(
    g_ptr,
    a_log_ptr,
    bias_ptr,
    out_ptr,
    T,
    C,
    n_chunks,
    chunk_size,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_gs,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_os,
    scale,
    lower_bound,
    S: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_LB: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = pid_t // n_chunks
    chunk = pid_t % n_chunks

    t_local = tl.arange(0, CHUNK)
    # Guard the (never expected) case chunk_size < CHUNK.
    t_mask = (t_local < chunk_size) & ((chunk * CHUNK + t_local) < T)
    t = chunk * CHUNK + t_local

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C

    # Channel c maps to (h, s) = (c // S, c % S).
    in_off = (c // S) * stride_gh + (c % S) * stride_gs
    g_ptrs = g_ptr + b * stride_gb + t[:, None] * stride_gt + in_off[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    x = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        x = x + bias[None, :]

    # exp(a) via exp2(a * log2(e)); cheaper than tl.exp and within tolerance.
    a = tl.math.exp2(
        tl.load(a_log_ptr + c // S, mask=c_mask, other=0.0).to(tl.float32)
        * _LOG2E
    )
    if HAS_LB:
        # Safe gate variant: lower_bound * sigmoid(exp(a) * x).
        z = a[None, :] * x
        sig = 1.0 / (1.0 + tl.math.exp2(-z * _LOG2E))
        gate = lower_bound * sig
    else:
        e = tl.math.exp2(x * _LOG2E)
        sp = tl.where(x > 20.0, x, tl.math.log2(1.0 + e) * _LN2)
        gate = -a[None, :] * sp

    out = tl.cumsum(gate, axis=0)

    if HAS_SCALE:
        out = out * scale

    out_off = (c // S) * stride_oh + (c % S) * stride_os
    o_ptrs = (
        out_ptr + b * stride_ob + t[:, None] * stride_ot + out_off[None, :]
    )
    tl.store(o_ptrs, out, mask=mask)


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    c = h * s
    chunk_size = int(chunk_size)
    n_chunks = (t + chunk_size - 1) // chunk_size
    out = torch.empty((b, t, h, s), dtype=torch.float32, device=g.device)

    block_c, num_warps = _pick_config(b * t * h * s)

    # dim0 = channel slice (contiguous, fastest varying), dim1 = (b, chunk).
    grid = (triton.cdiv(c, block_c), b * n_chunks)

    _kda_gate_chunk_cumsum_kernel[grid](
        g,
        A_log,
        dt_bias if dt_bias is not None else g,
        out,
        t,
        c,
        n_chunks,
        chunk_size,
        g.stride(0),
        g.stride(1),
        g.stride(2),
        g.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        scale if scale is not None else 0.0,
        lower_bound if lower_bound is not None else 0.0,
        S=s,
        CHUNK=triton.next_power_of_2(chunk_size),
        BLOCK_C=block_c,
        HAS_BIAS=dt_bias is not None,
        HAS_SCALE=scale is not None,
        HAS_LB=lower_bound is not None,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
