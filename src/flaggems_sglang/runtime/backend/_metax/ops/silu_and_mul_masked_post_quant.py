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

_LOG2E = tl.constexpr(1.4426950408889634)
_FP8_MAX = tl.constexpr(448.0)
_INV_FP8_MAX = tl.constexpr(1.0 / 448.0)
_ABS_MIN = tl.constexpr(1e-10)
_EXP2_ARG_MAX = tl.constexpr(80.0)
_RCP_MAGIC = tl.constexpr(0x7EF30000)
_ENC_BIAS = tl.constexpr(0x3BF80000)
_FP8_FIELD_MAX = tl.constexpr(0x7E)
_BLOCK = 1024


@triton.jit
def _encode_fp8e4m3(q):
    u = q.to(tl.int32, bitcast=True)
    sign = (u >> 24) & 0x80
    field = ((u & 0x7FFFFFFF) - _ENC_BIAS) >> 20
    field = tl.minimum(tl.maximum(field, 0), _FP8_FIELD_MAX)
    return (sign | field).to(tl.uint8).to(tl.int8, bitcast=True)


@triton.jit
def _fast_kernel(
    in_ptr,
    masked_ptr,
    q_ptr,
    s_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    QGS: tl.constexpr,
    NG: tl.constexpr,
):
    G: tl.constexpr = H // QGS
    pid_c = tl.program_id(0)
    t = tl.program_id(1)
    pid_e = tl.program_id(2)
    n = tl.load(masked_ptr + pid_e)
    g = pid_c * NG + tl.arange(0, NG)
    c = tl.arange(0, QGS)
    ch = g[:, None] * QGS + c[None, :]
    in_row = in_ptr + pid_e * (T * 2 * H) + t * (2 * H)
    gate = tl.load(in_row + ch).to(tl.float32)
    up = tl.load(in_row + ch + H).to(tl.float32)
    e = tl.exp2(tl.minimum(gate * -_LOG2E, _EXP2_ARG_MAX))
    d = 1.0 + e
    r0 = (_RCP_MAGIC - d.to(tl.int32, bitcast=True)).to(
        tl.float32, bitcast=True
    )
    r1 = r0 * tl.fma(-d, r0, 2.0)
    inv = r1 * tl.fma(-d, r1, 2.0)
    val = gate * up * inv
    amax = tl.maximum(tl.max(tl.abs(val), axis=1), _ABS_MIN)
    enc = _encode_fp8e4m3(val * (_FP8_MAX / amax)[:, None])
    q_out = q_ptr.to(tl.pointer_type(tl.int8)) + pid_e * (T * H) + t * H
    s_out = s_ptr + pid_e * (T * G) + t * G + g
    if t < n:
        tl.store(q_out + ch, enc)
        tl.store(s_out, amax * _INV_FP8_MAX)
    else:
        tl.store(q_out + ch, tl.zeros((NG, QGS), tl.int8))
        tl.store(s_out, tl.zeros((NG,), tl.float32))


@triton.jit
def _gen_kernel(
    in_ptr,
    masked_ptr,
    q_ptr,
    s_ptr,
    sie: tl.constexpr,
    sit: tl.constexpr,
    sih: tl.constexpr,
    sqe: tl.constexpr,
    sqt: tl.constexpr,
    sse: tl.constexpr,
    sst: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    GS: tl.constexpr,
    QGS: tl.constexpr,
    MASKED_LANES: tl.constexpr,
    MASKED_GROUPS: tl.constexpr,
):
    pid_c = tl.program_id(0)
    t = tl.program_id(1)
    pid_e = tl.program_id(2)
    n = tl.load(masked_ptr + pid_e)
    valid = t < n
    g = pid_c * NG + tl.arange(0, NG)
    c = tl.arange(0, GS)
    ch = g[:, None] * QGS + c[None, :]
    if MASKED_LANES or MASKED_GROUPS:
        gmask = (c < QGS)[None, :]
        if MASKED_GROUPS:
            gmask = gmask & (g < G)[:, None]
    in_row = in_ptr + pid_e * sie + t * sit
    if MASKED_LANES or MASKED_GROUPS:
        gate = tl.load(in_row + ch * sih, mask=gmask, other=0.0)
        up = tl.load(in_row + ch * sih + H * sih, mask=gmask, other=0.0)
    else:
        gate = tl.load(in_row + ch * sih)
        up = tl.load(in_row + ch * sih + H * sih)
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    e = tl.exp2(tl.minimum(gate * -_LOG2E, _EXP2_ARG_MAX))
    d = 1.0 + e
    r0 = (_RCP_MAGIC - d.to(tl.int32, bitcast=True)).to(
        tl.float32, bitcast=True
    )
    r1 = r0 * tl.fma(-d, r0, 2.0)
    inv = r1 * tl.fma(-d, r1, 2.0)
    val = gate * up * inv
    amax = tl.maximum(tl.max(tl.abs(val), axis=1), _ABS_MIN)
    enc = _encode_fp8e4m3(val * (_FP8_MAX / amax)[:, None])
    q_out = q_ptr.to(tl.pointer_type(tl.int8)) + pid_e * sqe + t * sqt
    s_out = s_ptr + pid_e * sse + t * sst + g
    if valid:
        if MASKED_LANES or MASKED_GROUPS:
            tl.store(q_out + ch, enc, mask=gmask)
            if MASKED_GROUPS:
                tl.store(s_out, amax * _INV_FP8_MAX, mask=g < G)
            else:
                tl.store(s_out, amax * _INV_FP8_MAX)
        else:
            tl.store(q_out + ch, enc)
            tl.store(s_out, amax * _INV_FP8_MAX)
    else:
        if MASKED_LANES or MASKED_GROUPS:
            tl.store(q_out + ch, tl.zeros((NG, GS), tl.int8), mask=gmask)
            if MASKED_GROUPS:
                tl.store(s_out, tl.zeros((NG,), tl.float32), mask=g < G)
            else:
                tl.store(s_out, tl.zeros((NG,), tl.float32))
        else:
            tl.store(q_out + ch, tl.zeros((NG, GS), tl.int8))
            tl.store(s_out, tl.zeros((NG,), tl.float32))


def silu_and_mul_masked_post_quant(input, quant_group_size, masked_m):
    E = input.shape[0]
    T = input.shape[1]
    H = input.shape[2] >> 1
    G = H // quant_group_size
    out_q = torch.empty(
        (E, T, H), dtype=torch.float8_e4m3fn, device=input.device
    )
    out_s = torch.empty((E, T, G), dtype=torch.float32, device=input.device)
    if E == 0 or T == 0 or H == 0 or G == 0:
        return out_q, out_s
    gs = 1 << (quant_group_size - 1).bit_length()
    ng = _BLOCK // gs
    gp2 = 1 << (G - 1).bit_length()
    if ng > gp2:
        ng = gp2
    if gs == quant_group_size and G % ng == 0 and input.is_contiguous():
        _fast_kernel[((G // ng), T, E)](
            input,
            masked_m,
            out_q,
            out_s,
            T,
            H,
            quant_group_size,
            ng,
            num_warps=1,
            num_stages=1,
        )
    else:
        _gen_kernel[(((G + ng - 1) // ng), T, E)](
            input,
            masked_m,
            out_q,
            out_s,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            out_q.stride(0),
            out_q.stride(1),
            out_s.stride(0),
            out_s.stride(1),
            H,
            G,
            ng,
            gs,
            quant_group_size,
            gs != quant_group_size,
            G % ng != 0,
            num_warps=1,
            num_stages=1,
        )
    return out_q, out_s


__all__ = ["silu_and_mul_masked_post_quant"]
