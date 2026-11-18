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

_SCALE_GROUP = 128

_PROD_CLAMP = tl.constexpr(1.9365234375)


def _pick_bg(ng):
    for bg in (32, 16, 8, 4, 2, 1):
        if ng % bg == 0:
            return bg
    return 1


@triton.jit
def _requant_flat_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    out_ptr,
    stride_me,
    T,
    NG,
    WIDE: tl.constexpr,
    BG: tl.constexpr,
):
    row = tl.program_id(0)
    pid_g = tl.program_id(1)
    e = row // T
    t = row - e * T
    n = tl.load(m_ptr + e * stride_me).to(tl.int32)
    inv = 1.0 / tl.load(os_ptr)

    TILE: tl.constexpr = BG * 128
    j = tl.arange(0, TILE)
    k0 = pid_g * TILE
    g = pid_g * BG + tl.arange(0, BG)
    if WIDE:
        base = row.to(tl.int64) * (NG * 128)
        soff = row.to(tl.int64) * NG + g
    else:
        base = row * (NG * 128)
        soff = row * NG + g

    if t < n:
        xb = tl.load(x_ptr + base + k0 + j)
        s = tl.load(s_ptr + soff)
    else:
        xb = tl.zeros((TILE,), dtype=x_ptr.dtype.element_ty)
        s = tl.zeros((BG,), dtype=tl.float32)

    c16 = (s * inv).to(tl.float16)
    b = xb.to(tl.uint8, bitcast=True).to(tl.int32)
    b16 = ((b & 0x7F) << 7).to(tl.uint16)
    f = b16.to(tl.float16, bitcast=True)

    CL = tl.full((), _PROD_CLAMP.value, tl.float16)
    prod2 = tl.minimum(tl.reshape(f, (BG, 128)) * c16[:, None], CL)
    prod = tl.reshape(prod2, (TILE,))

    p = prod.to(tl.uint16, bitcast=True).to(tl.int32)
    q = (p + 0x40) >> 7
    bits = (q | (b & 0x80)).to(tl.uint8)
    o = bits.to(out_ptr.dtype.element_ty, bitcast=True)
    tl.store(out_ptr + base + k0 + j, o)


@triton.jit
def _requant_strided_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    out_ptr,
    stride_xe,
    stride_xt,
    stride_xk,
    stride_se,
    stride_st,
    stride_sg,
    stride_me,
    stride_oe,
    stride_ot,
    stride_ok,
    T,
    NG,
    BG: tl.constexpr,
):
    pid_g = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.program_id(2)

    n = tl.load(m_ptr + e * stride_me).to(tl.int32)
    inv = 1.0 / tl.load(os_ptr)
    live = t < n

    x_row = x_ptr + e.to(tl.int64) * stride_xe + t.to(tl.int64) * stride_xt
    s_row = s_ptr + e.to(tl.int64) * stride_se + t.to(tl.int64) * stride_st
    o_row = out_ptr + e.to(tl.int64) * stride_oe + t.to(tl.int64) * stride_ot

    offs_g = pid_g * BG + tl.arange(0, BG)
    g_in = offs_g < NG
    offs_i = tl.arange(0, 128)
    f8ty = out_ptr.dtype.element_ty
    CL = tl.full((), _PROD_CLAMP.value, tl.float16)

    s = tl.load(
        s_row + offs_g[:, None] * stride_sg,
        mask=g_in[:, None] & live,
        other=0.0,
    )
    c16 = (s * inv).to(tl.float16)

    k = offs_g[:, None] * 128 + offs_i[None, :]
    xb = tl.load(
        x_row + k * stride_xk,
        mask=g_in[:, None] & live,
        other=0.0,
    )
    b = xb.to(tl.uint8, bitcast=True).to(tl.int32)
    b16 = ((b & 0x7F) << 7).to(tl.uint16)
    f = b16.to(tl.float16, bitcast=True)

    prod = tl.minimum(f * c16, CL)

    p = prod.to(tl.uint16, bitcast=True).to(tl.int32)
    q = (p + 0x40) >> 7
    bits = (q | (b & 0x80)).to(tl.uint8)
    o = bits.to(f8ty, bitcast=True)
    tl.store(o_row + k * stride_ok, o, mask=g_in[:, None])


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    E, T, K = x.shape
    NG = K // _SCALE_GROUP
    out = torch.empty((E, T, K), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    x_packed = x.stride(2) == 1 and x.stride(1) == K and x.stride(0) == T * K
    s_packed = (
        x_scale.stride(2) == 1
        and x_scale.stride(1) == NG
        and x_scale.stride(0) == T * NG
    )

    if x_packed and s_packed:
        BG = _pick_bg(NG)
        _requant_flat_kernel[(E * T, triton.cdiv(NG, BG))](
            x,
            x_scale,
            masked_m,
            output_scale,
            out,
            masked_m.stride(0),
            T,
            NG,
            WIDE=E * T * K >= 2**31,
            BG=BG,
            num_warps=2 if E * T <= 1024 else 4,
        )
    else:
        _requant_strided_kernel[(triton.cdiv(NG, 16), T, E)](
            x,
            x_scale,
            masked_m,
            output_scale,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x_scale.stride(0),
            x_scale.stride(1),
            x_scale.stride(2),
            masked_m.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            T,
            NG,
            BG=16,
            num_warps=1,
        )
    return out


__all__ = ["fp8_per_token_to_per_tensor_quant"]
