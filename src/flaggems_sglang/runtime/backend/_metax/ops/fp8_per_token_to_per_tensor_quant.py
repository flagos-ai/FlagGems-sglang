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


@triton.jit
def _enc_lane(val, L: tl.constexpr):
    b = val.to(tl.uint32, bitcast=True)
    t = tl.minimum((b + 0x80000) >> 20, 0x7F)
    return t << (L * 8)


@triton.jit
def _requant_contig_lean_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    y_ptr,
    T: tl.constexpr,
    KW: tl.constexpr,
):
    row = tl.program_id(0)
    e = row // T
    t = row - e * T
    n = tl.load(m_ptr + e)
    xw = x_ptr.to(tl.pointer_type(tl.uint32))
    yw = y_ptr.to(tl.pointer_type(tl.uint32))
    inv = 1.0 / tl.load(os_ptr).to(tl.float32)
    mval = tl.where(t < n, 0x07F00000, 0)
    msign = tl.where(t < n, 0x80808080, 0)
    sbase = e * (T * (KW >> 5)) + t * (KW >> 5)
    FBW: tl.constexpr = 512
    FG: tl.constexpr = FBW // 32
    C: tl.constexpr = KW // FBW
    for c in tl.static_range(C):
        w = tl.load(xw + row * KW + c * FBW + tl.arange(0, FBW))
        sv = tl.load(s_ptr + sbase + c * FG + tl.arange(0, FG)) * inv
        s_inv = tl.reshape(
            tl.broadcast_to(tl.expand_dims(sv, 1), (FG, 32)), (FBW,)
        )
        d0 = ((w << 20) & mval).to(tl.float32, bitcast=True)
        d1 = ((w << 12) & mval).to(tl.float32, bitcast=True)
        d2 = ((w << 4) & mval).to(tl.float32, bitcast=True)
        d3 = ((w >> 4) & mval).to(tl.float32, bitcast=True)
        out = (
            (w & msign)
            | _enc_lane(d0 * s_inv, 0)
            | _enc_lane(d1 * s_inv, 1)
            | _enc_lane(d2 * s_inv, 2)
            | _enc_lane(d3 * s_inv, 3)
        )
        tl.store(yw + row * KW + c * FBW + tl.arange(0, FBW), out)


@triton.jit
def _requant_contig_words_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    y_ptr,
    T,
    KW,
    NG,
    ss_e,
    ss_t,
    sm_e,
    NBW: tl.constexpr,
    FBW: tl.constexpr,
    FG: tl.constexpr,
    NEED_MASK: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NBW
    i = pid - row * NBW
    if INT64:
        row = row.to(tl.int64)
    e = row // T
    t = row - e * T
    n = tl.load(m_ptr + e * sm_e)
    xw = x_ptr.to(tl.pointer_type(tl.uint32))
    yw = y_ptr.to(tl.pointer_type(tl.uint32))
    w4 = i * FBW + tl.arange(0, FBW)
    if t < n:
        inv = 1.0 / tl.load(os_ptr).to(tl.float32)
        xrow = xw + row * KW
        yrow = yw + row * KW
        srow = s_ptr + e * ss_e + t * ss_t
        if NEED_MASK:
            mkw = w4 < KW
            w = tl.load(xrow + w4, mask=mkw, other=0)
        else:
            w = tl.load(xrow + w4)
        g = i * FG + tl.arange(0, FG)
        if NEED_MASK:
            sv = tl.load(srow + g, mask=g < NG, other=0.0) * inv
        else:
            sv = tl.load(srow + g) * inv
        s_inv = tl.reshape(
            tl.broadcast_to(tl.expand_dims(sv, 1), (FG, 32)), (FBW,)
        )
        d0 = ((w << 20) & 0x07F00000).to(tl.float32, bitcast=True)
        d1 = ((w << 12) & 0x07F00000).to(tl.float32, bitcast=True)
        d2 = ((w << 4) & 0x07F00000).to(tl.float32, bitcast=True)
        d3 = ((w >> 4) & 0x07F00000).to(tl.float32, bitcast=True)
        out = (
            (w & 0x80808080)
            | _enc_lane(d0 * s_inv, 0)
            | _enc_lane(d1 * s_inv, 1)
            | _enc_lane(d2 * s_inv, 2)
            | _enc_lane(d3 * s_inv, 3)
        )
        if NEED_MASK:
            tl.store(yrow + w4, out, mask=mkw)
        else:
            tl.store(yrow + w4, out)
    else:
        z = tl.zeros((FBW,), dtype=tl.uint32)
        if NEED_MASK:
            tl.store(yw + row * KW + w4, z, mask=w4 < KW)
        else:
            tl.store(yw + row * KW + w4, z)


@triton.jit
def _requant_row_words_kernel(
    xw_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    yw_ptr,
    NBW,
    K,
    KW,
    sxw_e,
    sxw_t,
    ss_e,
    ss_t,
    ss_g,
    sm_e,
    syw_e,
    syw_t,
    FBW: tl.constexpr,
    NEED_MASK: tl.constexpr,
    INT64: tl.constexpr,
):
    e = tl.program_id(0)
    t = tl.program_id(1)
    if INT64:
        e = e.to(tl.int64)
        t = t.to(tl.int64)
    n = tl.load(m_ptr + e * sm_e)
    xrow = xw_ptr + e * sxw_e + t * sxw_t
    srow = s_ptr + e * ss_e + t * ss_t
    yrow = yw_ptr + e * syw_e + t * syw_t
    if t < n:
        inv = 1.0 / tl.load(os_ptr).to(tl.float32)
        for i in range(NBW):
            w4 = i * FBW + tl.arange(0, FBW)
            if NEED_MASK:
                mkw = w4 < KW
                w = tl.load(xrow + w4, mask=mkw, other=0)
            else:
                w = tl.load(xrow + w4)
            g = w4 >> 5
            if NEED_MASK:
                sv = tl.load(srow + g * ss_g, mask=g < (KW >> 5), other=0.0)
            else:
                sv = tl.load(srow + g * ss_g)
            s_inv = sv * inv
            d0 = ((w << 20) & 0x07F00000).to(tl.float32, bitcast=True)
            d1 = ((w << 12) & 0x07F00000).to(tl.float32, bitcast=True)
            d2 = ((w << 4) & 0x07F00000).to(tl.float32, bitcast=True)
            d3 = ((w >> 4) & 0x07F00000).to(tl.float32, bitcast=True)
            out = (
                (w & 0x80808080)
                | _enc_lane(d0 * s_inv, 0)
                | _enc_lane(d1 * s_inv, 1)
                | _enc_lane(d2 * s_inv, 2)
                | _enc_lane(d3 * s_inv, 3)
            )
            if NEED_MASK:
                tl.store(yrow + w4, out, mask=mkw)
            else:
                tl.store(yrow + w4, out)
    else:
        z = tl.zeros((FBW,), dtype=tl.uint32)
        for i in range(NBW):
            w4 = i * FBW + tl.arange(0, FBW)
            if NEED_MASK:
                tl.store(yrow + w4, z, mask=w4 < KW)
            else:
                tl.store(yrow + w4, z)


@triton.jit
def _fp8e4m3_decode2(u8):
    u = u8
    sgn = (u & 0x80) << 24
    p = u & 0x7F
    return ((p << 20) | sgn).to(tl.float32, bitcast=True)


@triton.jit
def _fp8e4m3_encode2(val2):
    u32 = val2.to(tl.uint32, bitcast=True)
    mag = u32 & 0x7FFFFFFF
    r = mag + 0x7FFFF + ((mag >> 20) & 1)
    t = r >> 20
    res = tl.minimum(t, 0x7F)
    return (res | ((u32 >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _requant_row_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    y_ptr,
    NB,
    K,
    sx_e,
    sx_t,
    sx_k,
    ss_e,
    ss_t,
    ss_g,
    sm_e,
    sy_e,
    sy_t,
    sy_k,
    FB: tl.constexpr,
    NEED_MASK: tl.constexpr,
    INT64: tl.constexpr,
):
    e = tl.program_id(0)
    t = tl.program_id(1)
    if INT64:
        e = e.to(tl.int64)
        t = t.to(tl.int64)
    n = tl.load(m_ptr + e * sm_e)
    xrow = x_ptr + e * sx_e + t * sx_t
    srow = s_ptr + e * ss_e + t * ss_t
    yrow = y_ptr + e * sy_e + t * sy_t
    if t < n:
        inv = 1.0 / tl.load(os_ptr).to(tl.float32)
        for i in range(NB):
            offs = i * FB + tl.arange(0, FB)
            if NEED_MASK:
                mk = offs < K
                u8 = tl.load(xrow + offs * sx_k, mask=mk, other=0.0).to(
                    tl.uint8, bitcast=True
                )
                sv = tl.load(srow + (offs >> 7) * ss_g, mask=mk, other=0.0)
            else:
                u8 = tl.load(xrow + offs * sx_k).to(tl.uint8, bitcast=True)
                sv = tl.load(srow + (offs >> 7) * ss_g)
            val2 = _fp8e4m3_decode2(u8.to(tl.uint32)) * sv * inv
            out = _fp8e4m3_encode2(val2)
            if NEED_MASK:
                tl.store(
                    yrow + offs * sy_k,
                    out.to(tl.float8e4nv, bitcast=True),
                    mask=mk,
                )
            else:
                tl.store(
                    yrow + offs * sy_k, out.to(tl.float8e4nv, bitcast=True)
                )
    else:
        z = tl.zeros((FB,), dtype=tl.float8e4nv)
        for i in range(NB):
            offs = i * FB + tl.arange(0, FB)
            if NEED_MASK:
                tl.store(yrow + offs * sy_k, z, mask=offs < K)
            else:
                tl.store(yrow + offs * sy_k, z)


def _word_path_ok(K, xs_k, xs_e, xs_t, ys_e, ys_t, x_ptr):
    return (
        K % 4 == 0
        and xs_k == 1
        and x_ptr % 4 == 0
        and xs_t % 4 == 0
        and xs_e % 4 == 0
        and ys_t % 4 == 0
        and ys_e % 4 == 0
    )


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    E, T, K = x.shape
    if (
        x.is_contiguous()
        and K % 2048 == 0
        and E * T * K < 2**31
        and x_scale.is_contiguous()
        and x.data_ptr() % 4 == 0
    ):
        y = torch.empty_like(x)
        _requant_contig_lean_kernel[(E * T,)](
            x,
            x_scale,
            masked_m,
            output_scale,
            y,
            T,
            K >> 2,
            num_warps=(1 if E * T < 8192 else 2),
        )
        return y
    y = torch.empty((E, T, K), dtype=x.dtype, device=x.device)
    if E == 0 or T == 0 or K == 0:
        return y
    KW = K // 4
    xs_e, xs_t = x.stride(0), x.stride(1)
    ys_e, ys_t = y.stride(0), y.stride(1)
    int64 = E * T * K >= 2**31
    if (
        x.stride(2) == 1
        and xs_t == K
        and xs_e == T * K
        and ys_t == K
        and ys_e == T * K
        and x_scale.stride(2) == 1
        and K >= 128
        and K % 4 == 0
        and x.data_ptr() % 4 == 0
    ):
        NG = (KW + 31) // 32
        fbw = min(1024, triton.next_power_of_2(KW))
        nbw = (KW + fbw - 1) // fbw
        _requant_contig_words_kernel[(E * T * nbw,)](
            x,
            x_scale,
            masked_m,
            output_scale,
            y,
            T,
            KW,
            NG,
            x_scale.stride(0),
            x_scale.stride(1),
            masked_m.stride(0),
            NBW=nbw,
            FBW=fbw,
            FG=fbw // 32,
            NEED_MASK=(K % (fbw * 4) != 0),
            INT64=int64,
            num_warps=2,
        )
        return y
    if _word_path_ok(K, x.stride(2), xs_e, xs_t, ys_e, ys_t, x.data_ptr()):
        fbw = min(1024, max(1, triton.next_power_of_2(KW)))
        nbw = (KW + fbw - 1) // fbw
        _requant_row_words_kernel[(E, T)](
            x.view(torch.uint8).view(torch.uint32),
            x_scale,
            masked_m,
            output_scale,
            y.view(torch.uint8).view(torch.uint32),
            nbw,
            K,
            KW,
            xs_e // 4,
            xs_t // 4,
            x_scale.stride(0),
            x_scale.stride(1),
            x_scale.stride(2),
            masked_m.stride(0),
            y.stride(0) // 4,
            y.stride(1) // 4,
            FBW=fbw,
            NEED_MASK=(K % (fbw * 4) != 0),
            INT64=int64,
            num_warps=2,
        )
    else:
        fb = min(4096, triton.next_power_of_2(K))
        nb = (K + fb - 1) // fb
        _requant_row_kernel[(E, T)](
            x,
            x_scale,
            masked_m,
            output_scale,
            y,
            nb,
            K,
            xs_e,
            xs_t,
            x.stride(2),
            x_scale.stride(0),
            x_scale.stride(1),
            x_scale.stride(2),
            masked_m.stride(0),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            FB=fb,
            NEED_MASK=(K % fb != 0),
            INT64=int64,
            num_warps=2,
        )
    return y


__all__ = ["fp8_per_token_to_per_tensor_quant"]
