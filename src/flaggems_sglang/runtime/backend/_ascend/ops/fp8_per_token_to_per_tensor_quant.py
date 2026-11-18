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

_GROUP = 128
_TILE_U32 = 256
_WARPS_U32 = 4
_STAGES_U32 = 2
_TILE = 64
_LOOP_W = 8
_WARPS_W = 1
_STAGES_W = 2


@triton.jit
def _requant_kernel(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    y_ptr,
    T,
    G,
    TILE: tl.constexpr,
    LOOP: tl.constexpr,
):
    pb = tl.program_id(0)
    e = tl.program_id(1)

    et = e.to(tl.int64) * T
    H = G * 64
    x_base = x_ptr + et * H
    s_base = s_ptr + et * G
    y_base = y_ptr + et * H

    n = tl.load(m_ptr + e)
    inv = 1.0 / tl.load(os_ptr)
    lanes = tl.arange(0, 64)
    for i in tl.static_range(LOOP):
        gs = (pb * LOOP + i) * TILE + tl.arange(0, TILE)
        off = gs[:, None] * 64 + lanes[None, :]
        gm = (gs < n * G)[:, None]
        pm = (gs < T * G)[:, None]
        x2 = tl.load(x_base + off, mask=gm, other=0)
        s2 = tl.load(s_base + gs, mask=gs < T * G, other=0.0)[:, None]
        lo = (x2 << 7) & 0x3F80
        hi = (x2 >> 1) & 0x3F80
        vlo = lo.to(tl.float16, bitcast=True)
        vhi = hi.to(tl.float16, bitcast=True)
        mlt = (s2 * inv * 256.0).to(tl.float16)
        wlo = vlo * mlt
        whi = vhi * mlt
        blo = wlo.to(tl.int16, bitcast=True)
        bhi = whi.to(tl.int16, bitcast=True)
        rlo = (blo - 8128) >> 7
        rhi = (bhi - 8128) >> 7
        clo = tl.minimum(tl.maximum(rlo, 0), 127)
        chi = tl.minimum(tl.maximum(rhi, 0), 127)
        sign = x2 & 0x8080
        y2 = ((clo | chi << 8) | sign).to(tl.uint16)
        tl.store(y_base + off, y2, mask=pm)


@triton.jit
def _requant_kernel_u32(
    x_ptr,
    s_ptr,
    m_ptr,
    os_ptr,
    y_ptr,
    T,
    G,
    TILE: tl.constexpr,
):
    pb = tl.program_id(0)
    e = tl.program_id(1)

    et = e.to(tl.int64) * T
    H = G * 32
    x_base = x_ptr + et * H
    s_base = s_ptr + et * G
    y_base = y_ptr + et * H

    n = tl.load(m_ptr + e)
    inv = 1.0 / tl.load(os_ptr)
    lanes = tl.arange(0, 32)
    gs = pb * TILE + tl.arange(0, TILE)
    off = gs[:, None] * 32 + lanes[None, :]
    gm = (gs < n * G)[:, None]
    pm = (gs < T * G)[:, None]
    w = tl.load(x_base + off, mask=gm, other=0)
    sgn = w & 0x80808080
    va = ((w & 0x7F) << 7).to(tl.uint16).to(tl.float16, bitcast=True)
    vb = ((w >> 1) & 0x3F80).to(tl.uint16).to(tl.float16, bitcast=True)
    vc = ((w >> 9) & 0x3F80).to(tl.uint16).to(tl.float16, bitcast=True)
    vd = ((w >> 17) & 0x3F80).to(tl.uint16).to(tl.float16, bitcast=True)
    s2 = tl.load(s_base + gs, mask=gs < T * G, other=0.0)[:, None]
    mlt = (s2 * inv * 256.0).to(tl.float16)
    ba = (va * mlt).to(tl.int16, bitcast=True)
    bb = (vb * mlt).to(tl.int16, bitcast=True)
    bc = (vc * mlt).to(tl.int16, bitcast=True)
    bd = (vd * mlt).to(tl.int16, bitcast=True)
    ca = tl.minimum(tl.maximum((ba - 8128) >> 7, 0), 127).to(tl.uint32)
    cb = tl.minimum(tl.maximum((bb - 8128) >> 7, 0), 127).to(tl.uint32)
    cc = tl.minimum(tl.maximum((bc - 8128) >> 7, 0), 127).to(tl.uint32)
    cd = tl.minimum(tl.maximum((bd - 8128) >> 7, 0), 127).to(tl.uint32)
    y2 = (ca | (cb << 8)) | ((cc << 16) | (cd << 24)) | sgn
    tl.store(y_base + off, y2, mask=pm)


def _dense(t):
    return t if t.is_contiguous() else t.contiguous()


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    E, T, K = x.shape
    G = K // _GROUP

    y = x.new_empty((E, T, K), dtype=torch.float8_e4m3fn)

    if E < 16:
        _requant_kernel_u32[(triton.cdiv(T * G, _TILE_U32), E)](
            _dense(x).view(torch.uint32),
            _dense(x_scale),
            _dense(masked_m),
            _dense(output_scale),
            y.view(torch.uint32),
            T,
            G,
            TILE=_TILE_U32,
            num_warps=_WARPS_U32,
            num_stages=_STAGES_U32,
        )
    else:
        _requant_kernel[(triton.cdiv(T * G, _TILE * _LOOP_W), E)](
            _dense(x).view(torch.uint16),
            _dense(x_scale),
            _dense(masked_m),
            _dense(output_scale),
            y.view(torch.uint16),
            T,
            G,
            TILE=_TILE,
            LOOP=_LOOP_W,
            num_warps=_WARPS_W,
            num_stages=_STAGES_W,
        )
    return y


__all__ = ["fp8_per_token_to_per_tensor_quant"]
