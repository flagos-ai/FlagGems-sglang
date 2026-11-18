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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    mag = bits & 2147483647
    rounded = mag + 524287 + ((mag >> 20) & 1)
    normal = (rounded >> 20) - 960
    small = tl.abs(v) * 512.0
    lo = small.to(tl.int32)
    fraction = small - lo.to(tl.float32)
    subnormal = lo + (
        (fraction > 0.5) | ((fraction == 0.5) & ((lo & 1) != 0))
    ).to(tl.int32)
    code = tl.where(tl.abs(v) < 0.015625, subnormal, normal)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_non_nan(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = (raw >> 24) & 128
    mag = raw & 2147483647
    mag = tl.minimum(mag, 1138753536)
    normal = ((mag + 524287 + ((mag >> 20) & 1)) >> 20) - 960
    a = mag.to(tl.float32, bitcast=True)
    rounded = a + 16384.0
    sub = rounded.to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < 1015021568, sub, normal)
    return (code | sign).to(tl.uint8)


@triton.jit
def _quant(X, Q, S, G: tl.constexpr, R: tl.constexpr):
    col = tl.arange(0, 32)
    for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
        row = tile * R + tl.arange(0, R)
        x = tl.load(
            X + row[:, None] * 32 + col[None, :], row[:, None] < G, 0
        ).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-30)
        exponent = tl.minimum(
            tl.maximum(tl.ceil(tl.log2(maximum / 448.0)) + 127.0, 0.0), 254.0
        )
        scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        v = x / scale[:, None]
        v = tl.minimum(tl.maximum(v, -448.0), 448.0)
        tl.store(
            Q + row[:, None] * 32 + col[None, :], _encode(v), row[:, None] < G
        )
        tl.store(S + row, exponent.to(tl.uint8), row < G)


@triton.jit
def _quant_bf16(X, Q, S, G: tl.constexpr, R: tl.constexpr):
    xp = X.to(tl.pointer_type(tl.int16))
    offset = tl.program_id(0) * R * 32 + tl.arange(0, R * 32)
    mask = offset < G * 32
    raw = tl.load(xp + offset, mask, 0).to(tl.int32) & 65535
    raw = tl.reshape(raw, (R, 32))
    mag = raw & 32767
    amax = tl.max(mag, 1)
    exponent = tl.maximum(
        (amax >> 7) - 8 + ((amax & 127) > 96).to(tl.int32), 19
    )
    exponent = tl.where(amax == 32640, 254, exponent)
    inverse = ((255 - exponent) << 23).to(tl.float32, bitcast=True)
    x = (raw << 16).to(tl.float32, bitcast=True)
    q = _encode_non_nan(x * inverse[:, None] * 0.5)
    group_nan = amax > 32640
    nan_code = tl.where(raw > 65408, 255, 127)
    q = tl.where(group_nan[:, None], nan_code, q).to(tl.uint8)
    tl.store(Q + offset, tl.reshape(q, (R * 32,)), mask)
    row = tl.program_id(0) * R + tl.arange(0, R)
    tl.store(S + row, tl.where(group_nan, 0, exponent).to(tl.uint8), row < G)


@triton.jit
def _quant_pair(X, Q, S, G: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, 16)
    offset = row[:, None] * 16 + col[None, :]
    words = tl.load(X + offset, row[:, None] < G, 0).to(tl.uint32)
    r0 = (words & 65535).to(tl.int32)
    r1 = (words >> 16).to(tl.int32)
    mag = tl.maximum(r0 & 32767, r1 & 32767)
    amax = tl.max(mag, 1)
    exponent = tl.maximum((amax - 993) >> 7, 19)
    exponent = tl.where(amax == 32640, 254, exponent)
    nan = amax > 32640
    if G >= 131072:
        amin = tl.min(tl.minimum(r0 & 32767, r1 & 32767), 1)
        safe = tl.min((amin >= ((exponent - 6) << 7)) & (amax < 32640), 0)
    else:
        safe = False
    if safe:
        absolute = words & 2147450879
        rounded = absolute + 458759 + ((absolute >> 4) & 65537)
        normal = ((rounded >> 4) & 268374015) - (
            (exponent[:, None] * 8 - 56).to(tl.uint32) * 65537
        )
        signed = normal | ((words >> 8) & 8388736)
        packed = ((signed & 255) | ((signed >> 8) & 65280)).to(tl.uint16)
    else:
        inverse = ((255 - exponent) << 23).to(tl.float32, bitcast=True)
        x0 = (r0 << 16).to(tl.float32, bitcast=True)
        x1 = (r1 << 16).to(tl.float32, bitcast=True)
        q0 = _encode_non_nan(x0 * inverse[:, None] * 0.5)
        q1 = _encode_non_nan(x1 * inverse[:, None] * 0.5)
        nan = amax > 32640
        q0 = tl.where(nan[:, None], tl.where(r0 > 65408, 255, 127), q0)
        q1 = tl.where(nan[:, None], tl.where(r1 > 65408, 255, 127), q1)
        packed = q0.to(tl.uint16) | (q1.to(tl.uint16) << 8)
    tl.store(Q + offset, packed, row[:, None] < G)
    tl.store(S + row, tl.where(nan, 0, exponent).to(tl.uint8), row < G)


def to_mxfp8(x):
    source = x.contiguous()
    if x.shape[-1] % 32:
        raise ValueError("last dimension must be divisible by 32")
    q = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    scale = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 32), dtype=torch.uint8, device=x.device
    )
    groups = x.numel() // 32
    if groups:
        if (
            x.dtype == torch.bfloat16
            and source.data_ptr() % 4 == 0
            and q.data_ptr() % 2 == 0
        ):
            rows = 16 if groups <= 8192 else 32
            _quant_pair.run(
                source.view(torch.int32),
                q.view(torch.int16),
                scale,
                groups,
                rows,
                num_warps=1 if groups >= 131072 else 2,
                enable_fp_fusion=False,
                grid=(triton.cdiv(groups, rows),),
                warmup=False,
            )
            return q.view(torch.float8_e4m3fn), scale
        if x.dtype == torch.bfloat16:
            rows = 16 if groups <= 8192 else 64
            _quant_bf16.run(
                source,
                q,
                scale,
                groups,
                rows,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(groups, rows),),
                warmup=False,
            )
        else:
            _quant[(min(triton.cdiv(groups, 4), 4096),)](
                source,
                q,
                scale,
                groups,
                4,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return q.view(torch.float8_e4m3fn), scale


__all__ = ["to_mxfp8"]
