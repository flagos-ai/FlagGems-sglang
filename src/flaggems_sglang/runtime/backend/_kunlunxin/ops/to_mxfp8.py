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
def _encode_scaled(raw, exponent):
    mag = raw & 32767
    normal = ((mag + 7 + ((mag >> 4) & 1)) >> 4) - exponent * 8 + 56
    rebased = tl.maximum(mag - ((exponent - 127) << 7), 0)
    value = (rebased << 16).to(tl.float32, bitcast=True)
    sub = (value + 16384.0).to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < ((exponent - 6) << 7), sub, normal)
    code = tl.where(mag == 32640, 126, code)
    return (code | ((raw >> 8) & 128)).to(tl.uint8)


@triton.jit
def _max4(a0, a1, a2, a3, b0, b1, b2, b3):
    return (
        tl.maximum(a0, b0),
        tl.maximum(a1, b1),
        tl.maximum(a2, b2),
        tl.maximum(a3, b3),
    )


@triton.jit
def _scale(amax):
    exponent = tl.maximum(
        (amax >> 7) - 8 + ((amax & 127) > 96).to(tl.int32), 19
    )
    exponent = tl.where(amax == 32640, 254, exponent)
    return exponent, amax > 32640


@triton.jit
def _code(raw, exponent, nan):
    q = _encode_scaled(raw, exponent)
    return tl.where(nan, tl.where(raw > 65408, 255, 127), q).to(tl.uint32)


@triton.jit
def _quant_quad(X, Q, S, R: tl.constexpr):
    xp = X.to(tl.pointer_type(tl.uint64))
    qp = Q.to(tl.pointer_type(tl.uint32))
    sp = S.to(tl.pointer_type(tl.uint32))
    row = tl.program_id(0) * R + tl.arange(0, R)
    lane = tl.arange(0, 32)
    words = tl.load(xp + row[:, None] * 32 + lane[None, :])
    r0 = (words & 65535).to(tl.int32)
    r1 = ((words >> 16) & 65535).to(tl.int32)
    r2 = ((words >> 32) & 65535).to(tl.int32)
    r3 = ((words >> 48) & 65535).to(tl.int32)
    mag = tl.maximum(
        tl.maximum(r0 & 32767, r1 & 32767), tl.maximum(r2 & 32767, r3 & 32767)
    )
    group = tl.arange(0, 32) // 8
    a0, a1, a2, a3 = tl.reduce(
        (
            tl.where((group == 0)[None, :], mag, 0),
            tl.where((group == 1)[None, :], mag, 0),
            tl.where((group == 2)[None, :], mag, 0),
            tl.where((group == 3)[None, :], mag, 0),
        ),
        1,
        _max4,
    )
    s0, n0 = _scale(a0)
    s1, n1 = _scale(a1)
    s2, n2 = _scale(a2)
    s3, n3 = _scale(a3)
    raw_scales = (
        s0.to(tl.uint32)
        | (s1.to(tl.uint32) << 8)
        | (s2.to(tl.uint32) << 16)
        | (s3.to(tl.uint32) << 24)
    )
    scale_word = (
        tl.where(n0, 0, s0).to(tl.uint32)
        | (tl.where(n1, 0, s1).to(tl.uint32) << 8)
        | (tl.where(n2, 0, s2).to(tl.uint32) << 16)
        | (tl.where(n3, 0, s3).to(tl.uint32) << 24)
    )
    nan_word = (
        n0.to(tl.uint32)
        | (n1.to(tl.uint32) << 1)
        | (n2.to(tl.uint32) << 2)
        | (n3.to(tl.uint32) << 3)
    )
    exponent = ((raw_scales[:, None] >> (group[None, :] * 8)) & 255).to(
        tl.int32
    )
    nan = ((nan_word[:, None] >> group[None, :]) & 1) != 0
    packed = (
        _code(r0, exponent, nan)
        | (_code(r1, exponent, nan) << 8)
        | (_code(r2, exponent, nan) << 16)
        | (_code(r3, exponent, nan) << 24)
    )
    tl.store(qp + row[:, None] * 32 + lane[None, :], packed)
    tl.store(sp + row, scale_word)


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
def _quant_bf16(
    X,
    Q,
    S,
    G: tl.constexpr,
    R: tl.constexpr,
    START: tl.constexpr,
    MASK: tl.constexpr,
):
    col = tl.arange(0, 32)
    xp = X.to(tl.pointer_type(tl.int16))
    row = START + tl.program_id(0) * R + tl.arange(0, R)
    if MASK:
        raw = (
            tl.load(
                xp + row[:, None] * 32 + col[None, :], row[:, None] < G, 0
            ).to(tl.int32)
            & 65535
        )
    else:
        raw = (
            tl.load(xp + row[:, None] * 32 + col[None, :]).to(tl.int32) & 65535
        )
    mag = raw & 32767
    amax = tl.max(mag, 1)
    exponent = tl.maximum(
        (amax >> 7) - 8 + ((amax & 127) > 96).to(tl.int32), 19
    )
    exponent = tl.where(amax == 32640, 254, exponent)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    x = (raw << 16).to(tl.float32, bitcast=True)
    q = _encode_non_nan(x / scale[:, None])
    group_nan = amax > 32640
    nan_code = tl.where(raw > 65408, 255, 127)
    q = tl.where(group_nan[:, None], nan_code, q).to(tl.uint8)
    if MASK:
        tl.store(Q + row[:, None] * 32 + col[None, :], q, row[:, None] < G)
        tl.store(
            S + row, tl.where(group_nan, 0, exponent).to(tl.uint8), row < G
        )
    else:
        tl.store(Q + row[:, None] * 32 + col[None, :], q)
        tl.store(S + row, tl.where(group_nan, 0, exponent).to(tl.uint8))


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
        if x.dtype == torch.bfloat16:
            rows = 16 if groups <= 8192 else 64
            quad_rows = (
                64
                if groups > 8192 and groups % 256 == 0
                else (
                    16
                    if groups >= 256 and groups % 64 == 0
                    else (4 if groups >= 16 else 1)
                )
            )
            aligned = (
                source.data_ptr() % 8 == 0
                and q.data_ptr() % 4 == 0
                and scale.data_ptr() % 4 == 0
            )
            if aligned and groups % (quad_rows * 4) == 0:
                _quant_quad[(groups // (quad_rows * 4),)](
                    source,
                    q,
                    scale,
                    quad_rows,
                    core_num=16 if quad_rows == 1 else 64,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                full = groups // rows
                if full:
                    _quant_bf16[(full,)](
                        source,
                        q,
                        scale,
                        groups,
                        rows,
                        0,
                        False,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                if full * rows < groups:
                    _quant_bf16[(1,)](
                        source,
                        q,
                        scale,
                        groups,
                        rows,
                        full * rows,
                        True,
                        num_warps=4,
                        enable_fp_fusion=False,
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
