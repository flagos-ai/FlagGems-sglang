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
    col = tl.arange(0, 32)
    xp = X.to(tl.pointer_type(tl.int16))
    row = tl.program_id(0) * R + tl.arange(0, R)
    raw = (
        tl.load(xp + row[:, None] * 32 + col[None, :], row[:, None] < G, 0).to(
            tl.int32
        )
        & 65535
    )
    mag = raw & 32767
    amax = tl.max(mag, 1)
    exponent = tl.maximum(
        (amax >> 7) - 8 + ((amax & 127) > 96).to(tl.int32), 19
    )
    exponent = tl.where(amax == 32640, 254, exponent)
    inverse = tl.where(exponent >= 254, 4194304, (254 - exponent) << 23).to(
        tl.float32, bitcast=True
    )
    x = (raw << 16).to(tl.float32, bitcast=True)
    q = _encode_non_nan(x * inverse[:, None])
    group_nan = amax > 32640
    nan_code = tl.where(raw > 65408, 255, 127)
    q = tl.where(group_nan[:, None], nan_code, q).to(tl.uint8)
    tl.store(Q + row[:, None] * 32 + col[None, :], q, row[:, None] < G)
    tl.store(S + row, tl.where(group_nan, 0, exponent).to(tl.uint8), row < G)


def to_mxfp8(x):
    source = x.contiguous()
    if x.shape[-1] % 32:
        raise ValueError("last dimension must be divisible by 32")
    q = torch.empty_like(source, dtype=torch.uint8)
    scale = torch.empty(
        x.shape[:-1] + (x.shape[-1] // 32,), dtype=torch.uint8, device=x.device
    )
    groups = x.numel() // 32
    if groups:
        if x.dtype == torch.bfloat16:
            rows = 16 if groups <= 8192 else 64
            _quant_bf16[((groups + rows - 1) // rows,)](
                source,
                q,
                scale,
                groups,
                rows,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _quant[(min((groups + 3) // 4, 4096),)](
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
