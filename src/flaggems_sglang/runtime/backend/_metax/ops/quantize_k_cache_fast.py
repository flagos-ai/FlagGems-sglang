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
import triton as tr
import triton.language as tl


@tr.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 2147483647
    normal = mag + 67633151 + (mag >> 20 & 1) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True)
    code = tl.where(mag < 1015021568, subnormal, tl.minimum(normal, 1150))
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _quant(x, s, factor, inv, RECIP: tl.constexpr):
    if not RECIP:
        return _encode(x / s)
    else:
        numerator = x * factor
        q = numerator * inv
        residual = tl.fma(-q, s * factor, numerator)
        corrected = tl.fma(residual, inv, q)
        return _encode(tl.where(s < float("inf"), corrected, q))


@tr.jit
def _pack(
    X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr, RECIP: tl.constexpr
):
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    P: tl.constexpr = 1
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    XS: tl.constexpr = M[4]
    XC: tl.constexpr = M[5]
    RS: tl.constexpr = M[6]
    RC: tl.constexpr = M[7]
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G // P)
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    mask = (
        (groups[:, None] < N * NG)
        & (c[None, :] < G // P)
        & (tl.arange(0, BG)[:, None] < BR * NG)
    )
    x = tl.load(
        X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
        mask,
        0,
    ).to(tl.float32)
    mag = x.to(tl.int32, bitcast=True) & 2147483647
    scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    inv = 1.0 / (scale * factor)
    tl.store(
        output_ptr + rows[:, None] * WIDTH + tiles[:, None] * G + c[None, :],
        _quant(x, scale[:, None], factor[:, None], inv[:, None], RECIP),
        mask,
    )
    tl.store(
        (output_ptr + D).to(tl.pointer_type(tl.float32))
        + rows * (WIDTH // 4)
        + tiles,
        scale,
        (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
    )
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    rc = tl.arange(0, tr.next_power_of_2(DR))
    rmask = (rr[:, None] < N) & (rc[None, :] < DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.int16)) + rr[:, None] * RS + rc[None, :] * RC,
        rmask,
        0,
    )
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.int16))
        + rr[:, None] * (WIDTH // 2)
        + rc[None, :],
        rope,
        rmask,
    )


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    dr = k_rope.shape[1]
    width = d + d // group_size * 4 + dr * 2
    output = k_nope.new_empty((n, width), dtype=torch.float8_e4m3fn)
    if n:
        br = 1 if n <= 16 else 4
        meta = (n, d, dr, group_size, *k_nope.stride(), *k_rope.stride())
        _pack.run(
            k_nope,
            k_rope,
            output,
            meta,
            br,
            n >= 4096,
            grid=(tr.cdiv(n, br),),
            warmup=False,
            enable_fp_fusion=False,
            num_warps=2 if n >= 4096 else 4,
        )
    return output


__all__ = ["quantize_k_cache_fast"]
