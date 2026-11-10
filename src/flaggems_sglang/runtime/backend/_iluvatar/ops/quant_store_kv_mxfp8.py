# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused K/V block quantization and paged scatter, preserving untouched slots."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _quant(x):
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-30)
    bits = amax.to(tl.int32, bitcast=True)
    exponent = (
        ((bits >> 23) & 255) - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    )
    exponent = tl.minimum(tl.maximum(exponent, 0), 254)
    inverse = tl.where(exponent == 254, 0x00400000, (254 - exponent) << 23).to(
        tl.float32, bitcast=True
    )
    v = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0)
    ax = tl.abs(v)
    b = ax.to(tl.int32, bitcast=True)
    normal = ((b + 0x7FFFF + ((b >> 20) & 1)) >> 20) - 960
    sub = (ax + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(ax < 0.015625, sub, normal)
    byte = code | ((v.to(tl.int32, bitcast=True) >> 24) & 128)
    return tl.where(byte >= 128, byte - 256, byte), exponent


@tr.jit
def _store(
    K,
    V,
    L,
    KC,
    VC,
    SK,
    SV,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    LS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    KCS: tl.constexpr,
    VCS: tl.constexpr,
    SKS: tl.constexpr,
    SVS: tl.constexpr,
    R: tl.constexpr,
    GX: tl.constexpr,
    LARGE: tl.constexpr,
):
    SF: tl.constexpr = D // 32
    GROUPS: tl.constexpr = T * H * SF
    kv = tl.program_id(1) % 2
    tile = tl.program_id(0) + (tl.program_id(1) // 2) * GX
    group = tile * R + tl.arange(0, R)
    if LARGE:
        group = group.to(tl.int64)
    token = group // (H * SF)
    head = (group // SF) % H
    block = group % SF
    col = block[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = group[:, None] < GROUPS
    loc = tl.load(L + token * LS, group < GROUPS, 0)
    if LARGE:
        loc = loc.to(tl.int64)
    else:
        loc = loc.to(tl.int32)
    X = tl.where(kv == 0, K, V)
    C = tl.where(kv == 0, KC, VC)
    S = tl.where(kv == 0, SK, SV)
    xo = (
        token[:, None] * tl.where(kv == 0, KS[0], VS[0])
        + head[:, None] * tl.where(kv == 0, KS[1], VS[1])
        + col * tl.where(kv == 0, KS[2], VS[2])
    )
    x = tl.load(X + xo, mask, 0).to(tl.float32)
    q, s = _quant(x)
    co = (
        loc[:, None] * tl.where(kv == 0, KCS[0], VCS[0])
        + head[:, None] * tl.where(kv == 0, KCS[1], VCS[1])
        + col * tl.where(kv == 0, KCS[2], VCS[2])
    )
    tl.store(C + co, q, mask)
    off = loc % PAGE
    si = (
        (loc // PAGE) * tl.where(kv == 0, SKS[0], SVS[0])
        + head * tl.where(kv == 0, SKS[1], SVS[1])
        + (off % 32) * tl.where(kv == 0, SKS[2], SVS[2])
        + (off // 32) * tl.where(kv == 0, SKS[3], SVS[3])
        + block * tl.where(kv == 0, SKS[4], SVS[4])
    )
    tl.store(S + si, s, group < GROUPS)


def quant_store_kv_mxfp8(k, v, loc, k_cache, v_cache, sfk, sfv, page_size):
    t, h, d = k.shape
    groups = t * h * (d // 32)
    if groups:
        r = 32
        tiles = tr.cdiv(groups, r)
        p = min(tiles, 65535)
        gy = tr.cdiv(tiles, p)
        large = (
            max(
                t * h * d,
                *(
                    sum((n - 1) * s for n, s in zip(x.shape, x.stride()))
                    for x in (k, v, k_cache, v_cache, sfk, sfv)
                ),
            )
            >= 2147483648
        )
        _store[(p, gy * 2)](
            k,
            v,
            loc,
            k_cache.view(torch.int8),
            v_cache.view(torch.int8),
            sfk,
            sfv,
            t,
            h,
            d,
            page_size,
            loc.stride(0),
            tuple(k.stride()),
            tuple(v.stride()),
            tuple(k_cache.stride()),
            tuple(v_cache.stride()),
            tuple(sfk.stride()),
            tuple(sfv.stride()),
            r,
            p,
            large,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return k_cache, v_cache, sfk, sfv


solution = quant_store_kv_mxfp8

__all__ = ["quant_store_kv_mxfp8"]
