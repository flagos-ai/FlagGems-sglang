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
def _half_tiles(
    KV,
    W,
    E,
    C,
    P,
    KO,
    VO,
    T: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    SKT: tl.constexpr,
    SKL: tl.constexpr,
    SW: tl.constexpr,
    SC: tl.constexpr,
    SP: tl.constexpr,
    BH: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NP: tl.constexpr,
):
    h = tl.arange(0, BH)
    p = tl.arange(0, BP)
    tail = tl.arange(0, BT)
    d = tl.arange(0, BD)
    half: tl.constexpr = RD // 2
    mp = (h[:, None] < H) & (p[None, :] < half)
    mt = (h[:, None] < H) & (tail[None, :] < D - RD)
    md = (h[:, None] < H) & (d[None, :] < D)
    for task in tl.range(tl.program_id(0), T * L, NP, num_stages=2):
        layer = task // T
        token = task % T
        base = token * SKT + layer * SKL + h[:, None] * D
        k1 = tl.load(KV + base + p[None, :], mp, 0).to(tl.float32)
        k2 = tl.load(KV + base + half + p[None, :], mp, 0).to(tl.float32)
        squares = tl.sum(k1 * k1, 1) + tl.sum(k2 * k2, 1)
        if D > RD:
            kt = tl.load(KV + base + RD + tail[None, :], mt, 0).to(tl.float32)
            squares += tl.sum(kt * kt, 1)
        epsilon = tl.load(E + layer).to(tl.float32)
        inv = tl.rsqrt(squares / D + epsilon)
        out = (layer * T + token) * H * D + h[:, None] * D
        if RD > 0:
            w1 = tl.load(W + layer * SW + p, p < half, 0).to(tl.float32)
            w2 = tl.load(W + layer * SW + half + p, p < half, 0).to(tl.float32)
            pos = tl.load(P + token * SP)
            cos = tl.load(C + pos * SC + p, p < half, 0).to(tl.float32)
            sin = tl.load(C + pos * SC + half + p, p < half, 0).to(tl.float32)
            wc1 = w1 * cos
            ws1 = w1 * sin
            wc2 = w2 * cos
            ws2 = w2 * sin
            tl.store(
                KO + out + p[None, :],
                (k1 * wc1[None, :] - k2 * ws2[None, :]) * inv[:, None],
                mp,
            )
            tl.store(
                KO + out + half + p[None, :],
                (k2 * wc2[None, :] + k1 * ws1[None, :]) * inv[:, None],
                mp,
            )
        if D > RD:
            wt = tl.load(W + layer * SW + RD + tail, tail < D - RD, 0).to(
                tl.float32
            )
            tl.store(
                KO + out + RD + tail[None, :],
                kt * inv[:, None] * wt[None, :],
                mt,
            )
        v = tl.load(KV + base + H * D + d[None, :], md, 0)
        tl.store(VO + out + d[None, :], v, md)


def fused_norm_rope_stacked(
    kv,
    k_norm_weight,
    eps,
    cos_sin_cache,
    positions,
    num_kv_heads,
    head_dim,
    rotary_dim,
):
    kv = kv.contiguous()
    t, layers, _ = kv.shape
    h, d, rd = (num_kv_heads, head_dim, rotary_dim)
    k = torch.empty((layers, t, h, d), dtype=kv.dtype, device=kv.device)
    v = torch.empty_like(k)
    if t == 0 or layers == 0 or h == 0:
        return (k, v)
    weight = k_norm_weight.contiguous().reshape(layers, d)
    epsilon = eps.contiguous().reshape(layers)
    cache = cos_sin_cache.contiguous()
    position = positions.contiguous()
    if position.dtype == torch.int64:
        position = position.view(torch.int32)
        stride = 2
    else:
        stride = position.stride(0)
    programs = min(t * layers, 256)
    _half_tiles.run(
        kv,
        weight,
        epsilon,
        cache,
        position,
        k,
        v,
        t,
        layers,
        h,
        d,
        rd,
        kv.stride(0),
        kv.stride(1),
        weight.stride(0),
        cache.stride(0),
        stride,
        triton.next_power_of_2(h),
        triton.next_power_of_2(max(rd // 2, 1)),
        triton.next_power_of_2(max(d - rd, 1)),
        triton.next_power_of_2(d),
        programs,
        grid=(programs,),
        warmup=False,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=True,
    )
    return (k, v)


__all__ = ["fused_norm_rope_stacked"]
