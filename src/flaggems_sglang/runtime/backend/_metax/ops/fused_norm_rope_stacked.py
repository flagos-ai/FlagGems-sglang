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
def _token_rows(
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
    BR: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NP: tl.constexpr,
):
    offset = tl.arange(0, BR)
    p = tl.arange(0, BP)
    tail = tl.arange(0, BT)
    d = tl.arange(0, BD)
    half: tl.constexpr = RD // 2
    groups: tl.constexpr = (L * H + BR - 1) // BR
    for work in tl.range(tl.program_id(0), T * groups, NP, num_stages=1):
        token = work // groups
        row = work % groups * BR + offset
        layer = row // H
        head = row % H
        valid = row < L * H
        mp = valid[:, None] & (p[None, :] < half)
        mt = valid[:, None] & (tail[None, :] < D - RD)
        md = valid[:, None] & (d[None, :] < D)
        base = token * SKT + layer[:, None] * SKL + head[:, None] * D
        k1 = tl.load(KV + base + p[None, :], mp, 0).to(tl.float32)
        k2 = tl.load(KV + base + half + p[None, :], mp, 0).to(tl.float32)
        squares = tl.sum(k1 * k1, 1) + tl.sum(k2 * k2, 1)
        if D > RD:
            kt = tl.load(KV + base + RD + tail[None, :], mt, 0).to(tl.float32)
            squares += tl.sum(kt * kt, 1)
        epsilon = tl.load(E + layer, valid, 1).to(tl.float32)
        inv = tl.rsqrt(squares / D + epsilon)
        out = (layer[:, None] * T + token) * H * D + head[:, None] * D
        if RD > 0:
            w1 = tl.load(W + layer[:, None] * SW + p[None, :], mp, 0).to(
                tl.float32
            )
            w2 = tl.load(
                W + layer[:, None] * SW + half + p[None, :], mp, 0
            ).to(tl.float32)
            n1 = k1 * inv[:, None] * w1
            n2 = k2 * inv[:, None] * w2
            pos = tl.load(P + token * SP)
            cos = tl.load(C + pos * SC + p, p < half, 0).to(tl.float32)
            sin = tl.load(C + pos * SC + half + p, p < half, 0).to(tl.float32)
            tl.store(
                KO + out + p[None, :],
                n1 * cos[None, :] - n2 * sin[None, :],
                mp,
            )
            tl.store(
                KO + out + half + p[None, :],
                n2 * cos[None, :] + n1 * sin[None, :],
                mp,
            )
        if D > RD:
            wt = tl.load(
                W + layer[:, None] * SW + RD + tail[None, :], mt, 0
            ).to(tl.float32)
            tl.store(KO + out + RD + tail[None, :], kt * inv[:, None] * wt, mt)
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
    rows = 32 if t * layers * h >= 4096 else 8
    tasks = t * triton.cdiv(layers * h, rows)
    programs = tasks
    _token_rows.run(
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
        rows,
        triton.next_power_of_2(max(rd // 2, 1)),
        triton.next_power_of_2(max(d - rd, 1)),
        triton.next_power_of_2(d),
        programs,
        grid=(programs,),
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (k, v)


__all__ = ["fused_norm_rope_stacked"]
