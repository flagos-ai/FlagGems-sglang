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
def direct__token_pair(
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
    SC: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    NP: tl.constexpr,
):
    groups: tl.constexpr = triton.cdiv(T, 4)
    ti = tl.arange(0, 4)
    head = tl.arange(0, BH)
    d = tl.arange(0, BD)
    pair = tl.arange(0, BP)
    half: tl.constexpr = RD // 2
    for work in tl.range(
        tl.program_id(0), L * groups, NP, num_stages=1, loop_unroll_factor=4
    ):
        layer = work % L
        token = work // L * 4 + ti
        base = (token[:, None, None] * L + layer) * (2 * H * D) + head[
            None, :, None
        ] * D
        valid = (token[:, None, None] < T) & (head[None, :, None] < H)
        k = tl.load(
            KV + base + d[None, None, :], valid & (d[None, None, :] < D), 0
        ).to(tl.float32)
        epsilon = tl.load(E + layer).to(tl.float32)
        inv = tl.rsqrt(tl.sum(k * k, 2) / D + epsilon)
        out = (
            (layer * T + token[:, None, None]) * H + head[None, :, None]
        ) * D
        if RD > 0:
            pos = tl.load(P + token, token < T, 0).to(tl.int32)
            cos = tl.load(
                C + pos[:, None] * SC + pair[None, :],
                (token[:, None] < T) & (pair[None, :] < half),
                0,
            ).to(tl.float32)
            sin = tl.load(
                C + pos[:, None] * SC + half + pair[None, :],
                (token[:, None] < T) & (pair[None, :] < half),
                0,
            ).to(tl.float32)
            mask = valid & (pair[None, None, :] < half)
            k1 = tl.load(KV + base + pair[None, None, :], mask, 0).to(
                tl.float32
            )
            k2 = tl.load(KV + base + half + pair[None, None, :], mask, 0).to(
                tl.float32
            )
            w1 = tl.load(W + layer * D + pair, pair < half, 0).to(tl.float32)
            w2 = tl.load(W + layer * D + half + pair, pair < half, 0).to(
                tl.float32
            )
            n1 = k1 * inv[:, :, None] * w1[None, None, :]
            n2 = k2 * inv[:, :, None] * w2[None, None, :]
            r1 = n1 * cos[:, None, :] - n2 * sin[:, None, :]
            r2 = n2 * cos[:, None, :] + n1 * sin[:, None, :]
            tl.store(KO + out + pair[None, None, :], r1, mask)
            tl.store(KO + out + half + pair[None, None, :], r2, mask)
        if RD < D:
            w = tl.load(W + layer * D + d, d < D, 0).to(tl.float32)
            normal = k * inv[:, :, None] * w[None, None, :]
            tl.store(
                KO + out + d[None, None, :],
                normal,
                valid & (d[None, None, :] >= RD) & (d[None, None, :] < D),
            )
        v = tl.load(
            KV + base + H * D + d[None, None, :],
            valid & (d[None, None, :] < D),
            0,
        )
        tl.store(
            VO + out + d[None, None, :], v, valid & (d[None, None, :] < D)
        )


def direct_fused_norm_rope_stacked(
    kv,
    k_norm_weight,
    eps,
    cos_sin_cache,
    positions,
    num_kv_heads,
    head_dim,
    rotary_dim,
):
    t, layers, _ = kv.shape
    h, d, rd = (num_kv_heads, head_dim, rotary_dim)
    k = torch.empty((layers, t, h, d), dtype=kv.dtype, device=kv.device)
    v = torch.empty_like(k)
    if t == 0 or layers == 0 or h == 0:
        return (k, v)
    if not kv.is_contiguous():
        kv = kv.contiguous()
    w = k_norm_weight
    if w.ndim != 2:
        w = w.reshape(layers, d)
    if not w.is_contiguous():
        w = w.contiguous()
    e = eps
    if e.ndim != 1:
        e = e.reshape(layers)
    if not e.is_contiguous():
        e = e.contiguous()
    c = cos_sin_cache
    if c.stride(1) != 1:
        c = c.contiguous()
    pos = positions if positions.is_contiguous() else positions.contiguous()
    half = rd // 2
    programs = min(layers * triton.cdiv(t, 4), 40)
    direct__token_pair.run(
        kv,
        w,
        e,
        c,
        pos,
        k,
        v,
        t,
        layers,
        h,
        d,
        rd,
        c.stride(0),
        triton.next_power_of_2(h),
        triton.next_power_of_2(d),
        triton.next_power_of_2(max(half, 1)),
        programs,
        grid=(programs,),
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (k, v)


@triton.jit
def cached__select_cache(
    C,
    P,
    O,
    T: tl.constexpr,
    RD: tl.constexpr,
    SC: tl.constexpr,
    BC: tl.constexpr,
    NP: tl.constexpr,
):
    row = tl.arange(0, 64)
    d = tl.arange(0, BC)
    for block in tl.range(
        tl.program_id(0), triton.cdiv(T, 64), NP, num_stages=1
    ):
        token = block * 64 + row
        pos = tl.load(P + token, token < T, 0).to(tl.int32)
        mask = (token[:, None] < T) & (d[None, :] < RD)
        value = tl.load(C + pos[:, None] * SC + d[None, :], mask, 0).to(
            tl.float32
        )
        tl.store(O + token[:, None] * RD + d[None, :], value, mask)


@triton.jit
def cached__token_pair(
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
    SC: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    NP: tl.constexpr,
    CACHED: tl.constexpr,
):
    groups: tl.constexpr = triton.cdiv(T, 4)
    ti = tl.arange(0, 4)
    head = tl.arange(0, BH)
    d = tl.arange(0, BD)
    pair = tl.arange(0, BP)
    half: tl.constexpr = RD // 2
    for work in tl.range(
        tl.program_id(0), L * groups, NP, num_stages=1, loop_unroll_factor=4
    ):
        layer = work % L
        token = work // L * 4 + ti
        base = (token[:, None, None] * L + layer) * (2 * H * D) + head[
            None, :, None
        ] * D
        valid = (token[:, None, None] < T) & (head[None, :, None] < H)
        k = tl.load(
            KV + base + d[None, None, :], valid & (d[None, None, :] < D), 0
        ).to(tl.float32)
        epsilon = tl.load(E + layer).to(tl.float32)
        inv = tl.rsqrt(tl.sum(k * k, 2) / D + epsilon)
        out = (
            (layer * T + token[:, None, None]) * H + head[None, :, None]
        ) * D
        if RD > 0:
            if CACHED:
                pos = token
            else:
                pos = tl.load(P + token, token < T, 0).to(tl.int32)
            cos = tl.load(
                C + pos[:, None] * SC + pair[None, :],
                (token[:, None] < T) & (pair[None, :] < half),
                0,
            ).to(tl.float32)
            sin = tl.load(
                C + pos[:, None] * SC + half + pair[None, :],
                (token[:, None] < T) & (pair[None, :] < half),
                0,
            ).to(tl.float32)
            mask = valid & (pair[None, None, :] < half)
            k1 = tl.load(KV + base + pair[None, None, :], mask, 0).to(
                tl.float32
            )
            k2 = tl.load(KV + base + half + pair[None, None, :], mask, 0).to(
                tl.float32
            )
            w1 = tl.load(W + layer * D + pair, pair < half, 0).to(tl.float32)
            w2 = tl.load(W + layer * D + half + pair, pair < half, 0).to(
                tl.float32
            )
            n1 = k1 * inv[:, :, None] * w1[None, None, :]
            n2 = k2 * inv[:, :, None] * w2[None, None, :]
            r1 = n1 * cos[:, None, :] - n2 * sin[:, None, :]
            r2 = n2 * cos[:, None, :] + n1 * sin[:, None, :]
            tl.store(KO + out + pair[None, None, :], r1, mask)
            tl.store(KO + out + half + pair[None, None, :], r2, mask)
        if RD < D:
            w = tl.load(W + layer * D + d, d < D, 0).to(tl.float32)
            normal = k * inv[:, :, None] * w[None, None, :]
            tl.store(
                KO + out + d[None, None, :],
                normal,
                valid & (d[None, None, :] >= RD) & (d[None, None, :] < D),
            )
        v = tl.load(
            KV + base + H * D + d[None, None, :],
            valid & (d[None, None, :] < D),
            0,
        )
        tl.store(
            VO + out + d[None, None, :], v, valid & (d[None, None, :] < D)
        )


def cached_fused_norm_rope_stacked(
    kv,
    k_norm_weight,
    eps,
    cos_sin_cache,
    positions,
    num_kv_heads,
    head_dim,
    rotary_dim,
):
    t, layers, _ = kv.shape
    h, d, rd = (num_kv_heads, head_dim, rotary_dim)
    k = torch.empty((layers, t, h, d), dtype=kv.dtype, device=kv.device)
    v = torch.empty_like(k)
    if t == 0 or layers == 0 or h == 0:
        return (k, v)
    if not kv.is_contiguous():
        kv = kv.contiguous()
    w = k_norm_weight
    if w.ndim != 2:
        w = w.reshape(layers, d)
    if not w.is_contiguous():
        w = w.contiguous()
    e = eps
    if e.ndim != 1:
        e = e.reshape(layers)
    if not e.is_contiguous():
        e = e.contiguous()
    c = cos_sin_cache
    if c.stride(1) != 1:
        c = c.contiguous()
    pos = positions if positions.is_contiguous() else positions.contiguous()
    half = rd // 2
    programs = min(layers * triton.cdiv(t, 4), 40)
    cached = t * layers >= 4096 and rd > 0
    if cached:
        selected = torch.empty((t, rd), dtype=torch.float32, device=kv.device)
        cache_programs = min(triton.cdiv(t, 64), 40)
        cached__select_cache[cache_programs,](
            c,
            pos,
            selected,
            t,
            rd,
            c.stride(0),
            triton.next_power_of_2(rd),
            cache_programs,
            num_warps=4,
            num_stages=1,
        )
        c = selected
    cached__token_pair.run(
        kv,
        w,
        e,
        c,
        pos,
        k,
        v,
        t,
        layers,
        h,
        d,
        rd,
        c.stride(0),
        triton.next_power_of_2(h),
        triton.next_power_of_2(d),
        triton.next_power_of_2(max(half, 1)),
        programs,
        cached,
        grid=(programs,),
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (k, v)


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
    if kv.shape[0] * kv.shape[1] >= 32768 and rotary_dim > 0:
        return cached_fused_norm_rope_stacked(
            kv,
            k_norm_weight,
            eps,
            cos_sin_cache,
            positions,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
    return direct_fused_norm_rope_stacked(
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )


__all__ = ["fused_norm_rope_stacked"]
