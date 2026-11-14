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
def _position_cache(
    C,
    P,
    O,
    T: tl.constexpr,
    RD: tl.constexpr,
    SC: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
    NP: tl.constexpr,
):
    d = tl.arange(0, BC)
    blocks: tl.constexpr = (T + BP - 1) // BP
    for blk in tl.range(tl.program_id(0), blocks, NP, num_stages=1):
        t = blk * BP + tl.arange(0, BP)
        pos = tl.load(P + t, t < T, 0)
        value = tl.load(
            C + pos[:, None] * SC + d[None, :],
            (t[:, None] < T) & (d[None, :] < RD),
            0,
        ).to(tl.float32)
        tl.store(
            O + t[:, None] * RD + d[None, :],
            value.to(O.dtype.element_ty),
            (t[:, None] < T) & (d[None, :] < RD),
        )


@triton.jit
def _cached_norm(
    KV,
    W,
    E,
    C,
    KO,
    VO,
    T: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    ST: tl.constexpr,
    SL: tl.constexpr,
    SW: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
    BX: tl.constexpr,
    NP: tl.constexpr,
):
    ts = tl.arange(0, BT)
    hs = tl.arange(0, BH)
    p = tl.arange(0, BP)
    d = tl.arange(0, BD)
    tail = tl.arange(0, BX)
    half: tl.constexpr = RD // 2
    groups: tl.constexpr = (T + BT - 1) // BT
    for task in tl.range(tl.program_id(0), L * groups, NP, num_stages=1):
        layer = task // groups
        token = task % groups * BT + ts
        valid = (token[:, None, None] < T) & (hs[None, :, None] < H)
        mp = valid & (p[None, None, :] < half)
        mt = valid & (tail[None, None, :] < D - RD)
        md = valid & (d[None, None, :] < D)
        base = token[:, None, None] * ST + layer * SL + hs[None, :, None] * D
        # Independent small loads issue first so their latencies overlap
        # with the K loads and the reduction tree.
        epsilon = tl.load(E + layer).to(tl.float32)
        if RD > 0:
            w1 = tl.load(W + layer * SW + p, p < half, 0).to(tl.float32)
            w2 = tl.load(W + layer * SW + half + p, p < half, 0).to(tl.float32)
            mc = (token[:, None] < T) & (p[None, :] < half)
            cos = tl.load(C + token[:, None] * RD + p[None, :], mc, 0).to(
                tl.float32
            )
            sin = tl.load(
                C + token[:, None] * RD + half + p[None, :], mc, 0
            ).to(tl.float32)
        k1 = tl.load(KV + base + p[None, None, :], mp, 0).to(tl.float32)
        k2 = tl.load(KV + base + half + p[None, None, :], mp, 0).to(tl.float32)
        squares = tl.sum(k1 * k1, 2) + tl.sum(k2 * k2, 2)
        if D > RD:
            kt = tl.load(
                KV + base + RD + tail[None, None, :], mt, 0
            ).to(tl.float32)
            squares += tl.sum(kt * kt, 2)
        inv = tl.rsqrt(squares / D + epsilon)
        out = (layer * T + token[:, None, None]) * H * D + hs[
            None, :, None
        ] * D
        if RD > 0:
            n1 = k1 * inv[:, :, None] * w1[None, None, :]
            n2 = k2 * inv[:, :, None] * w2[None, None, :]
            tl.store(
                KO + out + p[None, None, :],
                n1 * cos[:, None, :] - n2 * sin[:, None, :],
                mp,
            )
            tl.store(
                KO + out + half + p[None, None, :],
                n2 * cos[:, None, :] + n1 * sin[:, None, :],
                mp,
            )
        if D > RD:
            wt = tl.load(W + layer * SW + RD + tail, tail < D - RD, 0).to(
                tl.float32
            )
            tl.store(
                KO + out + RD + tail[None, None, :],
                kt * inv[:, :, None] * wt[None, None, :],
                mt,
            )
        v = tl.load(KV + base + H * D + d[None, None, :], md, 0)
        tl.store(VO + out + d[None, None, :], v, md)


@triton.jit
def _cached_norm_direct(
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
    ST: tl.constexpr,
    SL: tl.constexpr,
    SW: tl.constexpr,
    SC: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
    BX: tl.constexpr,
    NP: tl.constexpr,
):
    # Small-batch arm: a single launch fuses the position gather and the
    # norm/rope/passthrough work, skipping the selected-table round trip.
    ts = tl.arange(0, BT)
    hs = tl.arange(0, BH)
    p = tl.arange(0, BP)
    d = tl.arange(0, BD)
    tail = tl.arange(0, BX)
    half: tl.constexpr = RD // 2
    groups: tl.constexpr = (T + BT - 1) // BT
    for task in tl.range(tl.program_id(0), L * groups, NP, num_stages=1):
        layer = task // groups
        token = task % groups * BT + ts
        valid = (token[:, None, None] < T) & (hs[None, :, None] < H)
        mp = valid & (p[None, None, :] < half)
        mt = valid & (tail[None, None, :] < D - RD)
        md = valid & (d[None, None, :] < D)
        base = token[:, None, None] * ST + layer * SL + hs[None, :, None] * D
        epsilon = tl.load(E + layer).to(tl.float32)
        if RD > 0:
            w1 = tl.load(W + layer * SW + p, p < half, 0).to(tl.float32)
            w2 = tl.load(W + layer * SW + half + p, p < half, 0).to(tl.float32)
            pos = tl.load(P + token, token < T, 0)
            mc = (token[:, None] < T) & (p[None, :] < half)
            cos = tl.load(C + pos[:, None] * SC + p[None, :], mc, 0).to(
                tl.float32
            )
            sin = tl.load(
                C + pos[:, None] * SC + half + p[None, :], mc, 0
            ).to(tl.float32)
        k1 = tl.load(KV + base + p[None, None, :], mp, 0).to(tl.float32)
        k2 = tl.load(KV + base + half + p[None, None, :], mp, 0).to(tl.float32)
        squares = tl.sum(k1 * k1, 2) + tl.sum(k2 * k2, 2)
        if D > RD:
            kt = tl.load(
                KV + base + RD + tail[None, None, :], mt, 0
            ).to(tl.float32)
            squares += tl.sum(kt * kt, 2)
        inv = tl.rsqrt(squares / D + epsilon)
        out = (layer * T + token[:, None, None]) * H * D + hs[
            None, :, None
        ] * D
        if RD > 0:
            n1 = k1 * inv[:, :, None] * w1[None, None, :]
            n2 = k2 * inv[:, :, None] * w2[None, None, :]
            tl.store(
                KO + out + p[None, None, :],
                n1 * cos[:, None, :] - n2 * sin[:, None, :],
                mp,
            )
            tl.store(
                KO + out + half + p[None, None, :],
                n2 * cos[:, None, :] + n1 * sin[:, None, :],
                mp,
            )
        if D > RD:
            wt = tl.load(W + layer * SW + RD + tail, tail < D - RD, 0).to(
                tl.float32
            )
            tl.store(
                KO + out + RD + tail[None, None, :],
                kt * inv[:, :, None] * wt[None, None, :],
                mt,
            )
        v = tl.load(KV + base + H * D + d[None, None, :], md, 0)
        tl.store(VO + out + d[None, None, :], v, md)


_DSM_BUDGET = 850000


def _norm_tile_dsm(bt, bh, bp, bd, bx):
    # Measured (BT=128): BH=4 -> 921600B, i.e. the pair path materializes
    # ~6 fp32 [BT,BH,BP] tiles (k1,k2,k1^2,k2^2,n1,n2) plus the bf16 v
    # tile; pad the tile count for compile-variance margin.
    pair = 8 * 4 * bt * bh * bp
    tail = 4 * 4 * bt * bh * bx
    passthrough = 2 * bt * bh * bd
    cos_sin = 8 * bt * bp
    return pair + tail + passthrough + cos_sin + 8192


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
    pos = positions.contiguous().to(torch.int32)
    bh = triton.next_power_of_2(h)
    bp = triton.next_power_of_2(max(rd // 2, 1))
    bd = triton.next_power_of_2(d)
    bx = triton.next_power_of_2(max(d - rd, 1))
    if t <= 32:
        # Small batches: a single fused launch replaces the two-kernel
        # pipeline; the per-token gather is negligible at this size
        # (t=1 end to end 48.8 -> 17us against the staged path).
        tasks = layers * triton.cdiv(t, 2)
        programs = min(tasks, 48)
        _cached_norm_direct[(programs,)](
            kv,
            weight,
            epsilon,
            cache,
            pos,
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
            2,
            bh,
            bp,
            bd,
            bx,
            programs,
            num_warps=1,
            num_stages=1,
        )
        return (k, v)
    # The rotation table mirrors the cache dtype: values round-trip
    # losslessly, and bf16 caches halve the per-layer re-read traffic.
    selected = torch.empty((t, rd), dtype=cache.dtype, device=kv.device)
    if rd > 0:
        blocks = triton.cdiv(t, 32)
        programs1 = min(blocks, 48)
        _position_cache[(programs1,)](
            cache,
            pos,
            selected,
            t,
            rd,
            cache.stride(0),
            32,
            triton.next_power_of_2(rd),
            programs1,
            num_warps=4,
            num_stages=1,
        )
    tokens = 128
    while tokens > 1 and _norm_tile_dsm(tokens, bh, bp, bd, bx) > _DSM_BUDGET:
        tokens //= 2
    tasks = layers * triton.cdiv(t, tokens)
    programs = min(tasks, 48)
    _cached_norm[(programs,)](
        kv,
        weight,
        epsilon,
        selected,
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
        tokens,
        bh,
        bp,
        bd,
        bx,
        programs,
        num_warps=1,
        num_stages=1,
    )
    return (k, v)


__all__ = ["fused_norm_rope_stacked"]
