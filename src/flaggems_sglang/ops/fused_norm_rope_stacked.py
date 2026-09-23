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
def _pair_gather(
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
    BT: tl.constexpr,
    CACHE: tl.constexpr,
):
    layer = tl.program_id(1)
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.arange(0, BH)
    d = tl.arange(0, BD)
    half: tl.constexpr = RD // 2
    valid = (
        (token[:, None, None] < T)
        & (head[None, :, None] < H)
        & (d[None, None, :] < D)
    )
    base = (token[:, None, None] * L + layer) * (2 * H * D) + head[
        None, :, None
    ] * D
    k = tl.load(
        KV + base + d[None, None, :], valid, 0, cache_modifier=CACHE
    ).to(tl.float32)
    epsilon = tl.load(E + layer).to(tl.float32)
    inv = tl.rsqrt(tl.sum(k * k, 2) / D + epsilon)
    w = tl.load(W + layer * D + d, d < D, 0).to(tl.float32)
    normal = k * inv[:, :, None] * w[None, None, :]
    if RD > 0:
        partner = tl.where(d < RD, tl.where(d < half, d + half, d - half), d)
        paired = tl.gather(
            normal, tl.broadcast_to(partner[None, None, :], (BT, BH, BD)), 2
        )
        angle = tl.where(d < half, d, d - half)
        pos = tl.load(P + token, token < T, 0)
        cmask = (token[:, None] < T) & (d[None, :] < RD)
        cos = tl.load(C + pos[:, None] * SC + angle[None, :], cmask, 0).to(
            tl.float32
        )
        sin = tl.load(
            C + pos[:, None] * SC + half + angle[None, :], cmask, 0
        ).to(tl.float32)
        rotated = tl.where(
            d[None, None, :] < half,
            normal * cos[:, None, :] - paired * sin[:, None, :],
            normal * cos[:, None, :] + paired * sin[:, None, :],
        )
        normal = tl.where(d[None, None, :] < RD, rotated, normal)
    out = (
        (layer * T + token[:, None, None]) * H + head[None, :, None]
    ) * D + d[None, None, :]
    v = tl.load(
        KV + base + H * D + d[None, None, :], valid, 0, cache_modifier=CACHE
    )
    tl.store(KO + out, normal, valid)
    tl.store(VO + out, v, valid)


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
    t, layers, _ = kv.shape
    h, d, rd = (num_kv_heads, head_dim, rotary_dim)
    k = torch.empty((layers, t, h, d), dtype=kv.dtype, device=kv.device)
    v = torch.empty_like(k)
    if t == 0 or layers == 0 or h == 0:
        return (k, v)
    kv = kv.contiguous()
    w = k_norm_weight.reshape(layers, d).contiguous()
    e = eps.reshape(layers).contiguous()
    c = cos_sin_cache.contiguous()
    pos = positions.reshape(-1).contiguous()
    bt, warps, cache = (1, 8, ".cg")
    _pair_gather.run(
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
        bt,
        cache,
        grid=(triton.cdiv(t, bt), layers),
        warmup=False,
        num_warps=warps,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return (k, v)


__all__ = ["fused_norm_rope_stacked"]
