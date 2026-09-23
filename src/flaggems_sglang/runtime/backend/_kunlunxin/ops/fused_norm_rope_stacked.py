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
def _row_block(
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
    SP: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    safe_row = tl.minimum(row, L * T * H - 1)
    head = safe_row % H
    token = safe_row // H % T
    layer = safe_row // (T * H)
    offs = tl.arange(0, BD)
    idx = tl.where(offs < D, offs, 0)
    base = token[:, None] * ST + layer[:, None] * SL + head[:, None] * D
    raw = tl.load(KV + base + idx[None, :]).to(tl.float32)
    k = tl.where(offs[None, :] < D, raw, 0.0)
    epsilon = tl.load(E + layer).to(tl.float32)
    inv = tl.rsqrt(tl.sum(k * k, axis=1) / D + epsilon)
    weight = tl.load(W + layer[:, None] * SW + idx[None, :]).to(tl.float32)
    normal = raw * inv[:, None] * weight
    if RD > 0:
        half: tl.constexpr = RD // 2
        lower = offs < half
        in_rot = offs < RD
        partner = tl.where(
            in_rot, tl.where(lower, idx + half, idx - half), idx
        )
        angle = tl.where(in_rot, tl.where(lower, idx, idx - half), 0)
        kp = tl.load(KV + base + partner[None, :]).to(tl.float32)
        wp = tl.load(W + layer[:, None] * SW + partner[None, :]).to(tl.float32)
        partner_normal = kp * inv[:, None] * wp
        pos = tl.load(P + token * SP)
        cache_base = pos[:, None] * SC
        cos = tl.load(C + cache_base + angle[None, :]).to(tl.float32)
        sin = tl.load(C + cache_base + half + angle[None, :]).to(tl.float32)
        sign = tl.where(lower, -1.0, 1.0)
        rotated = normal * cos + sign[None, :] * partner_normal * sin
        normal = tl.where(in_rot[None, :], rotated, normal)
    output = row[:, None] * D + offs[None, :]
    valid = (row[:, None] < L * T * H) & (offs[None, :] < D)
    v = tl.load(KV + base + H * D + idx[None, :])
    tl.store(KO + output, normal, valid)
    tl.store(VO + output, v, valid)


def _rows_call(
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
    weight = k_norm_weight.reshape(layers, d).contiguous()
    epsilon = eps.reshape(layers).to(torch.float32).contiguous()
    cache = cos_sin_cache.contiguous()
    pos = positions.reshape(-1)
    rows = 8
    _row_block[triton.cdiv(layers * t * h, rows),](
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
        pos.stride(0),
        rows,
        triton.next_power_of_2(d),
        num_warps=1,
        num_stages=1,
    )
    return (k, v)


@triton.jit
def _token_heads(
    KV,
    W,
    E,
    C,
    P,
    KO,
    VO,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    ST: tl.constexpr,
    SL: tl.constexpr,
    SW: tl.constexpr,
    SC: tl.constexpr,
    SP: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0)
    layer = tl.program_id(1)
    head = tl.arange(0, BH)
    safe_head = tl.minimum(head, H - 1)
    offs = tl.arange(0, BD)
    idx = tl.where(offs < D, offs, 0)
    base = token * ST + layer * SL + safe_head[:, None] * D
    raw = tl.load(KV + base + idx[None, :]).to(tl.float32)
    k = tl.where(offs[None, :] < D, raw, 0.0)
    epsilon = tl.load(E + layer).to(tl.float32)
    inv = tl.rsqrt(tl.sum(k * k, axis=1) / D + epsilon)
    weight = tl.load(W + layer * SW + idx).to(tl.float32)
    normal = raw * inv[:, None] * weight[None, :]
    if RD > 0:
        half: tl.constexpr = RD // 2
        lower = offs < half
        in_rot = offs < RD
        partner = tl.where(
            in_rot, tl.where(lower, idx + half, idx - half), idx
        )
        angle = tl.where(in_rot, tl.where(lower, idx, idx - half), 0)
        kp = tl.load(KV + base + partner[None, :]).to(tl.float32)
        wp = tl.load(W + layer * SW + partner).to(tl.float32)
        partner_normal = kp * inv[:, None] * wp[None, :]
        pos = tl.load(P + token * SP)
        cos = tl.load(C + pos * SC + angle).to(tl.float32)
        sin = tl.load(C + pos * SC + half + angle).to(tl.float32)
        sign = tl.where(lower, -1.0, 1.0)
        rotated = (
            normal * cos[None, :]
            + sign[None, :] * partner_normal * sin[None, :]
        )
        normal = tl.where(in_rot[None, :], rotated, normal)
    output = ((layer * T + token) * H + head[:, None]) * D + offs[None, :]
    valid = (head[:, None] < H) & (offs[None, :] < D)
    v = tl.load(KV + base + H * D + idx[None, :])
    tl.store(KO + output, normal, valid)
    tl.store(VO + output, v, valid)


def _heads_call(
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
    weight = k_norm_weight.reshape(layers, d).contiguous()
    epsilon = eps.reshape(layers).to(torch.float32).contiguous()
    cache = cos_sin_cache.contiguous()
    pos = positions.reshape(-1)
    _token_heads[t, layers](
        kv,
        weight,
        epsilon,
        cache,
        pos,
        k,
        v,
        t,
        h,
        d,
        rd,
        kv.stride(0),
        kv.stride(1),
        weight.stride(0),
        cache.stride(0),
        pos.stride(0),
        triton.next_power_of_2(h),
        triton.next_power_of_2(d),
        num_warps=1,
        num_stages=1,
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
    if num_kv_heads >= 8:
        return _heads_call(
            kv,
            k_norm_weight,
            eps,
            cos_sin_cache,
            positions,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
    return _rows_call(
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
