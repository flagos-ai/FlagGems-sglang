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
def small__fnrs_outmajor_kernel(
    kv_ptr,
    w_ptr,
    eps_ptr,
    cache_ptr,
    pos_ptr,
    k_out_ptr,
    v_out_ptr,
    T: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    CS_S: tl.constexpr,
    PS: tl.constexpr,
):
    BLOCK_H: tl.constexpr = triton.next_power_of_2(H)
    BLOCK_D: tl.constexpr = triton.next_power_of_2(D)
    BLOCK_HALF: tl.constexpr = triton.next_power_of_2(max(RD // 2, 1))
    NUM_PROGRAMS: tl.constexpr = 256
    n_tasks: tl.constexpr = T * L
    HALF: tl.constexpr = RD // 2
    KV_ST: tl.constexpr = L * H * D * 2
    KV_SL: tl.constexpr = H * D * 2
    W_SL: tl.constexpr = D
    KO_SL: tl.constexpr = T * H * D
    KO_ST: tl.constexpr = H * D
    VO_SL: tl.constexpr = T * H * D
    VO_ST: tl.constexpr = H * D
    pid = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    pair = tl.arange(0, BLOCK_HALF)
    hm = h < H
    dm = d < D
    pm = pair < HALF
    for task in tl.range(pid, n_tasks, NUM_PROGRAMS, num_stages=2):
        layer = task // T
        t = task % T
        kv_tl = t * KV_ST + layer * KV_SL
        k = tl.load(
            kv_ptr + kv_tl + h[:, None] * D + d[None, :],
            mask=hm[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + layer * W_SL + d, mask=dm, other=0.0).to(
            tl.float32
        )
        eps = tl.load(eps_ptr + layer).to(tl.float32)
        inv = tl.rsqrt(tl.sum(k * k, axis=1) / D + eps)
        kn = k * inv[:, None] * w[None, :]
        k1 = tl.load(
            kv_ptr + kv_tl + h[:, None] * D + pair[None, :],
            mask=hm[:, None] & pm[None, :],
            other=0.0,
        ).to(tl.float32)
        k2 = tl.load(
            kv_ptr + kv_tl + h[:, None] * D + HALF + pair[None, :],
            mask=hm[:, None] & pm[None, :],
            other=0.0,
        ).to(tl.float32)
        w1 = tl.load(w_ptr + layer * W_SL + pair, mask=pm, other=0.0).to(
            tl.float32
        )
        w2 = tl.load(
            w_ptr + layer * W_SL + HALF + pair, mask=pm, other=0.0
        ).to(tl.float32)
        pos = tl.load(pos_ptr + t * PS)
        cache_base = cache_ptr + pos * CS_S
        cos = tl.load(cache_base + pair, mask=pm, other=0.0).to(tl.float32)
        sin = tl.load(cache_base + HALF + pair, mask=pm, other=0.0).to(
            tl.float32
        )
        wc1 = w1 * cos
        ws1 = w1 * sin
        wc2 = w2 * cos
        ws2 = w2 * sin
        rot1 = (k1 * wc1[None, :] - k2 * ws2[None, :]) * inv[:, None]
        rot2 = (k2 * wc2[None, :] + k1 * ws1[None, :]) * inv[:, None]
        k_base = layer * KO_SL + t * KO_ST
        out_ty = k_out_ptr.dtype.element_ty
        tl.store(
            k_out_ptr + k_base + h[:, None] * D + pair[None, :],
            rot1.to(out_ty),
            mask=hm[:, None] & pm[None, :],
        )
        tl.store(
            k_out_ptr + k_base + h[:, None] * D + HALF + pair[None, :],
            rot2.to(out_ty),
            mask=hm[:, None] & pm[None, :],
        )
        if RD < D:
            tl.store(
                k_out_ptr + k_base + h[:, None] * D + d[None, :],
                kn.to(out_ty),
                mask=hm[:, None] & dm[None, :] & (d[None, :] >= RD),
            )
        v = tl.load(
            kv_ptr + kv_tl + H * D + h[:, None] * D + d[None, :],
            mask=hm[:, None] & dm[None, :],
        )
        v_base = layer * VO_SL + t * VO_ST
        tl.store(
            v_out_ptr + v_base + h[:, None] * D + d[None, :],
            v.to(v_out_ptr.dtype.element_ty),
            mask=hm[:, None] & dm[None, :],
        )


def small_fused_norm_rope_stacked(
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
    h = int(num_kv_heads)
    d = int(head_dim)
    rd = int(rotary_dim)
    k_out = torch.empty((layers, t, h, d), dtype=kv.dtype, device=kv.device)
    v_out = torch.empty_like(k_out)
    if t == 0 or layers == 0 or h == 0:
        return (k_out, v_out)
    w = k_norm_weight.contiguous().reshape(layers, d)
    eps_f = eps.contiguous().reshape(layers)
    cache = cos_sin_cache.contiguous()
    pos = positions.contiguous()
    if pos.dtype == torch.int64:
        pos = pos.view(torch.int32)
        pos_stride = 2
    else:
        pos_stride = int(pos.stride(0))
    n_tasks = t * layers
    num_programs = n_tasks if n_tasks < 256 else 256
    small__fnrs_outmajor_kernel.run(
        kv,
        w,
        eps_f,
        cache,
        pos,
        k_out,
        v_out,
        T=t,
        L=layers,
        H=h,
        D=d,
        RD=rd,
        CS_S=cache.stride(0),
        PS=pos_stride,
        grid=(num_programs,),
        warmup=False,
        num_warps=1,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return (k_out, v_out)


@triton.jit
def wide__pair_gather(
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
    BT: tl.constexpr,
):
    BH: tl.constexpr = triton.next_power_of_2(H)
    BD: tl.constexpr = triton.next_power_of_2(D)
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
    k = tl.load(KV + base + d[None, None, :], valid, 0).to(tl.float32)
    epsilon = tl.load(E + layer).to(tl.float32)
    inv = tl.rsqrt(tl.sum(k * k, 2) / D + epsilon)
    w = tl.load(W + layer * D + d, d < D, 0).to(tl.float32)
    normal = k * inv[:, :, None] * w[None, None, :]
    if RD > 0:
        partner = tl.where(d < RD, tl.where(d < half, d + half, d - half), d)
        paired = tl.gather(
            k, tl.broadcast_to(partner[None, None, :], (BT, BH, BD)), 2
        )
        wp = tl.load(W + layer * D + partner, d < D, 0).to(tl.float32)
        angle = tl.where(d < half, d, d - half)
        pos = tl.load(P + token, token < T, 0)
        cmask = (token[:, None] < T) & (d[None, :] < RD)
        cos = tl.load(C + pos[:, None] * SC + angle[None, :], cmask, 0).to(
            tl.float32
        )
        sin = tl.load(
            C + pos[:, None] * SC + half + angle[None, :], cmask, 0
        ).to(tl.float32)
        wc = w[None, :] * cos
        ws = wp[None, :] * sin
        rotated = (
            tl.where(
                d[None, None, :] < half,
                k * wc[:, None, :] - paired * ws[:, None, :],
                k * wc[:, None, :] + paired * ws[:, None, :],
            )
            * inv[:, :, None]
        )
        normal = tl.where(d[None, None, :] < RD, rotated, normal)
    out = (
        (layer * T + token[:, None, None]) * H + head[None, :, None]
    ) * D + d[None, None, :]
    v = tl.load(KV + base + H * D + d[None, None, :], valid, 0)
    tl.store(KO + out, normal, valid)
    tl.store(VO + out, v, valid)


def wide_fused_norm_rope_stacked(
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
    bt = 2 if t * layers >= 512 else 1
    wide__pair_gather.run(
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
        bt,
        grid=(triton.cdiv(t, bt), layers),
        warmup=False,
        num_warps=4,
        num_stages=2,
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
    if kv.shape[0] * kv.shape[1] >= 512:
        return wide_fused_norm_rope_stacked(
            kv,
            k_norm_weight,
            eps,
            cos_sin_cache,
            positions,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
    return small_fused_norm_rope_stacked(
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
