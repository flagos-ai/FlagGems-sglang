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
def _pair_sum(ah, al, bh, bl):
    s = ah + bh
    bv = s - ah
    e = ah - (s - bv) + (bh - bv)
    e = e + al + bl
    high = s + e
    low = e - (high - s)
    return (high, low)


@triton.jit
def _weighted_sum(x, y, decay):
    product = x * y
    product_low = tl.fma(x, y, -product)
    high = product * decay
    low = tl.fma(product, decay, -high) + product_low * decay
    high, low = tl.reduce((high, low), 1, _pair_sum)
    return high + low


@triton.jit
def _compute(Q, K, G, BETA, A, AQ, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SG: tl.constexpr = META[6]
    SB: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    GS: tl.constexpr = META[9]
    BC: tl.constexpr = 4
    BD: tl.constexpr = triton.next_power_of_2(D)
    nc: tl.constexpr = triton.cdiv(BT, BC)
    p = tl.program_id(0)
    j = p % nc * BC + tl.arange(0, BC)
    row = p // nc
    h = row % H
    t = row // H % T
    b = row // (T * H)
    i = t % BT
    tj = t // BT * BT + j
    c = tl.arange(0, BD)
    out = row * BT + j
    q = tl.load(
        Q + b * SQ[0] + t * SQ[1] + h * SQ[2] + c * SQ[3], c < D, 0
    ).to(tl.float32)
    k = tl.load(
        K + b * SK[0] + t * SK[1] + h * SK[2] + c * SK[3], c < D, 0
    ).to(tl.float32)
    g = (
        tl.load(
            G + b * SG[0] + t * SG[1] + h * SG[2] + c * SG[3], c < D, 0
        ).to(tl.float32)
        * GS
    )
    valid = c[None, :] < D
    tj_load = tl.minimum(tj, T - 1)
    kj = tl.load(
        K
        + b * SK[0]
        + tj_load[:, None] * SK[1]
        + h * SK[2]
        + c[None, :] * SK[3],
        valid,
        0,
    ).to(tl.float32)
    gj = (
        tl.load(
            G
            + b * SG[0]
            + tj_load[:, None] * SG[1]
            + h * SG[2]
            + c[None, :] * SG[3],
            valid,
            0,
        ).to(tl.float32)
        * GS
    )
    decay = tl.exp((g[None, :] - gj) * 0.6931471805599453)
    av = _weighted_sum(k[None, :], kj, decay)
    qv = _weighted_sum(q[None, :], kj, decay)
    beta = tl.load(BETA + b * SB[0] + t * SB[1] + h * SB[2]).to(tl.float32)
    av = tl.where((i > j) & (tj < T), av * beta, 0.0)
    qv = tl.where((i >= j) & (tj < T), qv * SCALE, 0.0)
    tl.store(A + out, av, j < BT)
    tl.store(AQ + out, qv, j < BT)


def _vector_entry(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    bt = int(chunk_size)
    a = torch.empty((b, t, h, bt), device=k.device, dtype=torch.float32)
    aq = torch.empty_like(a)
    total = b * t * h * triton.cdiv(bt, 4)
    if total:
        meta = (
            t,
            h,
            d,
            bt,
            q.stride(),
            k.stride(),
            gk.stride(),
            beta.stride(),
            scale,
            gk_scale,
        )
        _compute.run(
            q,
            k,
            gk,
            beta,
            a,
            aq,
            meta,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(total,),
            warmup=False,
        )
    return (a, aq)


@triton.jit
def _energy_guard(
    Q,
    K,
    FLAGS,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    group = row // 64
    r = row % 64
    chunk = group % (T // 64)
    h = group // (T // 64) % H
    b = group // (T // 64 * H)
    c = tl.arange(0, D)
    address = ((b * T + chunk * 64 + r) * H + h) * D + c
    q = tl.load(Q + address).to(tl.float32)
    k = tl.load(K + address).to(tl.float32)
    eq = tl.sum(q * q, 0)
    ek = tl.sum(k * k, 0)
    tl.store(FLAGS + GROUPS + row, ((eq < 256.0) & (ek < 256.0)).to(tl.int32))


@triton.jit
def _prepare(Q, K, G, W, FLAGS, A, AQ, BETA, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    GS: tl.constexpr = META[3]
    BLOCK: tl.constexpr = META[4]
    SB: tl.constexpr = META[5]
    SCALE: tl.constexpr = META[6]
    p = tl.program_id(0)
    nt0: tl.constexpr = T // 64
    group = (p // (nt0 * H) * H + p % H) * nt0 + p // H % nt0
    nt: tl.constexpr = T // 64
    chunk = group % nt
    h = group // nt % H
    b = group // (nt * H)
    off = tl.arange(0, BLOCK)
    r = off // D
    c = off % D
    addr = ((b * T + chunk * 64 + r) * H + h) * D + c
    g = tl.load(G + addr).to(tl.float32) * GS
    q = tl.load(Q + addr).to(tl.float32)
    k = tl.load(K + addr).to(tl.float32)
    anchor_addr = ((b * T + chunk * 64 + 32) * H + h) * D
    anchor = tl.load(G + anchor_addr).to(tl.float32) * GS
    bound = tl.maximum(
        tl.abs(g - anchor) * 0.125, tl.maximum(tl.abs(q), tl.abs(k)) * 0.015625
    )
    prev_addr = ((b * T + chunk * 64 + tl.maximum(r - 1, 0)) * H + h) * D + c
    prev_g = tl.load(G + prev_addr).to(tl.float32) * GS
    growth = tl.max(g - prev_g, 0)
    energy_rows = r
    energy_ok = (
        tl.min(tl.load(FLAGS + META[7] + group * 64 + energy_rows), 0) != 0
    )
    guard_rows = r
    guard_beta = tl.load(
        BETA + b * SB[0] + (chunk * 64 + guard_rows) * SB[1] + h * SB[2]
    ).to(tl.float32)
    magnitude = tl.maximum(tl.max(tl.abs(q), 0), tl.max(tl.abs(k), 0))
    safe = (
        (tl.max(bound, 0) < 1.0)
        & (magnitude < 6.0)
        & (growth <= 0.0009765625)
        & energy_ok
        & (tl.max(tl.abs(guard_beta), 0) <= 1.0)
        & (tl.abs(SCALE) <= 1.0)
    )
    delta = tl.where(safe, g - anchor, 0.0)
    z = delta * 0.02166084939249829
    z2 = z * z
    even = (0.041718827117492176 * z2 + 0.4999996083749389) * z2 + 1.0
    odd = z * (0.16697968532092686 * z2 + 0.9999976497245077)
    left = even + odd
    right = even - odd
    left = left * left
    right = right * right
    left = left * left
    right = right * right
    left = left * left
    right = right * right
    left = left * left
    right = right * right
    left = left * left
    right = right * right
    tl.store(W + (group * 6 + 0) * BLOCK + off, q * left)
    qh = (q * left).to(tl.float16).to(tl.float32)
    tl.store(W + (group * 6 + 0 + 1) * BLOCK + off, q * left - qh)
    tl.store(W + (group * 6 + 2) * BLOCK + off, k * left)
    kh = (k * left).to(tl.float16).to(tl.float32)
    tl.store(W + (group * 6 + 2 + 1) * BLOCK + off, k * left - kh)
    tl.store(W + (group * 6 + 4) * BLOCK + off, k * right)
    rh = (k * right).to(tl.float16).to(tl.float32)
    tl.store(W + (group * 6 + 4 + 1) * BLOCK + off, k * right - rh)
    tl.store(FLAGS + group, safe.to(tl.int32))


@triton.jit
def _repair(Q, K, G, BETA, A, AQ, FLAGS, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SG: tl.constexpr = META[6]
    SB: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    GS: tl.constexpr = META[9]
    BC: tl.constexpr = 16
    BD: tl.constexpr = triton.next_power_of_2(D)
    nc: tl.constexpr = triton.cdiv(BT, BC)
    p = tl.program_id(0)
    j = p % nc * BC + tl.arange(0, BC)
    row = p // nc
    h = row % H
    t = row // H % T
    b = row // (T * H)
    i = t % BT
    tj = t // BT * BT + j
    c = tl.arange(0, BD)
    group = (b * H + h) * (T // 64) + t // 64
    out = group * 4096 + i * 64 + j
    for repair_iteration in range(1 - tl.load(FLAGS + group)):
        q = tl.load(
            Q + b * SQ[0] + t * SQ[1] + h * SQ[2] + c * SQ[3], c < D, 0
        ).to(tl.float32)
        k = tl.load(
            K + b * SK[0] + t * SK[1] + h * SK[2] + c * SK[3], c < D, 0
        ).to(tl.float32)
        g = (
            tl.load(
                G + b * SG[0] + t * SG[1] + h * SG[2] + c * SG[3], c < D, 0
            ).to(tl.float32)
            * GS
        )
        valid = c[None, :] < D
        tj_load = tl.minimum(tj, T - 1)
        kj = tl.load(
            K
            + b * SK[0]
            + tj_load[:, None] * SK[1]
            + h * SK[2]
            + c[None, :] * SK[3],
            valid,
            0,
        ).to(tl.float32)
        gj = (
            tl.load(
                G
                + b * SG[0]
                + tj_load[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                valid,
                0,
            ).to(tl.float32)
            * GS
        )
        decay = tl.exp((g[None, :] - gj) * 0.6931471805599453)
        av = _weighted_sum(k[None, :], kj, decay)
        qv = _weighted_sum(q[None, :], kj, decay)
        beta = tl.load(BETA + b * SB[0] + t * SB[1] + h * SB[2]).to(tl.float32)
        av = tl.where((i > j) & (tj < T), av * beta, 0.0)
        qv = tl.where((i >= j) & (tj < T), qv * SCALE, 0.0)
        tl.store(A + out, av, j < BT)
        tl.store(AQ + out, qv, j < BT)


@triton.jit
def _matrix(W, FLAGS, A, AQ, BETA, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SB: tl.constexpr = META[3]
    SCALE: tl.constexpr = META[4]
    p = tl.program_id(0)
    nt0: tl.constexpr = T // 64
    group = (p // (nt0 * H) * H + p % H) * nt0 + p // H % nt0
    for safe_iteration in range(tl.load(FLAGS + group)):
        i = tl.arange(0, 64)
        j = tl.arange(0, 64)
        c = tl.arange(0, D)
        k = tl.load(W + (group * 384 + 128) * D + i[:, None] * D + c[None, :])
        kl = tl.load(W + (group * 384 + 192) * D + i[:, None] * D + c[None, :])
        right = tl.load(
            W + (group * 384 + 256) * D + j[None, :] * D + c[:, None]
        )
        rl = tl.load(W + (group * 384 + 320) * D + j[None, :] * D + c[:, None])
        av = tl.dot(k, rl, input_precision="ieee")
        av = tl.dot(kl, right, av, input_precision="ieee")
        av = tl.dot(k, right, av, input_precision="ieee")
        nt: tl.constexpr = T // 64
        chunk = group % nt
        h = group // nt % H
        b = group // (nt * H)
        betas = tl.load(
            BETA + b * SB[0] + (chunk * 64 + i) * SB[1] + h * SB[2]
        ).to(tl.float32)
        av = tl.where(j[None, :] < i[:, None], av * betas[:, None], 0.0)
        tl.store(A + group * 4096 + i[:, None] * 64 + j[None, :], av)
        q = tl.load(W + (group * 384 + 0) * D + i[:, None] * D + c[None, :])
        ql = tl.load(W + (group * 384 + 64) * D + i[:, None] * D + c[None, :])
        qv = tl.dot(q, rl, input_precision="ieee")
        qv = tl.dot(ql, right, qv, input_precision="ieee")
        qv = tl.dot(q, right, qv, input_precision="ieee")
        qv = tl.where(j[None, :] <= i[:, None], qv * SCALE, 0.0)
        tl.store(AQ + group * 4096 + i[:, None] * 64 + j[None, :], qv)


def chunk_kda_scaled_dot_kkt(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    if (
        chunk_size != 64
        or t % 64
        or d not in (64, 128)
        or (not q.is_contiguous())
        or (not k.is_contiguous())
        or (not gk.is_contiguous())
    ):
        return _vector_entry(q, k, gk, beta, scale, gk_scale, chunk_size)
    groups = b * h * (t // 64)
    pa = torch.empty((b, h, t, 64), dtype=torch.float32, device=k.device)
    pqa = torch.empty_like(pa)
    if groups:
        workspace = torch.empty(
            (groups, 384, d), dtype=torch.float16, device=k.device
        )
        flags = torch.empty((groups * 65,), dtype=torch.int32, device=k.device)
        _energy_guard.run(
            q,
            k,
            flags,
            t,
            h,
            d,
            groups,
            grid=(groups * 64,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _prepare.run(
            q,
            k,
            gk,
            workspace,
            flags,
            pa,
            pqa,
            beta,
            (t, h, d, gk_scale, 64 * d, beta.stride(), scale, groups),
            isCloseMemoryAsync=False,
            enable_fp_fusion=False,
            grid=(groups,),
            warmup=False,
            budget_tiling=False,
            unroll_num=8,
        )
        _matrix.run(
            workspace,
            flags,
            pa,
            pqa,
            beta,
            (t, h, d, beta.stride(), scale),
            num_warps=4,
            num_stages=1,
            grid=(groups,),
            warmup=False,
        )
        meta_repair = (
            t,
            h,
            d,
            64,
            q.stride(),
            k.stride(),
            gk.stride(),
            beta.stride(),
            scale,
            gk_scale,
        )
        _repair.run(
            q,
            k,
            gk,
            beta,
            pa,
            pqa,
            flags,
            meta_repair,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(b * t * h * 4,),
            warmup=False,
        )
    return (pa.permute(0, 2, 1, 3), pqa.permute(0, 2, 1, 3))


__all__ = ["chunk_kda_scaled_dot_kkt"]
