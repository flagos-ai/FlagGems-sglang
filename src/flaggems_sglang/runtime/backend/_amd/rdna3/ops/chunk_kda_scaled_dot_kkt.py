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

"""chunk_kda_scaled_dot_kkt for RDNA3 (gfx1100).

A 64-row chunk splits into 16-row blocks, and each program computes one
block of the lower triangle on the matrix units. exp2(g_i - g_j) factors
into exp2(g_i - g_a) * exp2(g_a - g_j), which scale q and k: g_a is the
block's first row for an earlier column block (both factors at most 1 when
g decreases down the chunk) and its middle row for the diagonal block. The
scaled values go to the matrix units as fp16 high and low parts (three
products, about 22 bits). A block whose factors or values would leave that
range is summed exactly in fp32 instead. Inputs below 4096 (batch, time,
head) rows, strided inputs and other chunk sizes keep the generic op.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.chunk_kda_scaled_dot_kkt import (
    chunk_kda_scaled_dot_kkt as _generic,
)


@triton.jit
def _split16(x):
    hi = x.to(tl.float16)
    return hi, (x - hi.to(tl.float32)).to(tl.float16)


@triton.jit
def _exact(
    Q,
    K,
    G,
    b,
    h,
    ri,
    c0,
    T,
    SQ0,
    SQ1,
    SQ2,
    SQ3,
    SK0,
    SK1,
    SK2,
    SK3,
    SG0,
    SG1,
    SG2,
    SG3,
    gs,
    D: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
    DK: tl.constexpr,
    DIAG: tl.constexpr,
):
    # sum_d k_i k_j exp2(g_i - g_j) as written, one column per step; masked
    # entries get an exponent of -inf.
    jj = tl.arange(0, BC)
    acc_k = tl.zeros((BC, BC), tl.float32)
    acc_q = tl.zeros((BC, BC), tl.float32)
    for d0 in range(0, BD, DK):
        d = d0 + tl.arange(0, DK)
        mi = (ri[:, None] < T) & (d[None, :] < D)
        ki = tl.load(
            K + b * SK0 + ri[:, None] * SK1 + h * SK2 + d[None, :] * SK3,
            mi,
            0,
        ).to(tl.float32)
        qi = tl.load(
            Q + b * SQ0 + ri[:, None] * SQ1 + h * SQ2 + d[None, :] * SQ3,
            mi,
            0,
        ).to(tl.float32)
        gi = tl.load(
            G + b * SG0 + ri[:, None] * SG1 + h * SG2 + d[None, :] * SG3,
            mi,
            0,
        ).to(tl.float32)
        for j in range(BC):
            cj = c0 + j
            mj = (d < D) & (cj < T)
            kj = tl.load(K + b * SK0 + cj * SK1 + h * SK2 + d * SK3, mj, 0).to(
                tl.float32
            )
            gj = tl.load(G + b * SG0 + cj * SG1 + h * SG2 + d * SG3, mj, 0).to(
                tl.float32
            )
            keep = (ri < T) & (cj < T)
            if DIAG:
                keep = keep & (ri >= cj)
            expo = tl.where(
                keep[:, None], (gi - gj[None, :]) * gs, float("-inf")
            )
            wk = kj[None, :] * tl.exp2(expo)
            hit = jj[None, :] == j
            acc_k += tl.where(hit, tl.sum(ki * wk, 1)[:, None], 0.0)
            acc_q += tl.where(hit, tl.sum(qi * wk, 1)[:, None], 0.0)
    return acc_k, acc_q


@triton.jit
def _kkt_kernel(
    Q,
    K,
    G,
    BETA,
    A,
    AQ,
    T,
    scale,
    gs,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    DK: tl.constexpr,
):
    # (row block, column block <= row block) pairs of a 4-block chunk.
    NB: tl.constexpr = BT // BC
    NP: tl.constexpr = NB * (NB + 1) // 2
    SQ0 = T * H * D
    SQ1: tl.constexpr = H * D
    SQ2: tl.constexpr = D
    SQ3: tl.constexpr = 1
    SK0, SK1, SK2, SK3 = SQ0, SQ1, SQ2, SQ3
    SG0, SG1, SG2, SG3 = SQ0, SQ1, SQ2, SQ3
    SB0 = T * H
    SB1: tl.constexpr = H
    SB2: tl.constexpr = 1
    pid = tl.program_id(0)
    p = pid % NP
    h = pid // NP % H
    rest = pid // (NP * H)
    nt = tl.cdiv(T, BT)
    chunk = rest % nt
    b = rest // nt
    blk = (p >= 1).to(tl.int32) + (p >= 3).to(tl.int32) + (p >= 6).to(tl.int32)
    blk_j = p - blk * (blk + 1) // 2
    c0 = chunk * BT
    r0 = c0 + blk * BC
    if r0 < T:
        ri = r0 + tl.arange(0, BC)
        jj = tl.arange(0, BC)
        cj = c0 + blk_j * BC + jj
        diag = blk_j == blk
        ar = tl.where(diag, tl.minimum(r0 + BC // 2, T - 1), r0)
        lim = tl.where(diag, 4096.0, 2.0)
        acc_k = tl.zeros((BC, BC), tl.float32)
        acc_q = tl.zeros((BC, BC), tl.float32)
        ones = tl.full((BC, BC), 1.0, tl.float32)
        ok = True
        for d0 in range(0, BD, DK):
            d = d0 + tl.arange(0, DK)
            dm = d < D
            mr = (ri[:, None] < T) & dm[None, :]
            mc = (cj[:, None] < T) & dm[None, :]
            ga = tl.load(G + b * SG0 + ar * SG1 + h * SG2 + d * SG3, dm, 0).to(
                tl.float32
            )
            gr = tl.load(
                G + b * SG0 + ri[:, None] * SG1 + h * SG2 + d[None, :] * SG3,
                mr,
                0,
            ).to(tl.float32)
            gc = tl.load(
                G + b * SG0 + cj[:, None] * SG1 + h * SG2 + d[None, :] * SG3,
                mc,
                0,
            ).to(tl.float32)
            fr = tl.exp2(tl.where(mr, gr - ga[None, :], 0.0) * gs)
            fc = tl.exp2(tl.where(mc, ga[None, :] - gc, 0.0) * gs)
            kr = (
                tl.load(
                    K
                    + b * SK0
                    + ri[:, None] * SK1
                    + h * SK2
                    + d[None, :] * SK3,
                    mr,
                    0,
                ).to(tl.float32)
                * fr
            )
            qr = (
                tl.load(
                    Q
                    + b * SQ0
                    + ri[:, None] * SQ1
                    + h * SQ2
                    + d[None, :] * SQ3,
                    mr,
                    0,
                ).to(tl.float32)
                * fr
            )
            kc = (
                tl.load(
                    K
                    + b * SK0
                    + cj[:, None] * SK1
                    + h * SK2
                    + d[None, :] * SK3,
                    mc,
                    0,
                ).to(tl.float32)
                * fc
            )
            fmax = tl.maximum(
                tl.max(tl.max(fr, 1), 0), tl.max(tl.max(fc, 1), 0)
            )
            fmin = tl.minimum(
                tl.min(tl.min(fr, 1), 0), tl.min(tl.min(fc, 1), 0)
            )
            vmax = tl.maximum(
                tl.max(tl.max(tl.abs(kc), 1), 0),
                tl.maximum(
                    tl.max(tl.max(tl.abs(kr), 1), 0),
                    tl.max(tl.max(tl.abs(qr), 1), 0),
                ),
            )
            ok = (
                ok
                & (fmax <= lim)
                & ((fmin * lim >= 1.0) | (blk_j != blk))
                & (vmax < 30000.0)
            )
            kr_hi, kr_lo = _split16(kr)
            qr_hi, qr_lo = _split16(qr)
            kc_hi, kc_lo = _split16(kc)
            kc_hi = tl.trans(kc_hi)
            kc_lo = tl.trans(kc_lo)
            # A dot accumulating into acc truncates the fp32 addition.
            acc_k = tl.fma(tl.dot(kr_hi, kc_hi), ones, acc_k)
            acc_k = tl.fma(tl.dot(kr_hi, kc_lo), ones, acc_k)
            acc_k = tl.fma(tl.dot(kr_lo, kc_hi), ones, acc_k)
            acc_q = tl.fma(tl.dot(qr_hi, kc_hi), ones, acc_q)
            acc_q = tl.fma(tl.dot(qr_hi, kc_lo), ones, acc_q)
            acc_q = tl.fma(tl.dot(qr_lo, kc_hi), ones, acc_q)
        if not ok:
            if diag:
                acc_k, acc_q = _exact(
                    Q,
                    K,
                    G,
                    b,
                    h,
                    ri,
                    r0,
                    T,
                    SQ0,
                    SQ1,
                    SQ2,
                    SQ3,
                    SK0,
                    SK1,
                    SK2,
                    SK3,
                    SG0,
                    SG1,
                    SG2,
                    SG3,
                    gs,
                    D,
                    BD,
                    BC,
                    DK,
                    True,
                )
            else:
                acc_k, acc_q = _exact(
                    Q,
                    K,
                    G,
                    b,
                    h,
                    ri,
                    c0 + blk_j * BC,
                    T,
                    SQ0,
                    SQ1,
                    SQ2,
                    SQ3,
                    SK0,
                    SK1,
                    SK2,
                    SK3,
                    SG0,
                    SG1,
                    SG2,
                    SG3,
                    gs,
                    D,
                    BD,
                    BC,
                    DK,
                    False,
                )
        beta = tl.load(BETA + b * SB0 + ri * SB1 + h * SB2, ri < T, 0).to(
            tl.float32
        )
        out = ((b * T + ri[:, None]) * H + h) * BT + jj[None, :]
        rows = ri[:, None] < T
        keep_k = (jj[:, None] > jj[None, :]) | (blk_j != blk)
        keep_q = (jj[:, None] >= jj[None, :]) | (blk_j != blk)
        tl.store(
            A + out + blk_j * BC,
            tl.where(keep_k, acc_k * beta[:, None], 0.0),
            rows,
        )
        tl.store(
            AQ + out + blk_j * BC, tl.where(keep_q, acc_q * scale, 0.0), rows
        )
        if diag:
            zero = tl.zeros((BC, BC), tl.float32)
            for blk_z in range(blk + 1, NB):
                tl.store(A + out + blk_z * BC, zero, rows)
                tl.store(AQ + out + blk_z * BC, zero, rows)


def chunk_kda_scaled_dot_kkt(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    if (
        int(chunk_size) != 64
        or d > 256
        or b * t * h < 4096
        or not all(x.is_contiguous() for x in (q, k, gk, beta))
    ):
        return _generic(q, k, gk, beta, scale, gk_scale, chunk_size)
    bt = 64
    a = torch.empty((b, t, h, bt), device=k.device, dtype=torch.float32)
    aq = torch.empty_like(a)
    grid = b * h * triton.cdiv(t, bt) * 10
    if grid:
        _kkt_kernel[(grid,)](
            q,
            k,
            gk,
            beta,
            a,
            aq,
            t,
            float(scale),
            float(gk_scale),
            h,
            d,
            max(32, triton.next_power_of_2(d)),
            bt,
            16,
            32,
            num_warps=1,
        )
    return a, aq


__all__ = ["chunk_kda_scaled_dot_kkt"]
