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

"""fused_recurrent_kda for RDNA3 (gfx1100).

Same kernels and arithmetic as the generic op; only the tiles change. Each
row of the state evolves independently, so a recurrent program can own any
number of value rows: mid-size batches split the state into more, smaller
programs to fill the GPU, and decode gives each program more warps. With
the L2 norm, the preparation kernel gives each token row one warp, so its
sums stay within the warp.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _sum_rows(x, RV: tl.constexpr, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            x = tl.sum(tl.reshape(x, (RV, BK >> (level + 1), 2)), 2)
    return tl.sum(x, 1)


@triton.jit
def _prepare(
    Q,
    K,
    G,
    QP,
    KP,
    D,
    N,
    H: tl.constexpr,
    T: tl.constexpr,
    KD: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    SCALE: tl.constexpr,
    NORM: tl.constexpr,
    BK: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    hi = rows % H
    ti = rows // H % T
    bi = rows // (T * H)
    kk = tl.arange(0, BK)
    valid = (rows[:, None] < N) & (kk[None, :] < KD)
    q = tl.load(
        Q + (bi * QS[0] + ti * QS[1] + hi * QS[2])[:, None] + kk * QS[3],
        valid,
        0,
    ).to(tl.float32)
    k = tl.load(
        K + (bi * KS[0] + ti * KS[1] + hi * KS[2])[:, None] + kk * KS[3],
        valid,
        0,
    ).to(tl.float32)
    g = tl.load(
        G + (bi * GS[0] + ti * GS[1] + hi * GS[2])[:, None] + kk * GS[3],
        valid,
        0,
    ).to(tl.float32)
    if NORM:
        qd = libdevice.sqrt(
            (_sum_rows(tl.fma(q, q, -0.0), R, BK) + 1.0e-6).to(tl.float64)
        ).to(tl.float32)
        kd = libdevice.sqrt(
            (_sum_rows(tl.fma(k, k, -0.0), R, BK) + 1.0e-6).to(tl.float64)
        ).to(tl.float32)
        q = (q.to(tl.float64) / qd.to(tl.float64)[:, None]).to(tl.float32)
        k = (k.to(tl.float64) / kd.to(tl.float64)[:, None]).to(tl.float32)
    q = tl.fma(q, SCALE, -0.0)
    d = libdevice.exp(g.to(tl.float64)).to(tl.float32)
    offs = rows[:, None] * KD + kk[None, :]
    tl.store(QP + offs, q, valid)
    tl.store(KP + offs, k, valid)
    tl.store(D + offs, d, valid)


@triton.jit
def _read_token(
    Q,
    K,
    D,
    V,
    Beta,
    bi,
    hi,
    ti,
    kk,
    vv,
    T: tl.constexpr,
    H: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    VS: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    PER_V: tl.constexpr,
):
    token = (bi * T + ti) * H + hi
    q = tl.load(Q + token * KD + kk, (kk < KD) & (ti < T), 0).to(tl.float32)
    if SCALE != 1.0:
        q = tl.fma(q, SCALE, -0.0)
    k = tl.load(K + token * KD + kk, (kk < KD) & (ti < T), 0).to(tl.float32)
    d = tl.load(D + token * KD + kk, (kk < KD) & (ti < T), 0).to(tl.float32)
    value = tl.load(
        V + bi * VS[0] + ti * VS[1] + hi * VS[2] + vv * VS[3],
        (vv < VD) & (ti < T),
        0,
    ).to(tl.float32)
    if PER_V:
        beta = tl.load(
            Beta + bi * BS[0] + ti * BS[1] + hi * BS[2] + vv * BS[3],
            (vv < VD) & (ti < T),
            0,
        ).to(tl.float32)
    else:
        beta = tl.load(
            Beta + bi * BS[0] + ti * BS[1] + hi * BS[2], ti < T, 0
        ).to(tl.float32)
    return q, k, d, value, beta


@triton.jit
def _recurrent(
    Q,
    K,
    D,
    V,
    Beta,
    S0,
    O,
    S1,
    T: tl.constexpr,
    H: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    VS: tl.constexpr,
    BS: tl.constexpr,
    PER_V: tl.constexpr,
    RV: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles: tl.constexpr = triton.cdiv(VD, RV)
    bh = pid // tiles
    bi = bh // H
    hi = bh % H
    vv = (pid % tiles) * RV + tl.arange(0, RV)
    kk = tl.arange(0, BK)
    valid = (vv[:, None] < VD) & (kk[None, :] < KD)
    state = tl.load(
        S0 + (bh * VD + vv[:, None]) * KD + kk[None, :], valid, 0
    ).to(tl.float32)
    q, k, d, value, beta = _read_token(
        Q, K, D, V, Beta, bi, hi, 0, kk, vv, T, H, KD, VD, VS, BS, 1.0, PER_V
    )
    for ti in tl.range(0, T, loop_unroll_factor=8):
        nq, nk, nd, nv, nb = _read_token(
            Q,
            K,
            D,
            V,
            Beta,
            bi,
            hi,
            ti + 1,
            kk,
            vv,
            T,
            H,
            KD,
            VD,
            VS,
            BS,
            1.0,
            PER_V,
        )
        state = tl.fma(state, d[None, :], -0.0)
        pred = _sum_rows(tl.fma(state, k[None, :], -0.0), RV, BK)
        delta = tl.fma(value - pred, beta, -0.0)
        state = tl.where(
            kk[None, :] < KD,
            state + tl.fma(delta[:, None], k[None, :], -0.0),
            0.0,
        )
        out = _sum_rows(tl.fma(state, q[None, :], -0.0), RV, BK)
        token = (bi * T + ti) * H + hi
        tl.store(O + token * VD + vv, out, vv < VD)
        q, k, d, value, beta = nq, nk, nd, nv, nb
    tl.store(S1 + (bh * VD + vv[:, None]) * KD + kk[None, :], state, valid)


def _config(b, t, h, vd):
    """Value rows per recurrent program and num_warps.

    Decode (t == 1) wants more warps and mid-size prefill RV=2; large grids
    keep the generic (4, 1), so even the NaNs of an overflowing state keep
    their sign. Avoid 16-row tiles with 4 warps: on gfx1100 they corrupt
    some outputs, differently on every run.
    """
    work = b * h * vd
    if t == 1 and work >= 4096:
        return 4, 8
    if work <= 1024:
        return 1, 1
    if work <= 16384:
        return 2, 1
    return 4, 1


def fused_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel
):
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    qprep = torch.empty((b, t, h, kd), dtype=torch.float32, device=q.device)
    kprep = torch.empty_like(qprep)
    decay = torch.empty_like(qprep)
    output = torch.empty((b, t, h, vd), dtype=q.dtype, device=q.device)
    final_state = torch.empty(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    bk = triton.next_power_of_2(kd)
    rv, nw = _config(b, t, h, vd)
    rows = b * t * h
    pr, pw = (2, 2) if use_qk_l2norm_in_kernel and rows >= 128 else (1, 4)
    if rows:
        _prepare[(triton.cdiv(rows, pr),)](
            q,
            k,
            g,
            qprep,
            kprep,
            decay,
            rows,
            h,
            t,
            kd,
            q.stride(),
            k.stride(),
            g.stride(),
            scale,
            use_qk_l2norm_in_kernel,
            bk,
            pr,
            num_warps=pw,
            enable_fp_fusion=False,
        )
    if b * h * vd:
        _recurrent[(b * h * triton.cdiv(vd, rv),)](
            qprep,
            kprep,
            decay,
            v,
            beta,
            initial_state.contiguous(),
            output,
            final_state,
            t,
            h,
            kd,
            vd,
            v.stride(),
            beta.stride(),
            beta.ndim == v.ndim,
            rv,
            bk,
            num_warps=nw,
            enable_fp_fusion=False,
        )
    return output, final_state


__all__ = ["fused_recurrent_kda"]
