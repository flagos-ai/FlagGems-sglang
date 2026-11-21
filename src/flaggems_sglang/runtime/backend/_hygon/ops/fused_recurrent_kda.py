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
from triton.language.extra import libdevice


@triton.jit
def _sum1(x, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            x = tl.sum(tl.reshape(x, (BK >> (level + 1), 2)), 1)
    return tl.sum(x, 0)


@triton.jit
def _sum_rows(x, RV: tl.constexpr, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            x = tl.sum(tl.reshape(x, (BK >> (level + 1), 2, RV)), 1)
    return tl.sum(x, 0)


@triton.jit
def _prepare(
    Q,
    K,
    G,
    QP,
    KP,
    D,
    H: tl.constexpr,
    T: tl.constexpr,
    KD: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    SCALE: tl.constexpr,
    NORM: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    hi = row % H
    ti = row // H % T
    bi = row // (T * H)
    kk = tl.arange(0, BK)
    q = tl.load(
        Q + bi * QS[0] + ti * QS[1] + hi * QS[2] + kk * QS[3], kk < KD, 0
    ).to(tl.float32)
    k = tl.load(
        K + bi * KS[0] + ti * KS[1] + hi * KS[2] + kk * KS[3], kk < KD, 0
    ).to(tl.float32)
    g = tl.load(
        G + bi * GS[0] + ti * GS[1] + hi * GS[2] + kk * GS[3], kk < KD, 0
    ).to(tl.float32)
    if NORM:
        qd = libdevice.sqrt(
            (_sum1(tl.fma(q, q, -0.0), BK) + 1.0e-6).to(tl.float64)
        ).to(tl.float32)
        kd = libdevice.sqrt(
            (_sum1(tl.fma(k, k, -0.0), BK) + 1.0e-6).to(tl.float64)
        ).to(tl.float32)
        q = (q.to(tl.float64) / qd.to(tl.float64)).to(tl.float32)
        k = (k.to(tl.float64) / kd.to(tl.float64)).to(tl.float32)
    q = tl.fma(q, SCALE, -0.0)
    d = libdevice.exp(g.to(tl.float64)).to(tl.float32)
    tl.store(QP + row * KD + kk, q, kk < KD)
    tl.store(KP + row * KD + kk, k, kk < KD)
    tl.store(D + row * KD + kk, d, kk < KD)


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
    valid = (vv[None, :] < VD) & (kk[:, None] < KD)
    state = tl.load(
        S0 + (bh * VD + vv[None, :]) * KD + kk[:, None], valid, 0
    ).to(tl.float32)
    q, k, d, value, beta = _read_token(
        Q, K, D, V, Beta, bi, hi, 0, kk, vv, T, H, KD, VD, VS, BS, 1.0, PER_V
    )
    for ti in tl.range(0, T, loop_unroll_factor=4):
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
        state = tl.fma(state, d[:, None], -0.0)
        pred = _sum_rows(tl.fma(state, k[:, None], -0.0), RV, BK)
        delta = tl.fma(value - pred, beta, -0.0)
        state = tl.where(
            kk[:, None] < KD,
            state + tl.fma(delta[None, :], k[:, None], -0.0),
            0.0,
        )
        out = _sum_rows(tl.fma(state, q[:, None], -0.0), RV, BK)
        token = (bi * T + ti) * H + hi
        tl.store(O + token * VD + vv, out, vv < VD)
        q, k, d, value, beta = nq, nk, nd, nv, nb
    tl.store(S1 + (bh * VD + vv[None, :]) * KD + kk[:, None], state, valid)


@triton.jit
def _prepare_flat(
    Q,
    K,
    G,
    QP,
    KP,
    D,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + i, i < N, 0).to(tl.float32)
    k = tl.load(K + i, i < N, 0).to(tl.float32)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    d = libdevice.exp(g.to(tl.float64)).to(tl.float32)
    q = tl.fma(q, SCALE, -0.0)
    tl.store(QP + i, q, i < N)
    tl.store(KP + i, k, i < N)
    tl.store(D + i, d, i < N)


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
    if b * t * h:
        if (
            not use_qk_l2norm_in_kernel
            and q.is_contiguous()
            and k.is_contiguous()
            and g.is_contiguous()
        ):
            _prepare_flat[(triton.cdiv(q.numel(), 512),)](
                q,
                k,
                g,
                qprep,
                kprep,
                decay,
                q.numel(),
                scale,
                512,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _prepare[(b * t * h,)](
                q,
                k,
                g,
                qprep,
                kprep,
                decay,
                h,
                t,
                kd,
                q.stride(),
                k.stride(),
                g.stride(),
                scale,
                use_qk_l2norm_in_kernel,
                bk,
                num_warps=4,
                enable_fp_fusion=False,
            )
    if b * h * vd:
        _recurrent[(b * h * triton.cdiv(vd, 8),)](
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
            8,
            bk,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, final_state


__all__ = ["fused_recurrent_kda"]
