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
def _paired_sum(x, N: tl.constexpr):
    for level in tl.static_range(0, 8):
        if N >= (1 << (level + 1)):
            x = tl.sum(tl.reshape(x, (N >> (level + 1), 2)), 1)
    return tl.sum(x, 0)


@triton.jit
def _elementary64(X, Y, N: tl.constexpr, KIND: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    if KIND == 0:
        y = libdevice.exp(x)
    else:
        y = libdevice.sqrt(x)
    tl.store(Y + i, y, i < N)


@triton.jit
def _square_sum(X, Y, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    x = tl.load(X + row * K + i, i < K, 0).to(tl.float32)
    y = _paired_sum(x * x, B) + 1.0e-6
    tl.store(Y + row, y)


@triton.jit
def _divide64(X, D, Y, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    d = tl.load(D + i // K, i < N, 1)
    tl.store(Y + i, x / d, i < N)


def _elementary_buffer(x, kind):
    x64 = x.float().double().contiguous()
    y64 = torch.empty_like(x64)
    _elementary64[(triton.cdiv(x.numel(), 128),)](
        x64, y64, x.numel(), kind, 128, enable_fp_fusion=False
    )
    return y64.float()


def _normalize_buffer(x):
    xf = x.float().contiguous()
    kd = x.shape[-1]
    sums = torch.empty(xf.shape[:-1], device=x.device, dtype=torch.float32)
    _square_sum[(x.numel() // kd,)](
        xf, sums, kd, triton.next_power_of_2(kd), enable_fp_fusion=False
    )
    den = _elementary_buffer(sums, 1).double().contiguous()
    x64 = xf.double()
    y64 = torch.empty_like(x64)
    _divide64[(triton.cdiv(x.numel(), 128),)](
        x64, den, y64, x.numel(), kd, 128, enable_fp_fusion=False
    )
    return y64.float()


@triton.jit
def _sum_rows(x, RV: tl.constexpr, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            x = tl.sum(tl.reshape(x, (RV, BK >> (level + 1), 2)), 2)
    return tl.sum(x, 1)


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
    SCALE: tl.constexpr,
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
    for ti in tl.range(0, T, loop_unroll_factor=4):
        token = (bi * T + ti) * H + hi
        q = tl.load(Q + token * KD + kk, kk < KD, 0).to(tl.float32)
        q = tl.fma(q, SCALE, -0.0)
        k = tl.load(K + token * KD + kk, kk < KD, 0).to(tl.float32)
        d = tl.load(D + token * KD + kk, kk < KD, 0).to(tl.float32)
        state = tl.fma(state, d[None, :], -0.0)
        pred = _sum_rows(tl.fma(state, k[None, :], -0.0), RV, BK)
        value = tl.load(
            V + bi * VS[0] + ti * VS[1] + hi * VS[2] + vv * VS[3], vv < VD, 0
        ).to(tl.float32)
        if PER_V:
            beta = tl.load(
                Beta + bi * BS[0] + ti * BS[1] + hi * BS[2] + vv * BS[3],
                vv < VD,
                0,
            ).to(tl.float32)
        else:
            beta = tl.load(Beta + bi * BS[0] + ti * BS[1] + hi * BS[2]).to(
                tl.float32
            )
        delta = tl.fma(value - pred, beta, -0.0)
        state = tl.where(
            kk[None, :] < KD,
            state + tl.fma(delta[:, None], k[None, :], -0.0),
            0.0,
        )
        out = _sum_rows(tl.fma(state, q[None, :], -0.0), RV, BK)
        tl.store(O + token * VD + vv, out, vv < VD)
    tl.store(S1 + (bh * VD + vv[:, None]) * KD + kk[None, :], state, valid)


def fused_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel
):
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    g64 = g.float().double().contiguous()
    decay = torch.empty_like(g64)
    if g.numel():
        _elementary64[(triton.cdiv(g.numel(), 512),)](
            g64, decay, g.numel(), 0, 512, num_warps=4, enable_fp_fusion=False
        )
    qprep = (
        _normalize_buffer(q)
        if use_qk_l2norm_in_kernel and b * t * h
        else q.contiguous()
    )
    kprep = (
        _normalize_buffer(k)
        if use_qk_l2norm_in_kernel and b * t * h
        else k.contiguous()
    )
    output = torch.empty((b, t, h, vd), dtype=q.dtype, device=q.device)
    final_state = torch.empty(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    rv = 1 if b * h * vd <= 1024 else 4
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
            scale,
            beta.ndim == v.ndim,
            rv,
            triton.next_power_of_2(kd),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return output, final_state


__all__ = ["fused_recurrent_kda"]
