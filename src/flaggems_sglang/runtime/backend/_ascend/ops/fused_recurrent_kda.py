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
from triton.language.extra.cann.extension import extract_slice


@triton.jit
def _dekker_add(a, al, b, bl):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    e = (e + al) + bl
    h = s + e
    return h, e - (h - s)


@triton.jit
def _dekker_mul(a, al, b, bl):
    p = a * b
    ah = (a.to(tl.uint32, bitcast=True) & 0xFFFFF000).to(
        tl.float32, bitcast=True
    )
    bh = (b.to(tl.uint32, bitcast=True) & 0xFFFFF000).to(
        tl.float32, bitcast=True
    )
    ax = a - ah
    bx = b - bh
    e = (((ah * bh - p) + ah * bx) + ax * bh) + ax * bx
    e = e + (a * bl + al * b)
    e = e + al * bl
    h = p + e
    return h, e - (h - p)


@triton.jit
def _exp_dekker(x):
    xc = tl.minimum(tl.maximum(x, -104.0), 88.72283935546875)
    n = tl.floor(xc * 1.4426950408889634 + 0.5)
    nh = n * 0.6931471824645996
    nl = tl.fma(n, 0.6931471824645996, -nh) + n * -1.904654323148236e-9
    rh, rl = _dekker_add(xc - nh, 0.0, -nl, 0.0)
    ph = tl.full(x.shape, 2.0876755879584152e-09, tl.float32)
    pl = tl.full(x.shape, 1.1082839809204342e-16, tl.float32)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(
        ph, pl, 2.5052107943679403e-08, 4.4176230446483665e-16
    )
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 2.755731998149713e-07, -7.575112209051195e-15)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 2.7557318844628753e-06, 3.793571224297229e-14)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 2.4801587642286904e-05, -3.40699609366682e-13)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(
        ph, pl, 0.00019841270113829523, -2.725596874933456e-12
    )
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(
        ph, pl, 0.0013888889225199819, -3.3631094437103215e-11
    )
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 0.008333333767950535, -4.34617203337595e-10)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 0.0416666679084301, -1.2417634698280722e-09)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 0.1666666716337204, -4.967053879312289e-09)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 0.5, 0.0)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 1.0, 0.0)
    ph, pl = _dekker_mul(ph, pl, rh, rl)
    ph, pl = _dekker_add(ph, pl, 1.0, 0.0)
    exponent = n.to(tl.int32)
    first = tl.minimum(tl.maximum(exponent, -126), 127)
    second = exponent - first
    s1 = ((first + 127) << 23).to(tl.float32, bitcast=True)
    s2 = ((second + 127) << 23).to(tl.float32, bitcast=True)
    out = ((ph + pl) * s1) * s2
    out = tl.where(x < -104.0, 0.0, out)
    out = tl.where(x > 88.72283935546875, float("inf"), out)
    return out


@triton.jit
def _sum1(x, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            j = tl.arange(0, BK >> (level + 1))
            x = tl.gather(x, 2 * j, 0) + tl.gather(x, 2 * j + 1, 0)
    return tl.sum(x, 0)


@triton.jit
def _sum_rows(x, RV: tl.constexpr, BK: tl.constexpr):
    for level in tl.static_range(0, 8):
        if BK >= (1 << (level + 1)):
            a = extract_slice(x, (0, 0), (BK >> (level + 1), RV), (2, 1))
            b = extract_slice(x, (1, 0), (BK >> (level + 1), RV), (2, 1))
            x = a + b
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
        qd = tl.sqrt(_sum1(tl.fma(q, q, -0.0), BK) + 1.0e-6)
        kd = tl.sqrt(_sum1(tl.fma(k, k, -0.0), BK) + 1.0e-6)
        q = tl.div_rn(q, qd)
        k = tl.div_rn(k, kd)
    q = tl.fma(q, SCALE, -0.0)
    d = _exp_dekker(g)
    d = tl.where(g != g, g, d)
    tl.store(QP + row * KD + kk, q, kk < KD)
    tl.store(KP + row * KD + kk, k, kk < KD)
    tl.store(D + row * KD + kk, d, kk < KD)


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
    valid = (vv[None, :] < VD) & (kk[:, None] < KD)
    state = tl.load(
        S0 + (bh * VD + vv[None, :]) * KD + kk[:, None], valid, 0
    ).to(tl.float32)
    for ti in range(T):
        token = (bi * T + ti) * H + hi
        q = tl.load(Q + token * KD + kk, kk < KD, 0).to(tl.float32)
        q = tl.fma(q, SCALE, -0.0)
        k = tl.load(K + token * KD + kk, kk < KD, 0).to(tl.float32)
        d = tl.load(D + token * KD + kk, kk < KD, 0)
        state = tl.fma(state, d[:, None], -0.0)
        pred = _sum_rows(tl.fma(state, k[:, None], -0.0), RV, BK)
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
            kk[:, None] < KD,
            state + tl.fma(delta[None, :], k[:, None], -0.0),
            0.0,
        )
        out = _sum_rows(tl.fma(state, q[:, None], -0.0), RV, BK)
        tl.store(O + token * VD + vv, out, vv < VD)
    tl.store(S1 + (bh * VD + vv[None, :]) * KD + kk[:, None], state, valid)


@triton.jit
def _prepare_decay(G, D, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(
        tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK
    ):
        i = start + tl.arange(0, BLOCK)
        x = tl.load(G + i, i < N, 0).to(tl.float32)
        y = _exp_dekker(x)
        y = tl.where(x != x, x, y)
        tl.store(D + i, y, i < N)


def fused_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel
):
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    bk = triton.next_power_of_2(kd)
    decay = torch.empty(g.shape, dtype=torch.float32, device=g.device)
    if use_qk_l2norm_in_kernel:
        qprep = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        kprep = torch.empty_like(qprep)
        if b * t * h:
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
                True,
                bk,
                num_warps=4,
                enable_fp_fusion=False,
            )
        rec_scale = 1.0
    else:
        qprep = q.contiguous()
        kprep = k.contiguous()
        if g.numel():
            _prepare_decay[(min(triton.cdiv(g.numel(), 2048), 40),)](
                g.contiguous(), decay, g.numel(), 2048, enable_fp_fusion=False
            )
        rec_scale = scale
    output = torch.empty((b, t, h, vd), dtype=q.dtype, device=q.device)
    final_state = torch.empty(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    if b * h * vd:
        _recurrent[(b * h * triton.cdiv(vd, 64),)](
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
            rec_scale,
            beta.ndim == v.ndim,
            64,
            bk,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, final_state


__all__ = ["fused_recurrent_kda"]
