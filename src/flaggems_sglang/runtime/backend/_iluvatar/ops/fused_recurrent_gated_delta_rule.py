# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of the fused recurrent gated delta-rule (GDN) kernel.

Single-launch recurrent scan: each program owns one fp32 state tile for a fixed
(batch, value-head, V-block) triple and walks the T timesteps sequentially,
mirroring the float32 reference recurrence exactly.

Two specializations, selected by BLOCK_K:

* BLOCK_K <= 64: the state is held as [BLOCK_K, BLOCK_V] with the value axis on
  the lane axis so both per-step reductions run along the register axis with
  tl.sum(..., axis=0). The [V, K] contiguous state is transposed in registers
  once on entry and once on exit.
* BLOCK_K > 64 (SPLIT): the K axis is represented as two explicit 64-wide
  register chunks, each held as [BLOCK_KH, BLOCK_V] with axis=0 reductions.
  Chunking keeps the layout stable across the scan; a single 128-wide axis-0
  tile makes the compiler re-transpose the loop-carried state every step.

When `use_qk_l2norm_in_kernel` is set the per-step normalization adds two more
cross-lane reductions (the sums of squares of k and of q) to a loop that is
already reduction-issue bound. For the long / chunked scans a small separate
kernel precomputes those norms once per (batch, step, key head) and the scan
loads them as scalars.
"""

import torch
import triton
import triton.language as tl

_NORM_BT = 32


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    h0_ptr,
    o_ptr,
    ht_ptr,
    nk_ptr,
    nq_ptr,
    scale,
    T,
    HV,
    ratio,
    qsb,
    qst,
    qsh,
    ksb,
    kst,
    ksh,
    vsb,
    vst,
    vsh,
    gsb,
    gst,
    gsh,
    bsb,
    bst,
    bsh,
    h0sb,
    h0sh,
    h0sv,
    htsb,
    htsh,
    htsv,
    osb,
    ost,
    osh,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    SPLIT: tl.constexpr,
    L2NORM: tl.constexpr,
    PRE_NORM: tl.constexpr,
    HAS_INIT: tl.constexpr,
    BETA_PER_V: tl.constexpr,
    OUT_FINAL: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_h // HV
    hv = pid_h % HV
    hk = hv // ratio
    H = HV // ratio

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    q_base = qsb * b + qsh * hk
    k_base = ksb * b + ksh * hk
    v_base = vsb * b + vsh * hv
    g_base = gsb * b + gsh * hv
    b_base = bsb * b + bsh * hv
    o_base = osb * b + osh * hv
    n_base = (b * T) * H + hk

    if SPLIT:
        offs_k0 = tl.arange(0, BLOCK_KH)
        offs_k1 = BLOCK_KH + tl.arange(0, BLOCK_KH)
        mk0 = offs_k0 < K
        mk1 = offs_k1 < K
        m0 = mask_v[:, None] & mk0[None, :]
        m1 = mask_v[:, None] & mk1[None, :]
        st0 = h0sb * b + h0sh * hv + h0sv * offs_v[:, None] + offs_k0[None, :]
        st1 = h0sb * b + h0sh * hv + h0sv * offs_v[:, None] + offs_k1[None, :]
        ft0 = htsb * b + htsh * hv + htsv * offs_v[:, None] + offs_k0[None, :]
        ft1 = htsb * b + htsh * hv + htsv * offs_v[:, None] + offs_k1[None, :]
        if HAS_INIT:
            b_s0 = tl.trans(
                tl.load(h0_ptr + st0, mask=m0, other=0.0).to(tl.float32)
            )
            b_s1 = tl.trans(
                tl.load(h0_ptr + st1, mask=m1, other=0.0).to(tl.float32)
            )
        else:
            b_s0 = tl.zeros([BLOCK_KH, BLOCK_V], dtype=tl.float32)
            b_s1 = tl.zeros([BLOCK_KH, BLOCK_V], dtype=tl.float32)
    else:
        offs_k = tl.arange(0, BLOCK_K)
        mk = offs_k < K
        msk = mask_v[:, None] & mk[None, :]
        st = h0sb * b + h0sh * hv + h0sv * offs_v[:, None] + offs_k[None, :]
        ft = htsb * b + htsh * hv + htsv * offs_v[:, None] + offs_k[None, :]
        if HAS_INIT:
            b_state = tl.trans(
                tl.load(h0_ptr + st, mask=msk, other=0.0).to(tl.float32)
            )
        else:
            b_state = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in range(0, T):
        b_v = tl.load(
            v_ptr + v_base + vst * t + offs_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        b_g = tl.load(g_ptr + g_base + gst * t)
        if BETA_PER_V:
            b_beta = tl.load(
                beta_ptr + b_base + bst * t + offs_v, mask=mask_v, other=0.0
            )
        else:
            b_beta = tl.load(beta_ptr + b_base + bst * t)

        if SPLIT:
            b_q0 = tl.load(
                q_ptr + q_base + qst * t + offs_k0, mask=mk0, other=0.0
            ).to(tl.float32)
            b_q1 = tl.load(
                q_ptr + q_base + qst * t + offs_k1, mask=mk1, other=0.0
            ).to(tl.float32)
            b_k0 = tl.load(
                k_ptr + k_base + kst * t + offs_k0, mask=mk0, other=0.0
            ).to(tl.float32)
            b_k1 = tl.load(
                k_ptr + k_base + kst * t + offs_k1, mask=mk1, other=0.0
            ).to(tl.float32)
            if L2NORM:
                if PRE_NORM:
                    nk = tl.load(nk_ptr + n_base + t * H)
                    nq = tl.load(nq_ptr + n_base + t * H)
                else:
                    nk = tl.sqrt(
                        tl.sum(b_k0 * b_k0, axis=0)
                        + tl.sum(b_k1 * b_k1, axis=0)
                        + 1e-6
                    )
                    nq = tl.sqrt(
                        tl.sum(b_q0 * b_q0, axis=0)
                        + tl.sum(b_q1 * b_q1, axis=0)
                        + 1e-6
                    )
                b_k0 = b_k0 / nk
                b_k1 = b_k1 / nk
                b_q0 = b_q0 * (scale / nq)
                b_q1 = b_q1 * (scale / nq)
            else:
                b_q0 = b_q0 * scale
                b_q1 = b_q1 * scale

            b_s0 = b_s0 * tl.exp(b_g)
            b_s1 = b_s1 * tl.exp(b_g)
            b_pred = tl.sum(b_s0 * b_k0[:, None], axis=0) + tl.sum(
                b_s1 * b_k1[:, None], axis=0
            )
            b_c = (b_v - b_pred) * b_beta
            b_s0 = b_s0 + b_k0[:, None] * b_c[None, :]
            b_s1 = b_s1 + b_k1[:, None] * b_c[None, :]
            b_o = tl.sum(b_s0 * b_q0[:, None], axis=0) + tl.sum(
                b_s1 * b_q1[:, None], axis=0
            )
        else:
            b_q = tl.load(
                q_ptr + q_base + qst * t + offs_k, mask=mk, other=0.0
            ).to(tl.float32)
            b_k = tl.load(
                k_ptr + k_base + kst * t + offs_k, mask=mk, other=0.0
            ).to(tl.float32)
            if L2NORM:
                if PRE_NORM:
                    nk = tl.load(nk_ptr + n_base + t * H)
                    nq = tl.load(nq_ptr + n_base + t * H)
                else:
                    nk = tl.sqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
                    nq = tl.sqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
                b_k = b_k / nk
                b_q = b_q * (scale / nq)
            else:
                b_q = b_q * scale

            b_state = b_state * tl.exp(b_g)
            b_pred = tl.sum(b_state * b_k[:, None], axis=0)
            b_c = (b_v - b_pred) * b_beta
            b_state = b_state + b_k[:, None] * b_c[None, :]
            b_o = tl.sum(b_state * b_q[:, None], axis=0)

        tl.store(
            o_ptr + o_base + ost * t + offs_v,
            b_o.to(o_ptr.dtype.element_ty),
            mask=mask_v,
        )

    if OUT_FINAL:
        if SPLIT:
            tl.store(ht_ptr + ft0, tl.trans(b_s0), mask=m0)
            tl.store(ht_ptr + ft1, tl.trans(b_s1), mask=m1)
        else:
            tl.store(ht_ptr + ft, tl.trans(b_state), mask=msk)


@triton.jit
def _qk_norms_kernel(
    q_ptr,
    k_ptr,
    nk_ptr,
    nq_ptr,
    T,
    H,
    qsb,
    qst,
    qsh,
    ksb,
    kst,
    ksh,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BT: tl.constexpr,
):
    """sqrt(sum(x^2) + 1e-6) for q and k, one program per (batch, key head, T-block).

    Writes into a [B, T, H] buffer so the scan can load each value as a scalar.
    """
    pid = tl.program_id(0)
    ntb = tl.cdiv(T, BT)
    b = pid // (ntb * H)
    rem = pid % (ntb * H)
    tb = rem // H
    hk = rem % H
    t0 = tb * BT
    ot = t0 + tl.arange(0, BT)
    ok = tl.arange(0, BLOCK_K)
    mt = ot < T
    mk = ok < K
    m = mt[:, None] & mk[None, :]
    kk = tl.load(
        k_ptr + ksb * b + kst * ot[:, None] + ksh * hk + ok[None, :],
        mask=m,
        other=0.0,
    ).to(tl.float32)
    qq = tl.load(
        q_ptr + qsb * b + qst * ot[:, None] + qsh * hk + ok[None, :],
        mask=m,
        other=0.0,
    ).to(tl.float32)
    snk = tl.sqrt(tl.sum(kk * kk, axis=1) + 1e-6)
    snq = tl.sqrt(tl.sum(qq * qq, axis=1) + 1e-6)
    dst = (b * T + ot) * H + hk
    tl.store(nk_ptr + dst, snk, mask=mt)
    tl.store(nq_ptr + dst, snq, mask=mt)


def fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    ratio = HV // H

    beta_per_v = beta.dim() == v.dim()
    l2norm = bool(use_qk_l2norm_in_kernel)
    out_final = bool(output_final_state)
    has_init = initial_state is not None

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    if out_final:
        final_state = torch.empty(
            (B, HV, V, K), dtype=torch.float32, device=v.device
        )
    else:
        final_state = None

    BLOCK_K = triton.next_power_of_2(K)
    n_base = B * HV
    split = BLOCK_K > 64
    if T == 1:
        # Pure state read + matvec + state write: no loop-carried dependency, so
        # throughput is governed by the number of resident programs. Target
        # probes show BLOCK_V=16 with 8 warps is best or tied at every decode
        # size (B*HV from 32 to 2048).
        BLOCK_V = min(triton.next_power_of_2(V), 16)
        num_warps = 8
    else:
        # Sequential scan: pick the per-program state tile and warp count that
        # keep the whole machine busy across the T steps. Measured on the
        # target, the narrow-reduction K<=64 shape wants many small 1-warp
        # programs, while the K-chunked K>64 shape wants wide 2-warp programs.
        bv_max = 32 if not split else 128
        prog_floor = 256 if not split else 128
        BLOCK_V = min(triton.next_power_of_2(V), bv_max)
        while BLOCK_V > 8 and n_base * triton.cdiv(V, BLOCK_V) < prog_floor:
            BLOCK_V //= 2
        num_warps = 1 if not split else 2
    BLOCK_KH = 64 if split else BLOCK_K
    grid = (triton.cdiv(V, BLOCK_V), B * HV)

    # Hoisting the q/k norms into a separate cheap kernel pays off on the long
    # and/or chunked scans; the short narrow scans recover less per step than
    # the extra pass costs.
    pre_norm = l2norm and (split or T >= 64)

    qs = q.stride()
    ks = k.stride()
    vs = v.stride()
    gs = g.stride()
    bs = beta.stride()
    os_ = o.stride()

    if has_init:
        h0s = initial_state.stride()
    else:
        h0s = (0, 0, 0, 0)
    ht = final_state if out_final else o
    hts = ht.stride()

    if pre_norm:
        nk = torch.empty((B * T * H,), dtype=torch.float32, device=v.device)
        nq = torch.empty((B * T * H,), dtype=torch.float32, device=v.device)
        _qk_norms_kernel[(B * triton.cdiv(T, _NORM_BT) * H,)](
            q,
            k,
            nk,
            nq,
            T,
            H,
            qs[0],
            qs[1],
            qs[2],
            ks[0],
            ks[1],
            ks[2],
            K=K,
            BLOCK_K=BLOCK_K,
            BT=_NORM_BT,
            num_warps=4,
        )
    else:
        nk = o
        nq = o

    _gdn_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        q if not has_init else initial_state,
        o,
        ht,
        nk,
        nq,
        scale,
        T,
        HV,
        ratio,
        qs[0],
        qs[1],
        qs[2],
        ks[0],
        ks[1],
        ks[2],
        vs[0],
        vs[1],
        vs[2],
        gs[0],
        gs[1],
        gs[2],
        bs[0],
        bs[1],
        bs[2],
        h0s[0],
        h0s[1],
        h0s[2],
        hts[0],
        hts[1],
        hts[2],
        os_[0],
        os_[1],
        os_[2],
        K=K,
        V=V,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        BLOCK_KH=BLOCK_KH,
        SPLIT=split,
        L2NORM=l2norm,
        PRE_NORM=pre_norm,
        HAS_INIT=has_init,
        BETA_PER_V=beta_per_v,
        OUT_FINAL=out_final,
        num_warps=num_warps,
        num_stages=2,
    )

    return o, final_state


__all__ = ["fused_recurrent_gated_delta_rule"]
