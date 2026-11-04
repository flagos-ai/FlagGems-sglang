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

"""Gated delta-rule linear attention recurrence (fused recurrent GDN), Triton.

One program per (batch, value-head, V-block). The fp32 state tile
[BLOCK_V, BLOCK_K] lives in registers for the whole sequential T walk, so the
state is touched in global memory exactly twice: the prologue load and the
epilogue store. The per-step loads of k, q, v, g and beta are issued one
timestep ahead so their memory latency overlaps the previous step's reduction
chain, which is the dominant stall on the serial T recurrence.
"""

import torch
import triton
import triton.language as tl


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
    T,
    H,
    HV,
    K,
    V,
    scale,
    RATIO,
    sqb,
    sqt,
    sqh,
    sqk,
    skb,
    skt,
    skh,
    skk,
    svb,
    svt,
    svh,
    svv,
    sgb,
    sgt,
    sgh,
    sbb,
    sbt,
    sbh,
    sbv,
    sob,
    sot,
    soh,
    sov,
    ihb,
    ihh,
    ihv,
    ihk,
    thb,
    thh,
    thv,
    thk,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_INIT: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    BETA_CHANNEL: tl.constexpr,
    MASK_K: tl.constexpr,
    MASK_V: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)
    hv = pid_bh % HV
    b = pid_bh // HV
    h = hv // RATIO

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    if HAS_INIT:
        if MASK_K or MASK_V:
            st_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
            state = tl.load(
                h0_ptr
                + b * ihb
                + hv * ihh
                + offs_v[:, None] * ihv
                + offs_k[None, :] * ihk,
                mask=st_mask,
                other=0.0,
            )
        else:
            state = tl.load(
                h0_ptr
                + b * ihb
                + hv * ihh
                + offs_v[:, None] * ihv
                + offs_k[None, :] * ihk,
            )
    else:
        state = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    k_base = k_ptr + b * skb + h * skh
    q_base = q_ptr + b * sqb + h * sqh
    v_base = v_ptr + b * svb + hv * svh
    o_base = o_ptr + b * sob + hv * soh
    g_base = g_ptr + b * sgb + hv * sgh
    b_base = beta_ptr + b * sbb + hv * sbh

    if MASK_K:
        kmask = offs_k < K
    if MASK_V:
        vmask = offs_v < V

    # Prime the one-step-ahead load registers with timestep 0.
    cur = 0
    if MASK_K:
        k_t = tl.load(k_base + offs_k * skk, mask=kmask, other=0.0)
        q_t = tl.load(q_base + offs_k * sqk, mask=kmask, other=0.0)
    else:
        k_t = tl.load(k_base + offs_k * skk)
        q_t = tl.load(q_base + offs_k * sqk)
    if MASK_V:
        v_t = tl.load(v_base + offs_v * svv, mask=vmask, other=0.0)
    else:
        v_t = tl.load(v_base + offs_v * svv)
    gval = tl.load(g_base)
    if BETA_CHANNEL:
        if MASK_V:
            bt = tl.load(b_base + offs_v * sbv, mask=vmask, other=0.0).to(
                tl.float32
            )
        else:
            bt = tl.load(b_base + offs_v * sbv).to(tl.float32)
    else:
        bt = tl.load(b_base).to(tl.float32)

    for t in range(0, T):
        # Issue the next timestep's loads before consuming the current ones.
        # `cur` is clamped so the final prefetch stays in bounds.
        cur = tl.minimum(t + 1, T - 1)
        if MASK_K:
            k_n = tl.load(
                k_base + cur * skt + offs_k * skk, mask=kmask, other=0.0
            )
            q_n = tl.load(
                q_base + cur * sqt + offs_k * sqk, mask=kmask, other=0.0
            )
        else:
            k_n = tl.load(k_base + cur * skt + offs_k * skk)
            q_n = tl.load(q_base + cur * sqt + offs_k * sqk)
        if MASK_V:
            v_n = tl.load(
                v_base + cur * svt + offs_v * svv, mask=vmask, other=0.0
            )
        else:
            v_n = tl.load(v_base + cur * svt + offs_v * svv)
        g_n = tl.load(g_base + cur * sgt)
        if BETA_CHANNEL:
            if MASK_V:
                bt_n = tl.load(
                    b_base + cur * sbt + offs_v * sbv, mask=vmask, other=0.0
                ).to(tl.float32)
            else:
                bt_n = tl.load(b_base + cur * sbt + offs_v * sbv).to(
                    tl.float32
                )
        else:
            bt_n = tl.load(b_base + cur * sbt).to(tl.float32)

        kf = k_t.to(tl.float32)
        qf = q_t.to(tl.float32)
        if USE_L2NORM:
            kf = kf / tl.sqrt(tl.sum(kf * kf, axis=0) + 1e-6)
            qf = qf / tl.sqrt(tl.sum(qf * qf, axis=0) + 1e-6)
        qs = qf * scale

        d = tl.exp(gval)

        pred = d * tl.sum(state * kf[None, :], axis=1)
        vf = v_t.to(tl.float32)
        vcorr = (vf - pred) * bt

        state = state * d + vcorr[:, None] * kf[None, :]

        o_t = tl.sum(state * qs[None, :], axis=1)
        if MASK_V:
            tl.store(
                o_base + t * sot + offs_v * sov,
                o_t.to(o_ptr.dtype.element_ty),
                mask=vmask,
            )
        else:
            tl.store(
                o_base + t * sot + offs_v * sov, o_t.to(o_ptr.dtype.element_ty)
            )

        k_t = k_n
        q_t = q_n
        v_t = v_n
        gval = g_n
        bt = bt_n

    if OUT_FINAL:
        if MASK_K or MASK_V:
            fin_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
            tl.store(
                ht_ptr
                + b * thb
                + hv * thh
                + offs_v[:, None] * thv
                + offs_k[None, :] * thk,
                state,
                mask=fin_mask,
            )
        else:
            tl.store(
                ht_ptr
                + b * thb
                + hv * thh
                + offs_v[:, None] * thv
                + offs_k[None, :] * thk,
                state,
            )


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
    beta_channel = beta.dim() == v.dim()

    BLOCK_K = max(16, triton.next_power_of_2(K))
    # The register-resident fp32 state tile [BLOCK_V, BLOCK_K] drives the whole
    # cost model. An interleaved A/B over all ten timing shapes (five repeats
    # each, min-vs-median spread under 1% for the retained configs) showed a
    # narrow tile is better everywhere than the wide one this kernel shipped
    # with: 32x128 at 4 warps beat 64x128 at 8 warps by 4-23% on the K=V=128
    # shapes, and 16x64 at 2 warps beat 64x64 at 4 warps by 27% on bench0 and 6%
    # on bench1. Two warps carry a 2048-element tile (32 fp32 per thread) and
    # four warps carry 4096 without spilling, so the tile is sized to that budget
    # and the extra V-blocks add independent programs instead of only widening
    # each program.
    BLOCK_V = max(
        16, min(16 if BLOCK_K <= 64 else 32, triton.next_power_of_2(V))
    )
    num_warps = 2 if BLOCK_V * BLOCK_K <= 2048 else 4

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    if output_final_state:
        final_state = torch.empty(
            (B, HV, V, K), dtype=torch.float32, device=v.device
        )
        thb, thh, thv, thk = final_state.stride()
        ht = final_state
    else:
        final_state = None
        thb = thh = thv = thk = 0
        ht = o

    if initial_state is not None:
        h0 = initial_state
        ihb, ihh, ihv, ihk = h0.stride()
        has_init = True
    else:
        h0 = q
        ihb = ihh = ihv = ihk = 0
        has_init = False

    sqb, sqt, sqh, sqk = q.stride()
    skb, skt, skh, skk = k.stride()
    svb, svt, svh, svv = v.stride()
    sgb, sgt, sgh = g.stride()
    sbb, sbt, sbh = beta.stride()[0], beta.stride()[1], beta.stride()[2]
    sbv = beta.stride()[3] if beta_channel else 0
    sob, sot, soh, sov = o.stride()

    grid = (triton.cdiv(V, BLOCK_V), B * HV)

    _gdn_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        h0,
        o,
        ht,
        T,
        H,
        HV,
        K,
        V,
        float(scale),
        ratio,
        sqb,
        sqt,
        sqh,
        sqk,
        skb,
        skt,
        skh,
        skk,
        svb,
        svt,
        svh,
        svv,
        sgb,
        sgt,
        sgh,
        sbb,
        sbt,
        sbh,
        sbv,
        sob,
        sot,
        soh,
        sov,
        ihb,
        ihh,
        ihv,
        ihk,
        thb,
        thh,
        thv,
        thk,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        HAS_INIT=has_init,
        OUT_FINAL=bool(output_final_state),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        BETA_CHANNEL=beta_channel,
        MASK_K=(K % BLOCK_K != 0),
        MASK_V=(V % BLOCK_V != 0),
        num_warps=num_warps,
    )

    return o, final_state


__all__ = ["fused_recurrent_gated_delta_rule"]
