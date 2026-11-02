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

"""Fused recurrent gated delta-rule (fla fused_recurrent_gdn / FlagGems-sglang).

Self-contained Triton implementation of the reference recurrence in
`flaggems_reference/fused_recurrent_gdn.py`.

Per timestep t (all arithmetic in float32):
    x  = q[b,t,hk] (optionally L2-normalized: x / sqrt(sum(x^2) + 1e-6))
    kk = k[b,t,hk] (optionally L2-normalized the same way)
    x  = x * scale
    d  = exp(g[b,t,hv])                      # negative log-decay, contractive
    S  = d * S                               # decay BEFORE the prediction
    p  = S @ kk                              # prediction of v
    u  = beta * (v - p)                      # per-head scalar or per-channel beta
    S  = S + outer(u, kk)                    # rank-1 update
    o[b,t,hv] = S @ x                        # readout with the POST-update state

The [BV, BK] float32 state tile lives in registers for the whole T loop; it is
loaded once from initial_state and stored once as final_state.  The K axis is
never split across programs (both reductions and the update are intra-program);
only the V axis is split, which needs no atomics and no cross-program sync.

Two per-step details matter for speed on this target:

* The two K-reductions consume k and q broadcast along V.  Loading them as
  [BV, BK] tiles whose rows all read the same addresses (every row reads the
  head's row of the timestep) removes the cross-thread layout conversion that a
  [BK] -> [BV, BK] broadcast otherwise costs, and lets the L2 normalization be
  a single axis-1 reduction of the tile.  The values are bit-identical to the
  [BK] formulation.  Without L2 normalization the [BK] load plus broadcast is
  kept, because there the tile would be pure overhead.

* The normalization divides by one scalar per vector, so a single hardware
  rsqrt multiplied into the vector replaces the per-element fp32 division
  sequence (~5 instructions per element) at a cost of ~1 ulp.

Launch configuration: the per-step dependency chain, not bandwidth, limits the
multi-step shapes, so the widest tile that still fits is used (BV=128 down to
BV=32 as the grid shrinks); the T<=2 shapes have no chain to shorten and take
the widest available program count instead.

Contiguous inputs use strides derived from the shapes inside the kernel; a
non-contiguous fallback passes a small int64 stride vector.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    s_ptr,
    o_ptr,
    ht_ptr,
    sp_ptr,
    T,
    H,
    HV,
    K,
    V,
    ratio,
    scale,
    BV: tl.constexpr,
    BK: tl.constexpr,
    L2NORM: tl.constexpr,
    BETA_HEADWISE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    STRIDED: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HV
    hv = pid_bh % HV
    hk = hv // ratio

    if STRIDED:
        qb = tl.load(sp_ptr + 0)
        qt = tl.load(sp_ptr + 1)
        qh = tl.load(sp_ptr + 2)
        kb = tl.load(sp_ptr + 3)
        kt = tl.load(sp_ptr + 4)
        kh = tl.load(sp_ptr + 5)
        vb = tl.load(sp_ptr + 6)
        vt = tl.load(sp_ptr + 7)
        vh = tl.load(sp_ptr + 8)
        gb = tl.load(sp_ptr + 9)
        gt = tl.load(sp_ptr + 10)
        gh = tl.load(sp_ptr + 11)
        bb = tl.load(sp_ptr + 12)
        bt = tl.load(sp_ptr + 13)
        bh = tl.load(sp_ptr + 14)
        bv = tl.load(sp_ptr + 15)
        sb = tl.load(sp_ptr + 16)
        sh = tl.load(sp_ptr + 17)
        sv = tl.load(sp_ptr + 18)
        sk = tl.load(sp_ptr + 19)
    else:
        qb = T * H * K
        qt = H * K
        qh = K
        kb = qb
        kt = qt
        kh = K
        vb = T * HV * V
        vt = HV * V
        vh = V
        gb = T * HV
        gt = HV
        gh = 1
        if BETA_HEADWISE:
            bb = T * HV * V
            bt = HV * V
            bh = V
            bv = 1
        else:
            bb = T * HV
            bt = HV
            bh = 1
            bv = 0
        sb = HV * V * K
        sh = V * K
        sv = K
        sk = 1

    # final_state is always a freshly allocated contiguous [B, HV, V, K] fp32
    htb = HV * V * K
    hth = V * K

    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    vmask = offs_v < V
    kmask = offs_k < K
    # The same K indices replicated down every row of the [BV, BK] tile.
    offs_k2 = tl.zeros([BV, 1], tl.int32) + offs_k[None, :]
    kmask2 = offs_k2 < K

    if HAS_INIT:
        s_ptrs = (
            s_ptr
            + b * sb
            + hv * sh
            + offs_v[:, None] * sv
            + offs_k[None, :] * sk
        )
        S = tl.load(s_ptrs, mask=vmask[:, None] & kmask[None, :], other=0.0)
    else:
        S = tl.zeros([BV, BK], dtype=tl.float32)

    q_base = q_ptr + b * qb + hk * qh
    k_base = k_ptr + b * kb + hk * kh
    v_base = v_ptr + b * vb + hv * vh
    g_base = g_ptr + b * gb + hv * gh
    beta_base = beta_ptr + b * bb + hv * bh
    o_base = o_ptr + b * (T * HV * V) + hv * V

    for t in range(0, T):
        if L2NORM:
            k2 = tl.load(k_base + t * kt + offs_k2, mask=kmask2, other=0.0).to(
                tl.float32
            )
            q2 = tl.load(q_base + t * qt + offs_k2, mask=kmask2, other=0.0).to(
                tl.float32
            )
            kn2 = tl.sum(k2 * k2, axis=1)
            qn2 = tl.sum(q2 * q2, axis=1)
            k_t = k2 * tl.rsqrt(kn2[:, None] + 1e-6)
            q_t = q2 * (tl.rsqrt(qn2[:, None] + 1e-6) * scale)
        else:
            k1 = tl.load(k_base + t * kt + offs_k, mask=kmask, other=0.0).to(
                tl.float32
            )
            q1 = tl.load(q_base + t * qt + offs_k, mask=kmask, other=0.0).to(
                tl.float32
            )
            k_t = tl.broadcast_to(k1[None, :], [BV, BK])
            q_t = tl.broadcast_to((q1 * scale)[None, :], [BV, BK])

        d = tl.exp(tl.load(g_base + t * gt))

        v_t = tl.load(v_base + t * vt + offs_v, mask=vmask, other=0.0).to(
            tl.float32
        )
        if BETA_HEADWISE:
            bt_v = tl.load(
                beta_base + t * bt + offs_v * bv, mask=vmask, other=0.0
            )
        else:
            bt_v = tl.load(beta_base + t * bt)

        S = S * d
        pred = tl.sum(S * k_t, axis=1)
        u = (v_t - pred) * bt_v
        S = S + u[:, None] * k_t
        o_t = tl.sum(S * q_t, axis=1)
        tl.store(
            o_base + t * (HV * V) + offs_v,
            o_t.to(o_ptr.dtype.element_ty),
            mask=vmask,
        )

    if HAS_FINAL:
        ht_ptrs = (
            ht_ptr + b * htb + hv * hth + offs_v[:, None] * K + offs_k[None, :]
        )
        tl.store(ht_ptrs, S, mask=vmask[:, None] & kmask[None, :])


def _strided_vec(q, k, v, g, beta, s, beta_headwise):
    vals = [
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
    ]
    if beta_headwise:
        vals += [
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            beta.stride(3),
        ]
    else:
        vals += [beta.stride(0), beta.stride(1), beta.stride(2), 0]
    if s is None:
        vals += [0, 0, 0, 0]
    else:
        vals += [s.stride(0), s.stride(1), s.stride(2), s.stride(3)]
    return torch.tensor(vals, dtype=torch.int64, device=q.device)


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
    beta_headwise = beta.dim() == v.dim()

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)

    has_init = initial_state is not None
    has_final = bool(output_final_state)

    if has_init:
        s = initial_state
    else:
        s = v  # dummy pointer, never dereferenced when HAS_INIT is false

    if has_final:
        ht = torch.empty((B, HV, V, K), dtype=torch.float32, device=v.device)
    else:
        ht = v  # dummy pointer, never dereferenced when HAS_FINAL is false

    contig = (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
        and beta.is_contiguous()
        and (not has_init or initial_state.is_contiguous())
    )
    if contig:
        sp = q
    else:
        sp = _strided_vec(
            q,
            k,
            v,
            g,
            beta,
            initial_state if has_init else None,
            beta_headwise,
        )

    BK = triton.next_power_of_2(K)
    # Wide V tiles amortize the fixed per-step reduction/update overhead over
    # more elements, and the state tile stays in registers with no spills up to
    # BV=BK=128 (131 registers).  The T<=2 shapes have no dependency chain to
    # shorten, so they take the smallest tile instead (more programs).
    BV = 64 if V >= 64 else min(32, triton.next_power_of_2(V))
    if V >= 128 and triton.cdiv(V, BV) * B * HV < 128:
        BV = 32
    if T > 2:
        NVW = triton.cdiv(V, BV)
        while BV < 128 and BV * 2 <= V and NVW * B * HV > 64:
            BV = BV * 2
            NVW = triton.cdiv(V, BV)
    grid = (triton.cdiv(V, BV), B * HV)

    tile = BV * BK
    if tile >= 1024:
        num_warps = 4
    elif tile >= 256:
        num_warps = 2
    else:
        num_warps = 1

    _gdn_fused_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        s,
        o,
        ht,
        sp,
        T,
        H,
        HV,
        K,
        V,
        ratio,
        float(scale),
        BV=BV,
        BK=BK,
        L2NORM=bool(use_qk_l2norm_in_kernel),
        BETA_HEADWISE=beta_headwise,
        HAS_INIT=has_init,
        HAS_FINAL=has_final,
        STRIDED=not contig,
        num_warps=num_warps,
    )

    return o, (ht if has_final else None)


__all__ = ["fused_recurrent_gated_delta_rule"]
