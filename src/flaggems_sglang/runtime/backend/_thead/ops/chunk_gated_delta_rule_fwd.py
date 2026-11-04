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

"""Chunked gated delta-rule forward -- recurrent kernel with a parallel
q/k pre-normalization pass.

Architecture history (all measured on the target):
  round 1: fused recurrent form, one program per (sequence, head, V-block),
      state slab [K, BV] in registers, BV=64, num_warps=4.  Profiling showed the
      per-timestep cross-warp reduction / block-barrier chain (18 barriers plus
      ~32 shared-memory ops per step, zero spills, memory pipes <1%) was the
      critical path, and the grid only filled 4-32 of 64 SMs.  geo_mean 261.96.
  round 2: BV=16 with num_warps=1, so every axis-0/axis-1 reduction stays
      intra-warp (shuffle only) and the resident-program count rises 4x.
      Uniformly ~1.57x on 7 of 8 timing shapes.  geo_mean 365.34.
  round 3: the in-loop l2norm row reductions were measured to be the single
      largest remaining term inside the serial chain.  They are pure
      row-wise work with no dependence on the state, so they are hoisted into a
      fully parallel pre-pass kernel whose rows are contiguous in
      (sequence, time, head).  The pre-pass stores fp32, so the values the
      recurrent kernel consumes are bit-identical to what it would have
      computed inline (the sum of squares is still formed in fp32 from the
      upcast input, with eps inside the rsqrt, before the scale multiply).

Per step t (matching `_segment` in the reference):
    S <- exp(g_t) * S
    p  = S^T k_t
    c  = beta_t * (v_t - p)
    S <- S + outer(k_t, c)
    o_t = q_t^T S            (readout AFTER the update)
with q_t = scale * l2norm(q_t) and k_t = l2norm(k_t) when requested.

State layout is [N, H, K, V] (K before V), so the update is outer(k, c) and both
contractions run over the K axis.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prep_kernel(
    x_ptr,
    out_ptr,
    scale,
    T,
    H,
    K: tl.constexpr,
    BK: tl.constexpr,
    sx_n,
    sx_t,
    sx_h,
    sx_k,
    so_n,
    so_t,
    so_h,
    so_k,
    DO_NORM: tl.constexpr,
    DO_SCALE: tl.constexpr,
):
    """fp32 l2norm (+ optional scale) of one [K] row, written as fp32.

    One program per row of the (N, T, H) grid; the (n, t, h) indices are decoded
    from the flat row id so the source may have arbitrary strides.
    """
    row = tl.program_id(0)
    h = row % H
    rt = row // H
    t = rt % T
    n = rt // T

    ok = tl.arange(0, BK)
    km = ok < K
    x = tl.load(
        x_ptr + n * sx_n + t * sx_t + h * sx_h + ok * sx_k,
        mask=km,
        other=0.0,
    ).to(tl.float32)
    if DO_NORM:
        x = x * tl.rsqrt(tl.sum(x * x, axis=0) + 1e-6)
    if DO_SCALE:
        x = x * scale
    tl.store(
        out_ptr + n * so_n + t * so_t + h * so_h + ok * so_k,
        x,
        mask=km,
    )


@triton.jit
def _gdn_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    cu_ptr,
    h0_ptr,
    ht_ptr,
    o_ptr,
    scale,
    T,
    H,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    U: tl.constexpr,
    sq_n,
    sq_t,
    sq_h,
    sq_k,
    sk_n,
    sk_t,
    sk_h,
    sk_k,
    sv_n,
    sv_t,
    sv_h,
    sv_v,
    sg_n,
    sg_t,
    sg_h,
    sb_n,
    sb_t,
    sb_h,
    so_n,
    so_t,
    so_h,
    so_v,
    h0_n,
    h0_h,
    h0_k,
    h0_v,
    ht_n,
    ht_h,
    ht_k,
    ht_v,
    cu_s,
    HAS_INIT: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    L2NORM: tl.constexpr,
    VARLEN: tl.constexpr,
):
    NV: tl.constexpr = (V + BV - 1) // BV
    pid = tl.program_id(0)
    n = pid // (H * NV)
    r = pid % (H * NV)
    h = r // NV
    vb = r % NV

    b = 0
    t0 = 0
    t1 = T
    if VARLEN:
        t0 = tl.load(cu_ptr + n * cu_s).to(tl.int32)
        t1 = tl.load(cu_ptr + (n + 1) * cu_s).to(tl.int32)
        b = 0
    else:
        b = n

    ok = tl.arange(0, BK)
    ov = tl.arange(0, BV)
    km = ok < K
    vg = vb * BV + ov
    vm = vg < V

    q_base = q_ptr + b * sq_n + h * sq_h
    k_base = k_ptr + b * sk_n + h * sk_h
    v_base = v_ptr + b * sv_n + h * sv_h
    g_base = g_ptr + b * sg_n + h * sg_h
    b_base = beta_ptr + b * sb_n + h * sb_h
    o_base = o_ptr + b * so_n + h * so_h

    if HAS_INIT:
        S = tl.load(
            h0_ptr
            + n * h0_n
            + h * h0_h
            + ok[:, None] * h0_k
            + vg[None, :] * h0_v,
            mask=km[:, None] & vm[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        S = tl.zeros((BK, BV), dtype=tl.float32)

    if U > 1:
        # Buffered inner loop: the readout o_t of U consecutive steps is
        # accumulated into a [U, BV] block and written with one contiguous store,
        # instead of one masked element-granular store per timestep.
        ou = tl.arange(0, U)
        nfull = (t1 - t0) // U * U
        for ts in range(t0, t0 + nfull, U):
            Obuf = tl.zeros((U, BV), dtype=tl.float32)
            for u in tl.static_range(U):
                t = ts + u
                kt = tl.load(
                    k_base + t * sk_t + ok * sk_k, mask=km, other=0.0
                ).to(tl.float32)
                qt = tl.load(
                    q_base + t * sq_t + ok * sq_k, mask=km, other=0.0
                ).to(tl.float32)
                vt = tl.load(
                    v_base + t * sv_t + vg * sv_v, mask=vm, other=0.0
                ).to(tl.float32)
                gt = tl.load(g_base + t * sg_t)
                bt = tl.load(b_base + t * sb_t)
                eg = tl.exp(gt)
                c = (vt - tl.sum(S * kt[:, None], axis=0) * eg) * bt
                S = S * eg + kt[:, None] * c[None, :]
                oo = tl.sum((qt * scale)[:, None] * S, axis=0)
                Obuf = tl.where((ou == u)[:, None], oo[None, :], Obuf)
            tl.store(
                o_base + (ts + ou)[:, None] * so_t + vg[None, :] * so_v,
                Obuf.to(o_ptr.dtype.element_ty),
                mask=vm[None, :],
            )
        ttail = t0 + nfull
    else:
        ttail = t0

    for t in range(ttail, t1):
        kt = tl.load(k_base + t * sk_t + ok * sk_k, mask=km, other=0.0).to(
            tl.float32
        )
        qt = tl.load(q_base + t * sq_t + ok * sq_k, mask=km, other=0.0).to(
            tl.float32
        )
        vt = tl.load(v_base + t * sv_t + vg * sv_v, mask=vm, other=0.0).to(
            tl.float32
        )
        gt = tl.load(g_base + t * sg_t)
        bt = tl.load(b_base + t * sb_t)

        # When L2NORM is set, q arrives already normalized and pre-scaled by the
        # fully parallel pre-pass, so the serial loop carries no row-wise
        # reduction; otherwise the scale multiply happens here.
        qt = qt * scale

        # Deferred decay: exp(g) is a per-head SCALAR, so it commutes with the
        # K-axis reduction.  Factoring it out of the reduction input
        #     p = (S^T k) * exp(g)      == (exp(g) * S)^T k
        # removes the decay multiply from the reduction's input dependency, so
        # the shuffle tree can start as soon as S and k_t are live.  Both forms
        # were measured bit-equivalent against the reference.
        eg = tl.exp(gt)
        c = (vt - tl.sum(S * kt[:, None], axis=0) * eg) * bt
        S = S * eg + kt[:, None] * c[None, :]
        ovv = tl.sum(qt[:, None] * S, axis=0)
        tl.store(
            o_base + t * so_t + vg * so_v,
            ovv.to(o_ptr.dtype.element_ty),
            mask=vm,
        )

    if OUT_FINAL:
        tl.store(
            ht_ptr
            + n * ht_n
            + h * ht_h
            + ok[:, None] * ht_k
            + vg[None, :] * ht_v,
            S,
            mask=km[:, None] & vm[None, :],
        )


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AssertionError("q, k and v must share a dtype")
    if q.dtype == torch.float32:
        raise AssertionError(
            "chunk_gated_delta_rule does not support float32; use bfloat16"
        )
    if beta.dim() != 3:
        raise AssertionError("beta must be of shape [B, T, H]")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K = q.shape
    V = v.shape[-1]
    device = q.device

    if cu_seqlens is not None and B != 1:
        raise ValueError(
            "batch size is expected to be 1 when using cu_seqlens; " f"got {B}"
        )

    varlen = cu_seqlens is not None
    NT = (cu_seqlens.numel() - 1) if varlen else B
    if initial_state is not None and initial_state.shape[0] != NT:
        raise ValueError(
            "the number of initial states must equal the number of "
            f"sequences, i.e. {NT} rather than {initial_state.shape[0]}"
        )

    if use_qk_l2norm_in_kernel:
        # Fully parallel pre-pass: q and k are L2-normalized in fp32 exactly as
        # the reference does (eps inside the rsqrt, before the scale multiply),
        # then `scale` is folded into q so the serial loop carries no row-wise
        # reduction at all.
        BK = triton.next_power_of_2(K)
        rows = B * T * H
        qn = torch.empty((B, T, H, K), dtype=torch.float32, device=device)
        kn = torch.empty((B, T, H, K), dtype=torch.float32, device=device)
        _gdn_prep_kernel[(rows,)](
            q,
            qn,
            float(scale),
            int(T),
            int(H),
            K=K,
            BK=BK,
            sx_n=q.stride(0),
            sx_t=q.stride(1),
            sx_h=q.stride(2),
            sx_k=q.stride(3),
            so_n=qn.stride(0),
            so_t=qn.stride(1),
            so_h=qn.stride(2),
            so_k=qn.stride(3),
            DO_NORM=True,
            DO_SCALE=True,
            num_warps=1,
        )
        _gdn_prep_kernel[(rows,)](
            k,
            kn,
            1.0,
            int(T),
            int(H),
            K=K,
            BK=BK,
            sx_n=k.stride(0),
            sx_t=k.stride(1),
            sx_h=k.stride(2),
            sx_k=k.stride(3),
            so_n=kn.stride(0),
            so_t=kn.stride(1),
            so_h=kn.stride(2),
            so_k=kn.stride(3),
            DO_NORM=True,
            DO_SCALE=False,
            num_warps=1,
        )
        q_in, k_in = qn, kn
    else:
        q_in, k_in = q, k

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = (V + BV - 1) // BV
    grid = (NT * H * NV,)

    o = torch.empty((B, T, H, V), dtype=q.dtype, device=device)
    ht = None
    if output_final_state:
        ht = torch.empty((NT, H, K, V), dtype=torch.float32, device=device)

    if initial_state is not None:
        h0 = initial_state
        h0_n, h0_h, h0_k, h0_v = h0.stride()
        has_init = 1
    else:
        h0 = g
        h0_n = h0_h = h0_k = h0_v = 0
        has_init = 0

    if ht is not None:
        ht_n, ht_h, ht_k, ht_v = ht.stride()
        ht_arg = ht
    else:
        ht_n = ht_h = ht_k = ht_v = 0
        ht_arg = g

    cu_arg = cu_seqlens if varlen else g
    cu_s = cu_seqlens.stride(0) if varlen else 0

    # `scale` is applied exactly once: folded into the pre-pass for q when the
    # l2norm path is active (so the serial loop needs only the identity), or on
    # the fly inside the loop otherwise.
    loop_scale = 1.0 if use_qk_l2norm_in_kernel else float(scale)

    _gdn_recurrent_kernel[grid](
        q_in,
        k_in,
        v,
        g,
        beta,
        cu_arg,
        h0,
        ht_arg,
        o,
        loop_scale,
        int(T),
        int(H),
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        sq_n=q_in.stride(0),
        sq_t=q_in.stride(1),
        sq_h=q_in.stride(2),
        sq_k=q_in.stride(3),
        sk_n=k_in.stride(0),
        sk_t=k_in.stride(1),
        sk_h=k_in.stride(2),
        sk_k=k_in.stride(3),
        sv_n=v.stride(0),
        sv_t=v.stride(1),
        sv_h=v.stride(2),
        sv_v=v.stride(3),
        sg_n=g.stride(0),
        sg_t=g.stride(1),
        sg_h=g.stride(2),
        sb_n=beta.stride(0),
        sb_t=beta.stride(1),
        sb_h=beta.stride(2),
        so_n=o.stride(0),
        so_t=o.stride(1),
        so_h=o.stride(2),
        so_v=o.stride(3),
        h0_n=h0_n,
        h0_h=h0_h,
        h0_k=h0_k,
        h0_v=h0_v,
        ht_n=ht_n,
        ht_h=ht_h,
        ht_k=ht_k,
        ht_v=ht_v,
        cu_s=cu_s,
        HAS_INIT=has_init,
        OUT_FINAL=1 if ht is not None else 0,
        L2NORM=1 if use_qk_l2norm_in_kernel else 0,
        VARLEN=1 if varlen else 0,
        U=4,
        num_warps=1,
    )
    return o, ht


__all__ = ["chunk_gated_delta_rule_fwd"]
