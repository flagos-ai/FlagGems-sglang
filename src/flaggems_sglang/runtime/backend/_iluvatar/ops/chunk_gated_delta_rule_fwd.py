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

"""Chunked gated delta-rule forward (vLLM-Ascend chunk_gated_delta_rule semantics).

Fused per-iteration recurrence: one Triton program per (sequence-slot, head,
V-block) holds the whole [BV, BK] fp32 state tile in registers and walks the
recurrence one timestep at a time with one warp per program.

The optional q/k L2-normalisation is hoisted out of the serial loop into a
separate, fully parallel kernel that writes the fp32 rsqrt factors for every
(b, t, h) once; the recurrence then only multiplies.  Its two extra cross-lane
reductions per timestep were a large fraction of the serial step cost.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _l2norm_factors(
    q_ptr,
    k_ptr,
    nq_ptr,
    nk_ptr,
    T,
    H,
    K,
    NROWS,
    s_qb,
    s_qt,
    s_qh,
    s_kb,
    s_kt,
    s_kh,
    s_nb,
    s_nt,
    s_nh,
    BH: tl.constexpr,
    BK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """One program per ROWS (batch row, timestep) pairs, all heads at once.

    This kernel is serial with the recurrence, so its whole latency is exposed;
    the profile showed it is launch/occupancy bound rather than work bound, and a
    measured sweep preferred ~8 rows per program over one.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, ROWS * BH)
    rowid = pid * ROWS + (r // BH)
    hh = r % BH
    live = rowid < NROWS
    b = rowid // T
    t = rowid % T
    ok = tl.arange(0, BK)
    m = live[:, None] & (hh < H)[:, None] & (ok < K)[None, :]
    rowoff = (b * s_qb + t * s_qt)[:, None]
    qo = rowoff + hh[:, None] * s_qh + ok[None, :]
    q = tl.load(q_ptr + qo, mask=m, other=0.0).to(tl.float32)
    nq = tl.rsqrt(tl.sum(q * q, axis=1) + 1e-6)
    ko = rowoff + hh[:, None] * s_kh + ok[None, :]
    k = tl.load(k_ptr + ko, mask=m, other=0.0).to(tl.float32)
    nk = tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)
    no = (b * s_nb + t * s_nt)[:, None] + hh[:, None] * s_nh
    sm = (live & (hh < H))[:, None]
    tl.store(nq_ptr + no, nq[:, None], mask=sm)
    tl.store(nk_ptr + no, nk[:, None], mask=sm)


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
    h0_ptr,
    ht_ptr,
    cu_ptr,
    nq_ptr,
    nk_ptr,
    scale,
    T,
    H,
    K,
    V,
    s_qb,
    s_qt,
    s_qh,
    s_vb,
    s_vt,
    s_vh,
    s_ob,
    s_ot,
    s_oh,
    s_gb,
    s_gt,
    s_gh,
    s_i0,
    s_i1,
    s_i2,
    s_i3,
    s_f0,
    s_f1,
    s_f2,
    s_f3,
    s_n0,
    s_n1,
    s_n2,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_FIN: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    pv = tl.program_id(1)

    offs_k = tl.arange(0, BK)
    offs_v = pv * BV + tl.arange(0, BV)
    m_k = offs_k < K
    m_v = offs_v < V
    # The state is held transposed, [BV, BK], so that both per-timestep
    # contractions reduce along the inner axis and stay inside one warp.
    mt = m_v[:, None] & m_k[None, :]

    S = tl.zeros([BV, BK], dtype=tl.float32)
    if HAS_INIT:
        S = tl.load(
            h0_ptr
            + n * s_i0
            + h * s_i1
            + offs_v[:, None] * s_i3
            + offs_k[None, :] * s_i2,
            mask=mt,
            other=0.0,
        )

    if HAS_CU:
        # n indexes a packed segment, not a batch row: the segment offset lives
        # in the time axis and the row base stays at the (single) batch slice.
        row = 0
        t0 = tl.load(cu_ptr + n).to(tl.int32)
        t1 = tl.load(cu_ptr + n + 1).to(tl.int32)
    else:
        row = n
        t0 = 0
        t1 = T

    qb = q_ptr + row * s_qb + h * s_qh
    kb = k_ptr + row * s_qb + h * s_qh
    vb = v_ptr + row * s_vb + h * s_vh
    ob = o_ptr + row * s_ob + h * s_oh
    gb = g_ptr + row * s_gb + h * s_gh
    bb = beta_ptr + row * s_gb + h * s_gh
    nb = row * s_n0 + h * s_n2

    for t in range(t0, t1):
        # k/q are loaded directly in the [BV, BK] state-tile layout (every row
        # reads the same elements).  A 1-D load would have to be re-laid-out
        # through shared memory before the elementwise product with the state;
        # profile evidence attributed ~20% of the executed instructions of the
        # loop to that staging (STS.B16/LDS.B16 plus bank conflicts).  v is
        # loaded and o stored in the same row layout for the same reason.
        kt = tl.load(
            kb + t * s_qt + offs_k[None, :],
            mask=m_k[None, :] & m_v[:, None],
            other=0.0,
        ).to(tl.float32)
        qt = tl.load(
            qb + t * s_qt + offs_k[None, :],
            mask=m_k[None, :] & m_v[:, None],
            other=0.0,
        ).to(tl.float32)
        vt = tl.load(
            vb + t * s_vt + offs_v[:, None], mask=m_v[:, None], other=0.0
        ).to(tl.float32)
        gt = tl.load(gb + t * s_gt)
        bt = tl.load(bb + t * s_gt)

        if HAS_NORM:
            # Fold the per-(t,h) normalisation and scale factors onto the [BV]
            # results instead of multiplying the whole [BV, BK] operand tiles.
            # sum(S * (n * x)) == n * sum(S * x) exactly in reals and n is a
            # single scalar per (t, h), so this removes 2*BV*BK multiplies per
            # step from the pipe that profile evidence shows is saturated.
            nq = tl.load(nq_ptr + nb + t * s_n1) * scale
            nk = tl.load(nk_ptr + nb + t * s_n1)
        else:
            nq = scale
            nk = 1.0

        # decay the state by exp(g) BEFORE predicting
        Sd = S * tl.exp(gt)
        # The readout uses the POST-update state, but writing it as
        #     S_new = Sd + k' (x) vc          with k' = nk * k
        #     o     = Sd . q' + (k' . q') * vc
        # makes BOTH contractions read the same decayed state Sd, so they are
        # independent and the readout leaves the loop's critical path (the
        # per-timestep cost is a fixed ~0.9-1.9 us regardless of tile width,
        # i.e. it is serial-chain bound).
        qs = (tl.sum(Sd * qt, axis=1) * nq)[:, None]
        pred = tl.sum(Sd * kt, axis=1) * nk
        kq = (tl.sum(kt * qt, axis=1) * (nk * nq))[:, None]
        vc = bt * (vt - pred[:, None])
        o = qs + kq * vc
        # rank-1 update: outer(k, v_corr) because the state is [K, V].  The
        # normalisation of k is carried on v_corr, not on the k tile.
        S = Sd + (vc * nk) * kt
        tl.store(
            ob + t * s_ot + offs_v[:, None],
            o.to(o_ptr.dtype.element_ty),
            mask=m_v[:, None],
        )

    if HAS_FIN:
        tl.store(
            ht_ptr
            + n * s_f0
            + h * s_f1
            + offs_v[:, None] * s_f3
            + offs_k[None, :] * s_f2,
            S,
            mask=mt,
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

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens; "
                f"got {B}"
            )
        num_segments = int(cu_seqlens.shape[0]) - 1
        if (
            initial_state is not None
            and initial_state.shape[0] != num_segments
        ):
            raise ValueError(
                "the number of initial states must equal the number of "
                f"sequences, i.e. {num_segments} rather than "
                f"{initial_state.shape[0]}"
            )
        N = num_segments
    else:
        N = B

    o = torch.empty((B, T, H, V), dtype=q.dtype, device=q.device)
    ht = (
        torch.empty((N, H, K, V), dtype=torch.float32, device=q.device)
        if output_final_state
        else o
    )
    h0 = initial_state if initial_state is not None else o
    cu = cu_seqlens if cu_seqlens is not None else o

    if use_qk_l2norm_in_kernel:
        nq = torch.empty((B, T, H), dtype=torch.float32, device=q.device)
        nk = torch.empty((B, T, H), dtype=torch.float32, device=q.device)
        # Rows are (batch row, timestep) pairs and the kernel is serial with the
        # recurrence, so give each program a few of them; measured sweep optimum.
        nrows = B * T
        rows = 8 if nrows >= 32 else 1
        _l2norm_factors[(triton.cdiv(nrows, rows),)](
            q,
            k,
            nq,
            nk,
            T,
            H,
            K,
            nrows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            nq.stride(0),
            nq.stride(1),
            nq.stride(2),
            BH=triton.next_power_of_2(H),
            BK=min(128, triton.next_power_of_2(K)),
            ROWS=rows,
            num_warps=4 if (rows * triton.next_power_of_2(H)) >= 32 else 2,
        )
    else:
        nq = o
        nk = o

    BK = min(128, triton.next_power_of_2(K))
    # One warp per program and a small V-block: the two per-timestep contractions
    # then reduce inside a single warp (shuffles) instead of being staged through
    # shared memory with barriers.
    BV = min(64, triton.next_power_of_2(V), max(8, 1024 // BK))
    # The device has 16 SMs, so a shape whose grid holds only a few tens of
    # programs leaves most SMs idle.  A measured BV sweep over all eight timing
    # shapes showed the optimum at >= ~256 programs, so shrink the V-block until
    # that many programs exist (floor of 2 keeps the vector axis usable).
    while BV > 2 and N * H * triton.cdiv(V, BV) < 256:
        BV //= 2
    nw = 1

    grid = (N * H, triton.cdiv(V, BV))
    _gdn_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o,
        h0,
        ht,
        cu,
        nq,
        nk,
        float(scale),
        T,
        H,
        K,
        V,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        *(
            tuple(initial_state.stride())
            if initial_state is not None
            else (0, 0, V, 1)
        ),
        ht.stride(0),
        ht.stride(1),
        ht.stride(2),
        ht.stride(3),
        nq.stride(0) if use_qk_l2norm_in_kernel else 0,
        nq.stride(1) if use_qk_l2norm_in_kernel else 0,
        nq.stride(2) if use_qk_l2norm_in_kernel else 1,
        BK=BK,
        BV=BV,
        HAS_INIT=initial_state is not None,
        HAS_FIN=bool(output_final_state),
        HAS_CU=cu_seqlens is not None,
        HAS_NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=nw,
    )

    if output_final_state:
        return o, ht
    return o, None


__all__ = ["chunk_gated_delta_rule_fwd"]
