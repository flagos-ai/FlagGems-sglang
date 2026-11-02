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

"""Chunked gated delta-rule forward (vLLM-Ascend chunk.py semantics).

Chunked schedule (chunk length C = 64).  Chunking is a pure scheduling choice:
the chunked and the per-step forms compute exactly the same gated delta-rule
recurrence, so this reproduces the reference semantics while exposing T-way
parallelism.

Per chunk: t0 = row_start + nc*C is the chunk start, i, j are chunk-local steps,
G_i is the INCLUSIVE chunk-local cumsum of g (log space, g <= 0), decay = exp(G_last),
and k, v, q are upcast to fp32.

  K1 prep  grid (chunk, row*head) -- no cross-chunk dependency
     kk     = k k^T  (accumulated over K-tiles)
     L[i,j] = beta_i (k_i . k_j) exp(G_i - G_j)   strictly lower, 0 elsewhere
     T_inv  = (I + L)^-1 = sum_{m<C} (-L)^m = (I-L)(I-L^2)(I-L^4)...  (nilpotent)
     w      = T_inv (beta * exp(G) * k)   [C, K]
     u      = T_inv (beta * v)            [C, V]
     kd     = k * exp(G_last - G)         [C, K]
  K2 scan  grid (K-tile, V-tile, row*head), serial over chunks
     store the INCOMING state S_c, then
     vcorr = u - w S_c ,   S = decay * S + kd^T vcorr
  K3 out   grid (chunk, V-tile, row*head) -- fully parallel
     o = exp(G) * ((q*scale) S_c) + P_o vcorr
     P_o[i,j] = (q_i . k_j) exp(G_i - G_j) for j <= i   (lower INCLUSIVE: the
     j = i term has coefficient exp(0) = 1 and must be kept)

The l2 normalisation factor is a per-(t, h) row quantity, so K1 and K3 apply it
to the accumulated q k^T products after the fact (row/column scaling of the
product) instead of scaling the operands, which lets the sum-of-squares and the
dot share a single pass over q and k, and lets the number of inner tile
iterations be a constexpr that matches the actual K/V extent (no fully masked
tile iterations when the extent is 64).
"""

import torch
import triton
import triton.language as tl

CS: int = 64  # chunk length
TK: int = 64  # K-axis tile
TV: int = 32  # V-axis tile


# ---------------------------------------------------------------- K1: prep
@triton.jit
def _k1_prep(
    k_ptr,
    v_ptr,
    g_ptr,
    b_ptr,
    kd_ptr,
    w_ptr,
    u_ptr,
    dec_ptr,
    cu_ptr,
    TLEN,
    s_kb,
    s_kt,
    s_kh,
    s_vb,
    s_vt,
    s_vh,
    s_gb,
    s_gt,
    s_gh,
    H,
    KDIM,
    VDIM,
    NCMAX,
    NKT,
    NVT,
    L2NORM: tl.constexpr,
    VARLEN: tl.constexpr,
    C: tl.constexpr,
    TILEK: tl.constexpr,
    VT: tl.constexpr,
    LTK: tl.constexpr,
    LTV: tl.constexpr,
):
    nc = tl.program_id(0)
    rh = tl.program_id(1)
    row = rh // H
    h = rh % H

    if VARLEN:
        beg = tl.load(cu_ptr + row).to(tl.int32)
        end = tl.load(cu_ptr + row + 1).to(tl.int32)
        dbi = 0
    else:
        beg = 0
        end = TLEN
        dbi = row

    io = tl.arange(0, C)
    e = tl.arange(0, TILEK)
    t = beg + nc * C + io
    mv = t < end

    gb = g_ptr + dbi * s_gb + h * s_gh
    bb = b_ptr + dbi * s_gb + h * s_gh
    gv = tl.load(gb + t * s_gt, mask=mv, other=0.0)
    bv = tl.load(bb + t * s_gt, mask=mv, other=0.0)
    G = tl.cumsum(gv, axis=0)
    total = tl.sum(gv, axis=0)

    kbase = k_ptr + dbi * s_kb + h * s_kh
    vbase = v_ptr + dbi * s_vb + h * s_vh

    # one pass over k: sum of squares (for the l2 factor) and k k^T
    ss = tl.zeros((C,), dtype=tl.float32)
    kk = tl.zeros((C, C), dtype=tl.float32)
    for kb in range(0, LTK):
        ko = kb * TILEK + e
        m = mv[:, None] & (ko[None, :] < KDIM)
        kc = tl.load(
            kbase + t[:, None] * s_kt + ko[None, :], mask=m, other=0.0
        ).to(tl.float32)
        if L2NORM:
            ss += tl.sum(kc * kc, axis=1)
        kk += tl.dot(kc, tl.trans(kc))
    nrm = tl.full((C,), 1.0, dtype=tl.float32)
    if L2NORM:
        nrm = tl.rsqrt(ss + 1e-6)
        kk = kk * nrm[:, None] * nrm[None, :]

    # T_inv via the nilpotent series: S <- S + B S, B <- B B,  S = sum_{m<64} B^m
    S = tl.where(io[:, None] == io[None, :], 1.0, 0.0)
    B = (
        -bv[:, None]
        * kk
        * tl.exp(
            tl.where(
                io[:, None] > io[None, :],
                G[:, None] - G[None, :],
                float("-inf"),
            )
        )
    )
    for _ in range(6):
        S = S + tl.dot(B, S)
        B = tl.dot(B, B)

    blk = (row * NCMAX + nc) * H + h

    # u = T_inv (beta * v)
    for vb in range(0, LTV):
        vo = vb * VT + tl.arange(0, VT)
        m = mv[:, None] & (vo[None, :] < VDIM)
        vc = tl.load(
            vbase + t[:, None] * s_vt + vo[None, :], mask=m, other=0.0
        ).to(tl.float32)
        ub = tl.dot(S, bv[:, None] * vc)
        tl.store(
            u_ptr
            + ((blk * NVT + vb) * C) * VT
            + io[:, None] * VT
            + tl.arange(0, VT)[None, :],
            ub,
        )

    eg = tl.exp(G)
    sc = tl.exp(total - G)
    for kb in range(0, LTK):
        ko = kb * TILEK + e
        m = mv[:, None] & (ko[None, :] < KDIM)
        kc = (
            tl.load(
                kbase + t[:, None] * s_kt + ko[None, :], mask=m, other=0.0
            ).to(tl.float32)
            * nrm[:, None]
        )
        wb = tl.dot(S, (bv * eg)[:, None] * kc)
        off = ((blk * NKT + kb) * C) * TILEK + io[:, None] * TILEK + e[None, :]
        tl.store(w_ptr + off, wb)
        tl.store(kd_ptr + off, kc * sc[:, None])

    tl.store(dec_ptr + blk, tl.exp(total))


# ---------------------------------------------------------------- K2: scan
@triton.jit
def _k2_scan(
    kd_ptr,
    w_ptr,
    u_ptr,
    dec_ptr,
    sc_ptr,
    init_ptr,
    fin_ptr,
    cu_ptr,
    TLEN,
    s_in0,
    s_in1,
    s_in2,
    s_fn0,
    s_fn1,
    s_fn2,
    H,
    KDIM,
    VDIM,
    NCMAX,
    NKT,
    NVT,
    NSTEP,
    HAS_INIT: tl.constexpr,
    WRITE_FIN: tl.constexpr,
    VARLEN: tl.constexpr,
    C: tl.constexpr,
    TILEK: tl.constexpr,
    VT: tl.constexpr,
):
    kb = tl.program_id(0)
    vt = tl.program_id(1)
    rh = tl.program_id(2)
    row = rh // H
    h = rh % H

    if VARLEN:
        beg = tl.load(cu_ptr + row).to(tl.int32)
        end = tl.load(cu_ptr + row + 1).to(tl.int32)
    else:
        beg = 0
        end = TLEN
    ncr = (end - beg + C - 1) // C

    e = tl.arange(0, TILEK)
    io = tl.arange(0, C)
    vv = tl.arange(0, VT)
    ko = kb * TILEK + e
    vo = vt * VT + vv
    mk = ko < KDIM
    mv = vo < VDIM
    m2 = mk[:, None] & mv[None, :]

    if HAS_INIT:
        state = tl.load(
            init_ptr
            + row * s_in0
            + h * s_in1
            + ko[:, None] * s_in2
            + vo[None, :],
            mask=m2,
            other=0.0,
        )
    else:
        state = tl.zeros((TILEK, VT), dtype=tl.float32)

    for cc in range(0, NSTEP):
        if cc < ncr:
            blk = (row * NCMAX + cc) * H + h
            # incoming state S_c (read by K3)
            tl.store(
                sc_ptr
                + ((blk * NKT + kb) * NVT + vt) * (TILEK * VT)
                + e[:, None] * VT
                + vv[None, :],
                state,
            )
            wv = tl.load(
                w_ptr
                + ((blk * NKT + kb) * C) * TILEK
                + io[:, None] * TILEK
                + e[None, :]
            ).to(tl.float32)
            kdv = tl.load(
                kd_ptr
                + ((blk * NKT + kb) * C) * TILEK
                + io[:, None] * TILEK
                + e[None, :]
            ).to(tl.float32)
            uv = tl.load(
                u_ptr
                + ((blk * NVT + vt) * C) * VT
                + io[:, None] * VT
                + vv[None, :]
            ).to(tl.float32)
            vcorr = uv - tl.dot(wv, state)
            d = tl.load(dec_ptr + blk)
            state = d * state + tl.dot(tl.trans(kdv), vcorr)

    if WRITE_FIN:
        tl.store(
            fin_ptr
            + row * s_fn0
            + h * s_fn1
            + ko[:, None] * s_fn2
            + vo[None, :],
            state,
            mask=m2,
        )


# ---------------------------------------------------------------- K3: out
@triton.jit
def _k3_out(
    q_ptr,
    k_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    sc_ptr,
    o_ptr,
    cu_ptr,
    TLEN,
    s_qb,
    s_qt,
    s_qh,
    s_kb,
    s_kt,
    s_kh,
    s_gb,
    s_gt,
    s_gh,
    s_ob,
    s_ot,
    s_oh,
    scale,
    H,
    KDIM,
    VDIM,
    NCMAX,
    NKT,
    NVT,
    L2NORM: tl.constexpr,
    VARLEN: tl.constexpr,
    C: tl.constexpr,
    TILEK: tl.constexpr,
    VT: tl.constexpr,
    LTK: tl.constexpr,
):
    nc = tl.program_id(0)
    vt = tl.program_id(1)
    rh = tl.program_id(2)
    row = rh // H
    h = rh % H

    if VARLEN:
        beg = tl.load(cu_ptr + row).to(tl.int32)
        end = tl.load(cu_ptr + row + 1).to(tl.int32)
        dbi = 0
    else:
        beg = 0
        end = TLEN
        dbi = row
    act = nc * C < (end - beg)

    io = tl.arange(0, C)
    e = tl.arange(0, TILEK)
    vv = tl.arange(0, VT)
    vo = vt * VT + vv
    t = beg + nc * C + io
    mv = (t < end) & act

    qbase = q_ptr + dbi * s_qb + h * s_qh
    kbase = k_ptr + dbi * s_kb + h * s_kh
    gb = g_ptr + dbi * s_gb + h * s_gh
    gv = tl.load(gb + t * s_gt, mask=mv, other=0.0)
    G = tl.cumsum(gv, axis=0)

    blk = (row * NCMAX + nc) * H + h
    sq = tl.zeros((C,), dtype=tl.float32)
    sk = tl.zeros((C,), dtype=tl.float32)
    po = tl.zeros((C, C), dtype=tl.float32)
    qs = tl.zeros((C, VT), dtype=tl.float32)
    ws = tl.zeros((C, VT), dtype=tl.float32)
    for kb in range(0, LTK):
        ko = kb * TILEK + e
        m = mv[:, None] & (ko[None, :] < KDIM)
        qc = tl.load(
            qbase + t[:, None] * s_qt + ko[None, :], mask=m, other=0.0
        ).to(tl.float32)
        kc = tl.load(
            kbase + t[:, None] * s_kt + ko[None, :], mask=m, other=0.0
        ).to(tl.float32)
        if L2NORM:
            sq += tl.sum(qc * qc, axis=1)
            sk += tl.sum(kc * kc, axis=1)
        po += tl.dot(qc, tl.trans(kc))
        sl = tl.load(
            sc_ptr
            + ((blk * NKT + kb) * NVT + vt) * (TILEK * VT)
            + e[:, None] * VT
            + vv[None, :]
        ).to(tl.float32)
        qs += tl.dot(qc, sl)
        wv = tl.load(
            w_ptr
            + ((blk * NKT + kb) * C) * TILEK
            + io[:, None] * TILEK
            + e[None, :]
        ).to(tl.float32)
        ws += tl.dot(wv, sl)

    qfac = tl.full((C,), scale, dtype=tl.float32)
    kfac = tl.full((C,), 1.0, dtype=tl.float32)
    if L2NORM:
        qfac = qfac * tl.rsqrt(sq + 1e-6)
        kfac = tl.rsqrt(sk + 1e-6)
    po = po * qfac[:, None] * kfac[None, :]
    qs = qs * qfac[:, None]

    uv = tl.load(
        u_ptr + ((blk * NVT + vt) * C) * VT + io[:, None] * VT + vv[None, :]
    ).to(tl.float32)
    vcorr = uv - ws
    po = po * tl.exp(
        tl.where(
            io[:, None] >= io[None, :], G[:, None] - G[None, :], float("-inf")
        )
    )
    o = tl.exp(G)[:, None] * qs + tl.dot(po, vcorr)

    ob = o_ptr + dbi * s_ob + h * s_oh
    tl.store(
        ob + t[:, None] * s_ot + vo[None, :],
        o.to(o_ptr.dtype.element_ty),
        mask=mv[:, None] & (vo[None, :] < VDIM),
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
        raise AssertionError("chunk_gated_delta_rule does not support float32")
    if beta.dim() != 3:
        raise AssertionError("beta must be of shape [B, T, H]")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    batch, t_len, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    dev = q.device

    varlen = cu_seqlens is not None
    if varlen:
        if batch != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens"
            )
        cu = cu_seqlens.to(dev)
        bounds = [int(x) for x in cu.detach().cpu().tolist()]
        nseg = len(bounds) - 1
        if initial_state is not None and initial_state.shape[0] != nseg:
            raise ValueError(
                "the number of initial states must equal the number of sequences"
            )
        rows = nseg
        # grid sizing only: per-row masking makes any upper bound safe
        ncmax = max(
            1,
            max(
                (bounds[i + 1] - bounds[i] + CS - 1) // CS for i in range(nseg)
            ),
        )
    else:
        cu = q  # dummy pointer, never dereferenced when VARLEN is false
        rows = batch
        ncmax = max(1, (t_len + CS - 1) // CS)

    nkt = triton.cdiv(k_dim, TK)
    nvt = triton.cdiv(v_dim, TV)
    ltk = triton.cdiv(k_dim, TK)  # tiles actually covering K
    ltv = triton.cdiv(v_dim, TV)  # tiles actually covering V
    nblk = rows * ncmax * heads

    kd = torch.empty(nblk * nkt * CS * TK, dtype=torch.float16, device=dev)
    wr = torch.empty(nblk * nkt * CS * TK, dtype=torch.float16, device=dev)
    uu = torch.empty(nblk * nvt * CS * TV, dtype=torch.float16, device=dev)
    dcv = torch.empty(nblk, dtype=torch.float32, device=dev)
    sc = torch.empty(
        nblk * nkt * nvt * TK * TV, dtype=torch.float16, device=dev
    )

    o = torch.empty((batch, t_len, heads, v_dim), dtype=q.dtype, device=dev)

    has_init = initial_state is not None
    write_fin = bool(output_final_state)
    init = initial_state if has_init else q
    fin = (
        torch.empty(
            (rows, heads, k_dim, v_dim), dtype=torch.float32, device=dev
        )
        if write_fin
        else q
    )
    l2 = bool(use_qk_l2norm_in_kernel)

    _k1_prep[(ncmax, rows * heads)](
        k,
        v,
        g,
        beta,
        kd,
        wr,
        uu,
        dcv,
        cu,
        t_len,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        heads,
        k_dim,
        v_dim,
        ncmax,
        nkt,
        nvt,
        L2NORM=l2,
        VARLEN=varlen,
        C=CS,
        TILEK=TK,
        VT=TV,
        LTK=ltk,
        LTV=ltv,
        num_warps=4,
    )
    _k2_scan[(nkt, nvt, rows * heads)](
        kd,
        wr,
        uu,
        dcv,
        sc,
        init,
        fin,
        cu,
        t_len,
        init.stride(0) if has_init else 1,
        init.stride(1) if has_init else 1,
        init.stride(2) if has_init else 1,
        fin.stride(0) if write_fin else 1,
        fin.stride(1) if write_fin else 1,
        fin.stride(2) if write_fin else 1,
        heads,
        k_dim,
        v_dim,
        ncmax,
        nkt,
        nvt,
        ncmax,
        HAS_INIT=has_init,
        WRITE_FIN=write_fin,
        VARLEN=varlen,
        C=CS,
        TILEK=TK,
        VT=TV,
        num_warps=8,
    )
    _k3_out[(ncmax, nvt, rows * heads)](
        q,
        k,
        g,
        wr,
        uu,
        sc,
        o,
        cu,
        t_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        scale,
        heads,
        k_dim,
        v_dim,
        ncmax,
        nkt,
        nvt,
        L2NORM=l2,
        VARLEN=varlen,
        C=CS,
        TILEK=TK,
        VT=TV,
        LTK=ltk,
        num_warps=4,
    )

    return o, (fin if write_fin else None)


__all__ = ["chunk_gated_delta_rule_fwd"]
