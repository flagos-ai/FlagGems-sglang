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

"""Chunked gated delta-rule forward.

Two implementations share this entry point:

1. Dense sequences go through the chunked decomposition (ported from
   FlagGems-vllm ``_enflame/ops/FLA``, GCU-verified): a fused
   cumsum+KKT+solve_tril kernel produces the chunk-local inverse, one
   recompute kernel materializes w/u, one recurrence kernel passes the
   chunk-level state h, and one output kernel computes o.  The intra-chunk
   work runs as [64, *] matrix ops instead of per-timestep rank-1 updates,
   which cuts the serial chain from T steps to T/64 chunk steps.

2. Packed (cu_seqlens) sequences -- whose chunked varlen form still hits
   GCU grid limits -- fall back to the decay-horizon segmented fused
   recurrent kernel below.  The recurrence is column-separable in V, and
   the state decays by at least exp(-0.693) per step, so a segment may
   start from the zero state OVL steps before its output range and skip
   writing the warm-up steps.  Segments are mutually independent, so the
   serial chain drops from T to SEG + OVL steps and the grid widens by
   about T/SEG.

State layout: initial_state / final_state are [N, H, K, V] (key axis first),
so the state is held transposed as [BLOCK_V, BLOCK_K] in registers, which
makes both reductions (prediction and readout) run along the last axis.  The
rank-1 update is therefore outer(k, c) with V on the rows.

The per-(b, t, h) q/k L2 normalization (``use_qk_l2norm_in_kernel``) is a
row operation that does not touch the recurrent state, so it runs in one
fully data-parallel pass instead of inside the sequential loop.  They are
pure normalizations; the ``scale`` multiply stays in the consumer kernels
and is applied exactly once.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.runtime.backend._enflame.ops._fla_chunk.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from flaggems_sglang.runtime.backend._enflame.ops._fla_chunk.chunk_o import (
    chunk_fwd_o,
)
from flaggems_sglang.runtime.backend._enflame.ops._fla_chunk.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flaggems_sglang.runtime.backend._enflame.ops._fla_chunk.wy_fast import (
    recompute_w_u_fwd,
)

_SEG = 64
_OVL = 16
_TARGET_PROGS = 1024
_MAX_BLOCK_V = 16


@triton.jit
def _qk_prep_kernel(
    x_ptr,
    y_ptr,
    TH,
    H,
    K,
    s_x0,
    s_x1,
    s_x2,
    s_y0,
    s_y1,
    s_y2,
    ROWS,
    RPB: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program normalizes RPB (b, t, h) rows of x into y, in float32."""
    rr = tl.program_id(0) * RPB + tl.arange(0, RPB)
    ok = rr < ROWS
    b = rr // TH
    rem = rr % TH
    t = rem // H
    h = rem % H

    offs = tl.arange(0, BLOCK_K)
    m = ok[:, None] & (offs < K)[None, :]
    src = (
        x_ptr
        + b[:, None] * s_x0
        + t[:, None] * s_x1
        + h[:, None] * s_x2
        + offs[None, :]
    )
    x = tl.load(src, mask=m, other=0.0).to(tl.float32)
    n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
    dst = (
        y_ptr
        + b[:, None] * s_y0
        + t[:, None] * s_y1
        + h[:, None] * s_y2
        + offs[None, :]
    )
    tl.store(dst, n.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _qk_prep_flat_kernel(
    x_ptr,
    y_ptr,
    ROWS,
    K,
    RPB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
):
    """Contiguous-input fast path of ``_qk_prep_kernel``.

    A contiguous ``[B, T, H, K]`` tensor is just a flat ``[B*T*H, K]`` matrix,
    so row ``rr`` lives at ``rr * K`` with no div/mod and no per-axis strides.
    The whole ``[RPB, BLOCK_K]`` tile is one fully coalesced region, which the
    strided pointer arithmetic of the generic kernel cannot prove."""
    rr = tl.program_id(0) * RPB + tl.arange(0, RPB)
    offs = tl.arange(0, BLOCK_K)
    if EVEN_ROWS:
        # rows % RPB == 0 and BLOCK_K == K: no row tail, no column tail
        x = tl.load(x_ptr + rr[:, None] * K + offs[None, :])
        x = x.to(tl.float32)
        n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
        y = y_ptr + rr[:, None] * K + offs[None, :]
        tl.store(y, n.to(y.dtype.element_ty))
    else:
        m = (rr < ROWS)[:, None] & (offs < K)[None, :]
        x = tl.load(x_ptr + rr[:, None] * K + offs[None, :], mask=m, other=0.0)
        x = x.to(tl.float32)
        n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
        y = y_ptr + rr[:, None] * K + offs[None, :]
        tl.store(y, n.to(y.dtype.element_ty), mask=m)


@triton.jit
def _qk_prep_flat2_kernel(
    q_ptr,
    k_ptr,
    qn_ptr,
    kn_ptr,
    ROWS,
    K,
    RPB: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused two-tensor variant of the flat fast path (q and k share a shape).

    One launch normalizes both tensors: programs ``[0, half)`` cover q and the
    rest cover k.  Requires ``ROWS % RPB == 0`` so no tile straddles the q/k
    boundary, and ``BLOCK_K == K`` so every access is unmasked."""
    pid = tl.program_id(0)
    half = tl.num_programs(0) // 2
    offs = tl.arange(0, BLOCK_K)
    if pid < half:
        rr = pid * RPB + tl.arange(0, RPB)
        x = tl.load(q_ptr + rr[:, None] * K + offs[None, :])
        x = x.to(tl.float32)
        n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
        y = qn_ptr + rr[:, None] * K + offs[None, :]
        tl.store(y, n.to(y.dtype.element_ty))
    else:
        rr = (pid - half) * RPB + tl.arange(0, RPB)
        x = tl.load(k_ptr + rr[:, None] * K + offs[None, :])
        x = x.to(tl.float32)
        n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
        y = kn_ptr + rr[:, None] * K + offs[None, :]
        tl.store(y, n.to(y.dtype.element_ty))


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bt_ptr,
    hi_ptr,
    ho_ptr,
    cu_ptr,
    o_ptr,
    T,
    H,
    K,
    V,
    scale,
    SEG,
    OVL,
    NBH,
    s_qb,
    s_qt,
    s_qh,
    s_vb,
    s_vt,
    s_vh,
    s_gb,
    s_gt,
    s_gh,
    s_bb,
    s_bt,
    s_bh,
    s_sn,
    s_sh,
    s_sk,
    s_ob,
    s_ot,
    s_oh,
    HAS_INIT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    seg = pid // NBH
    bh = pid % NBH
    h = bh % H
    n = bh // H

    offs_k = tl.arange(0, BLOCK_K)
    mk = offs_k < K
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mv = offs_v < V
    m_s = mk[:, None] & mv[None, :]

    if IS_VARLEN:
        seq_begin = tl.load(cu_ptr + n).to(tl.int32)
        seq_end = tl.load(cu_ptr + n + 1).to(tl.int32)
        b = 0
    else:
        seq_begin = 0
        seq_end = T
        b = n

    s0 = seq_begin + seg * SEG
    t_hi = tl.minimum(s0 + SEG, seq_end)
    # the OVL steps before the output range are the discarded warm-up
    t_lo = tl.minimum(tl.maximum(seq_begin, s0 - OVL), t_hi)
    is_last = (s0 < seq_end) & (s0 + SEG >= seq_end)

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_qb + h * s_qh
    v_base = v_ptr + b * s_vb + h * s_vh
    g_base = g_ptr + b * s_gb + h * s_gh
    b_base = bt_ptr + b * s_bb + h * s_bh
    o_base = o_ptr + b * s_ob + h * s_oh

    s_ptrs = (
        hi_ptr + n * s_sn + h * s_sh + offs_k[:, None] * s_sk + offs_v[None, :]
    )
    if HAS_INIT:
        raw = tl.load(s_ptrs, mask=m_s, other=0.0).to(tl.float32)
        # the true initial state feeds this segment only when its warm-up window
        # reaches the beginning of the sequence
        S = tl.trans(tl.where(t_lo <= seq_begin, raw, 0.0))
    else:
        S = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    for t in tl.range(t_lo, t_hi, num_stages=4):
        kt = tl.load(k_base + t * s_qt + offs_k, mask=mk, other=0.0).to(
            tl.float32
        )
        qt = tl.load(q_base + t * s_qt + offs_k, mask=mk, other=0.0).to(
            tl.float32
        )
        vt = tl.load(v_base + t * s_vt + offs_v, mask=mv, other=0.0).to(
            tl.float32
        )
        gt = tl.load(g_base + t * s_gt).to(tl.float32)
        bt = tl.load(b_base + t * s_bt).to(tl.float32)
        qt = qt * scale

        # decay, predict k, delta-rule correction, rank-1 update
        S = S * tl.exp(gt)
        pred = tl.sum(S * kt[None, :], axis=1)
        c = bt * (vt - pred)
        S = S + c[:, None] * kt[None, :]

        # read out with q *after* the update
        if t >= s0:
            o = tl.sum(S * qt[None, :], axis=1)
            tl.store(
                o_base + t * s_ot + offs_v,
                o.to(o_ptr.dtype.element_ty),
                mask=mv,
            )

    if OUT_FINAL:
        if is_last:
            ho = ho_ptr + n * s_sn + h * s_sh
            tl.store(
                ho + offs_k[:, None] * s_sk + offs_v[None, :],
                tl.trans(S).to(ho_ptr.dtype.element_ty),
                mask=m_s,
            )


def _pow2(x):
    return max(16, triton.next_power_of_2(int(x)))


def _pick_block_v(chains, v_dim, nseg, target):
    """BLOCK_V for the V split: enough programs to fill the 78-SM part, but not
    so many that every V block of the same chain re-reads q, k, g and beta."""
    total = int(v_dim) * max(1, int(chains)) * max(1, int(nseg))
    v = max(1, total // max(1, int(target)))
    return max(
        4,
        min(
            _MAX_BLOCK_V,
            triton.next_power_of_2(v),
            triton.next_power_of_2(max(1, int(v_dim))),
        ),
    )


def _qk_l2norm(q, k):
    """Row L2-normalize q and k in fp32, stored back in the input dtype.

    Contiguous inputs run through the flat fast path: a contiguous
    ``[B, T, H, K]`` tensor is a flat ``[B*T*H, K]`` matrix, so the tile per
    program is one fully coalesced region (measured on T=2048: 2x74.7us for
    the strided form -> ~18us fused, and bigger RPB tiles win because the
    kernel is per-CTA latency bound, not bandwidth bound).  q and k share a
    shape, so both normalize in one fused launch whenever the tiling is
    exact.  Non-contiguous or odd-sized inputs keep the original strided
    kernel (RPB stays in the compile-verified {32, 64} set there)."""
    batch, t_len, heads, k_dim = q.shape
    rows = batch * t_len * heads
    block_k = _pow2(k_dim)
    qn = torch.empty_like(q)
    kn = torch.empty_like(k)

    if (
        q.is_contiguous()
        and k.is_contiguous()
        and q.shape == k.shape
        and block_k == k_dim
    ):
        for rpb in (512, 256):
            if rows % rpb == 0:
                prep_grid = (2 * (rows // rpb),)
                _qk_prep_flat2_kernel[prep_grid](
                    q,
                    k,
                    qn,
                    kn,
                    rows,
                    k_dim,
                    RPB=rpb,
                    BLOCK_K=block_k,
                    num_warps=4,
                )
                return qn, kn
        # no exact tiling: masked flat path, one launch per tensor
        rpb = 256
        prep_grid = (triton.cdiv(rows, rpb),)
        for src, dst in ((q, qn), (k, kn)):
            _qk_prep_flat_kernel[prep_grid](
                src,
                dst,
                rows,
                k_dim,
                RPB=rpb,
                BLOCK_K=block_k,
                EVEN_ROWS=False,
                num_warps=4,
            )
        return qn, kn

    # RPB=1 (a [1, BLOCK_K] rank-2 tile) trips the memref_ext.slice_pad_start
    # attribute-conversion bug; pad the row dimension to >= 2. RPB is snapped
    # to the compile-verified set {32, 64} (16 tripped a deslice variant);
    # measured on T=2048 (rows=32768): rpb 8 -> 366us, 32 -> 114us, 64/nw4
    # -> 74us per launch.
    rpb = 64 if rows >= 16384 else 32
    num_warps = 4 if rpb == 64 else 2
    prep_grid = (triton.cdiv(rows, rpb),)
    for src, dst in ((q, qn), (k, kn)):
        _qk_prep_kernel[prep_grid](
            src,
            dst,
            t_len * heads,
            heads,
            k_dim,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            dst.stride(0),
            dst.stride(1),
            dst.stride(2),
            rows,
            RPB=rpb,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    return qn, kn


def _chunk_size_for_sequence(t_len: int) -> int:
    return min(64, max(16, 1 << (t_len - 1).bit_length()))


def _chunked_dense_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
):
    """Dense-path chunked decomposition (see module docstring, item 1)."""
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1])
    g_cumsum, a_inv = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=None,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    block_v = 128 if v.shape[-1] == 128 else 64
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        A=a_inv,
        cu_seqlens=None,
        block_v=block_v,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=None,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=None,
        chunk_size=chunk_size,
    )
    return o, final_state


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

    batch, t_len, heads, k_dim = q.shape
    v_dim = v.shape[-1]

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens; "
                f"got {batch}"
            )
        n_seq = int(cu_seqlens.shape[0]) - 1
        has_cu = True
        # GCU: Triton reads int64 buffers with garbage values (torch_gcu
        # stores them in narrowed 4-byte slots); normalize to int32 once so
        # the kernel reads real sequence offsets (public contract stays
        # torch.long).
        if cu_seqlens.dtype == torch.int64:
            cu_seqlens = cu_seqlens.to(torch.int32)
    else:
        n_seq = batch
        has_cu = False

    if initial_state is not None and initial_state.shape[0] != n_seq:
        raise ValueError(
            "the number of initial states must equal the number of "
            f"sequences, i.e. {n_seq} rather than {initial_state.shape[0]}"
        )

    qk_q, qk_k = q, k
    if use_qk_l2norm_in_kernel:
        qk_q, qk_k = _qk_l2norm(q, k)

    # The chunked decomposition reorders the intra-chunk math through the
    # (I - L)^-1 inverse, which is numerically safe only for normalized q/k
    # (row-bounded k_i.k_j and the log-sigmoid g contract give it geometric
    # off-diagonal decay). Bare (unnormalized) inputs make the inverse amplify
    # rounding noise to overflow, so l2norm-off calls keep the sequential
    # recurrent kernel, which cancels step by step and matches the baseline.
    if (
        use_qk_l2norm_in_kernel
        and not has_cu
        and k_dim <= 256
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        return _chunked_dense_fwd(
            qk_q,
            qk_k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
        )

    o = torch.empty(
        (batch, t_len, heads, v_dim), dtype=q.dtype, device=q.device
    )
    if output_final_state:
        final_state = torch.empty(
            (n_seq, heads, k_dim, v_dim), dtype=torch.float32, device=q.device
        )
    else:
        final_state = None

    BLOCK_K = _pow2(k_dim)

    # segment the packed path too: the grid is indexed per sequence with the
    # sequence bounds read on device, so a host-side segment count of
    # cdiv(t_len, SEG) is a safe upper bound -- programs whose segment starts
    # past their sequence's end run an empty loop and write nothing
    nseg = max(1, (t_len + _SEG - 1) // _SEG)
    BLOCK_V = _pick_block_v(n_seq * heads, v_dim, nseg, _TARGET_PROGS)
    NV = (v_dim + BLOCK_V - 1) // BLOCK_V

    if initial_state is not None:
        s_strides = initial_state.stride()
    elif output_final_state:
        s_strides = final_state.stride()
    else:
        s_strides = (heads * k_dim * v_dim, k_dim * v_dim, v_dim)

    _gdn_fwd_kernel[(nseg * n_seq * heads, NV)](
        qk_q,
        qk_k,
        v,
        g,
        beta,
        initial_state if initial_state is not None else o,
        final_state if final_state is not None else o,
        cu_seqlens if cu_seqlens is not None else o,
        o,
        t_len,
        heads,
        k_dim,
        v_dim,
        scale,
        _SEG,
        _OVL,
        n_seq * heads,
        qk_q.stride(0),
        qk_q.stride(1),
        qk_q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        s_strides[0],
        s_strides[1],
        s_strides[2],
        o.stride(0),
        o.stride(1),
        o.stride(2),
        HAS_INIT=initial_state is not None,
        IS_VARLEN=has_cu,
        OUT_FINAL=output_final_state,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        num_warps=1,
    )

    return o, final_state


__all__ = ["chunk_gated_delta_rule_fwd"]
