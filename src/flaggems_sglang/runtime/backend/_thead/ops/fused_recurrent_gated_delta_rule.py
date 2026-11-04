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

"""Fused recurrent gated delta-rule (GDN) linear attention — Triton implementation.

Reference semantics (all fp32 internally, see definition):
  per (b, hv), per timestep t:
    S <- S * exp(g_t)                      (scalar decay broadcast over (V, K))
    pred = S @ k_t
    vc   = (v_t - pred) * beta_t
    S   <- S + vc (x) k_t
    o_t  = S @ (q_t * scale)               (readout AFTER the update)
  optional L2 norm on q/k: x / sqrt(sum(x^2) + 1e-6), epsilon INSIDE the sqrt,
  applied to q and k separately before q is scaled.

Kernel layout: one program per (v-block, value head, batch).  The fp32 state tile
is held in registers for the whole T loop and is stored transposed as
[BLOCK_K, BLOCK_V] (H[k, v] for S[v, k]), so the pred/readout reductions run along
axis 0.  BLOCK_K covers the full K axis because the pred reduction and the rank-1
update share the k index.
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
    o_ptr,
    h0_ptr,
    ht_ptr,
    scale,
    T,
    H,
    HV,
    K,
    V,
    sqb: tl.constexpr,
    sqt: tl.constexpr,
    sqh: tl.constexpr,
    sqk: tl.constexpr,
    skb: tl.constexpr,
    skt: tl.constexpr,
    skh: tl.constexpr,
    skk: tl.constexpr,
    svb: tl.constexpr,
    svt: tl.constexpr,
    svh: tl.constexpr,
    svv: tl.constexpr,
    sgb: tl.constexpr,
    sgt: tl.constexpr,
    sgh: tl.constexpr,
    sbb: tl.constexpr,
    sbt: tl.constexpr,
    sbh: tl.constexpr,
    sbv: tl.constexpr,
    sob: tl.constexpr,
    sot: tl.constexpr,
    soh: tl.constexpr,
    sov: tl.constexpr,
    shb: tl.constexpr,
    shh: tl.constexpr,
    shv: tl.constexpr,
    shk: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    BETA_HEADWISE: tl.constexpr,
    L2NORM: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_b = tl.program_id(2)
    i_h = i_hv // (HV // H)

    v_offs = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = tl.arange(0, BLOCK_K)
    m_v = v_offs < V
    m_k = k_offs < K
    m_2d = m_k[:, None] & m_v[None, :]
    off_h = i_b * shb + i_hv * shh

    h_ptrs = h0_ptr + off_h + k_offs[:, None] * shk + v_offs[None, :] * shv
    if HAS_INIT:
        b_h = tl.load(h_ptrs, mask=m_2d, other=0.0)
    else:
        b_h = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    ht_ptrs = ht_ptr + off_h + k_offs[:, None] * shk + v_offs[None, :] * shv

    q_ptrs = q_ptr + i_b * sqb + i_h * sqh + k_offs * sqk
    k_ptrs = k_ptr + i_b * skb + i_h * skh + k_offs * skk
    v_ptrs = v_ptr + i_b * svb + i_hv * svh + v_offs * svv
    o_ptrs = o_ptr + i_b * sob + i_hv * soh + v_offs * sov
    g_ptrs = g_ptr + i_b * sgb + i_hv * sgh
    b_ptrs = beta_ptr + i_b * sbb + i_hv * sbh

    o_ty = o_ptr.dtype.element_ty

    for t in range(0, T):
        b_q = tl.load(q_ptrs + t * sqt, mask=m_k, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptrs + t * skt, mask=m_k, other=0.0).to(tl.float32)
        if L2NORM:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
        b_q = b_q * scale

        b_v = tl.load(v_ptrs + t * svt, mask=m_v, other=0.0).to(tl.float32)

        b_g = tl.load(g_ptrs + t * sgt)
        b_h = b_h * tl.exp(b_g)

        b_pred = tl.sum(b_h * b_k[:, None], axis=0)
        b_odec = tl.sum(b_h * b_q[:, None], axis=0)

        if BETA_HEADWISE:
            b_beta = tl.load(
                b_ptrs + t * sbt + v_offs * sbv, mask=m_v, other=0.0
            )
        else:
            b_beta = tl.load(b_ptrs + t * sbt)

        b_vc = (b_v - b_pred) * b_beta
        b_kq = tl.sum(b_k * b_q, axis=0)

        b_h = b_h + b_k[:, None] * b_vc[None, :]
        b_o = b_odec + b_vc * b_kq

        tl.store(o_ptrs + t * sot, b_o.to(o_ty), mask=m_v)

    if STORE_FINAL:
        tl.store(ht_ptrs, b_h, mask=m_2d)


def _next_pow2(x):
    return max(16, triton.next_power_of_2(x))


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
    beta_headwise = beta.dim() == v.dim()

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    final_state = None
    if output_final_state:
        final_state = torch.empty(
            (B, HV, V, K), dtype=torch.float32, device=q.device
        )

    has_init = initial_state is not None
    # Dummy pointer for the optional initial-state load; never dereferenced when
    # HAS_INIT is false.  Use q (read-only) so nothing aliases an output buffer.
    h0 = initial_state if has_init else q
    ht = final_state if output_final_state else q

    BLOCK_K = _next_pow2(K)
    BLOCK_V = min(64, _next_pow2(V))
    wide_tile = BLOCK_V * BLOCK_K
    NV = triton.cdiv(V, BLOCK_V)
    # Low-parallelism regime (grid smaller than one block per SM): the T loop is a
    # serial dependency chain on b_h, so the only way to hide its latency is more
    # resident blocks per SM, obtained by halving the V tile (and the fp32 register
    # tile) so NV doubles.  Only the 4-warp wide tile (wide_tile <= 4096) is
    # register-limited enough for this to pay: at BLOCK_K = 128 the wide tile
    # already runs 8 warps, so halving BLOCK_V keeps the per-thread element count
    # and merely doubles the wave count against the same per-step latency.
    if T >= 2 and wide_tile <= 4096 and NV * HV * B <= 128:
        BLOCK_V = max(16, min(32, _next_pow2(V)))
        NV = triton.cdiv(V, BLOCK_V)
    num_warps = 4 if BLOCK_V * BLOCK_K <= 4096 else 8

    sqb, sqt, sqh, sqk = q.stride()
    skb, skt, skh, skk = k.stride()
    svb, svt, svh, svv = v.stride()
    sgb, sgt, sgh = g.stride()
    if beta_headwise:
        sbb, sbt, sbh, sbv = beta.stride()
    else:
        sbb, sbt, sbh = beta.stride()
        sbv = 0
    sob, sot, soh, sov = o.stride()
    if has_init:
        shb, shh, shv, shk = initial_state.stride()
    else:
        shb, shh, shv, shk = HV * V * K, V * K, K, 1

    _gdn_fwd_kernel[(NV, HV, B)](
        q,
        k,
        v,
        g,
        beta,
        o,
        h0,
        ht,
        float(scale),
        int(T),
        int(H),
        int(HV),
        int(K),
        int(V),
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
        shb,
        shh,
        shv,
        shk,
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        HAS_INIT=has_init,
        STORE_FINAL=bool(output_final_state),
        BETA_HEADWISE=beta_headwise,
        L2NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=num_warps,
        num_stages=1,
    )
    return o, final_state


__all__ = ["fused_recurrent_gated_delta_rule"]
