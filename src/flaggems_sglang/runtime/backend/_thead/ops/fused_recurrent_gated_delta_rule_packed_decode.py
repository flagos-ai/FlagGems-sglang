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

"""Fused GDN packed decode (single token) in Triton.

One program owns a BLOCK_HV x BLOCK_V x K slab of the paged recurrent state of
one sequence, so the state is read exactly once and written exactly once per
call.  All arithmetic happens in fp32; the only rounding interfaces are the loads
(dtype -> fp32), the deliberate beta round-trip through b's dtype and the two
stores (fp32 -> declared dtype).

Measured context (ZW810E, 64 SMs, 64 MB L2): the paged state read+write is ~99%
of the DRAM traffic, so this is a state-bandwidth problem.  With the tile
selection below the large shapes run at ~95% of the measured same-footprint copy
roofline (522 us vs ~497 us at 2.16 TB/s on qwen3next-b512), so the remaining
headroom is in the small grids, which are limited by CTA dispatches and per-CTA
setup rather than bandwidth.

Every shape and stride is a constexpr, so all address arithmetic folds to
immediates and no runtime scalar argument is needed except `scale`.
"""

import triton
import triton.language as tl

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)


@triton.jit
def _gdn_packed_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    RATIO: tl.constexpr,
    Q_DIM: tl.constexpr,
    HV_GROUPS: tl.constexpr,
    N_SLOTS: tl.constexpr,
    STRIDE_QKV_TOK: tl.constexpr,
    SS_SLOT: tl.constexpr,
    SS_HV: tl.constexpr,
    SS_V: tl.constexpr,
    SS_K: tl.constexpr,
    OS_N: tl.constexpr,
    OS_HV: tl.constexpr,
    OS_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_V: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_nh = tl.program_id(1)
    n = pid_nh // HV_GROUPS
    i_hv0 = (pid_nh % HV_GROUPS) * BLOCK_HV

    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    # All heads in this CTA share one key head because RATIO >= BLOCK_HV.
    h = i_hv0 // RATIO

    # ---- q / k of the shared key head
    q_row = mixed_qkv_ptr + n * STRIDE_QKV_TOK + h * K
    k_row = mixed_qkv_ptr + n * STRIDE_QKV_TOK + Q_DIM + h * K
    if EVEN_K:
        q = tl.load(q_row + k_offs).to(tl.float32)
        kv = tl.load(k_row + k_offs).to(tl.float32)
    else:
        km = k_offs < K
        q = tl.load(q_row + k_offs, mask=km, other=0.0).to(tl.float32)
        kv = tl.load(k_row + k_offs, mask=km, other=0.0).to(tl.float32)

    # ---- per-sequence slot (shared by every head of this CTA)
    slot = tl.load(idx_ptr + n)
    valid = slot > 0
    slot_c = tl.minimum(tl.maximum(slot, 0), N_SLOTS - 1)

    # ---- normalization of q/k (epsilon inside the sqrt, as the reference does)
    if USE_L2NORM:
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        kv = kv / tl.sqrt(tl.sum(kv * kv) + 1e-6)
    q = q * scale

    for j in tl.static_range(BLOCK_HV):
        i_hv = i_hv0 + j

        # the state slab and the v slice are independent of the gate, so they are
        # issued before it; their latency then overlaps the gate maths instead of
        # stalling in front of the pred contraction
        v_base = (
            mixed_qkv_ptr + n * STRIDE_QKV_TOK + 2 * Q_DIM + i_hv * V + v_offs
        )
        if EVEN_V:
            v_val = tl.load(v_base).to(tl.float32)
        else:
            v_val = tl.load(v_base, mask=v_offs < V, other=0.0).to(tl.float32)

        s_ptrs = (
            state_ptr
            + slot_c * SS_SLOT
            + i_hv * SS_HV
            + v_offs[:, None] * SS_V
            + k_offs[None, :] * SS_K
        )
        if EVEN_V:
            if EVEN_K:
                S = tl.load(s_ptrs).to(tl.float32)
            else:
                S = tl.load(s_ptrs, mask=k_offs[None, :] < K, other=0.0).to(
                    tl.float32
                )
        else:
            vm2 = v_offs[:, None] < V
            if EVEN_K:
                S = tl.load(s_ptrs, mask=vm2, other=0.0).to(tl.float32)
            else:
                S = tl.load(
                    s_ptrs, mask=vm2 & (k_offs[None, :] < K), other=0.0
                ).to(tl.float32)

        # ---- fused gate for this value head
        a_val = tl.load(a_ptr + n * HV + i_hv).to(tl.float32)
        b_val = tl.load(b_ptr + n * HV + i_hv)
        beta = (
            tl.sigmoid(b_val.to(tl.float32))
            .to(b_ptr.dtype.element_ty)
            .to(tl.float32)
        )
        a_log = tl.load(a_log_ptr + i_hv).to(tl.float32)
        dt_b = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        x = a_val + dt_b
        e = tl.exp(tl.minimum(x, _SOFTPLUS_THRESHOLD))
        sp = tl.where(x > -15.0, tl.log(1.0 + e), e)
        sp = tl.where(x <= _SOFTPLUS_THRESHOLD, sp, x)
        decay = tl.exp(-tl.exp(a_log) * sp)

        S = S * decay
        pred = tl.sum(S * kv[None, :], axis=1)

        v_corr = (v_val - pred) * beta
        S = S + v_corr[:, None] * kv[None, :]

        o = tl.sum(S * q[None, :], axis=1)

        o_ptrs = out_ptr + n * OS_N + i_hv * OS_HV + v_offs * OS_V
        o_val = tl.where(valid, o, 0.0).to(out_ptr.dtype.element_ty)

        # Issue the state store first: it is the larger of the two writes and the
        # one the next call must read back, and steering it ahead of the small
        # out store measured 0.2-1.1% faster on seven of the ten timing shapes
        # (b512 520.6 vs 521.8 us, b128 136.33 vs 136.83, b32 39.79 vs 40.25,
        # h4hv8-b32 14.02 vs 14.18).
        s_val = S.to(state_ptr.dtype.element_ty)
        if EVEN_V:
            if EVEN_K:
                tl.store(s_ptrs, s_val, mask=valid)
            else:
                tl.store(s_ptrs, s_val, mask=(k_offs[None, :] < K) & valid)
        else:
            vm2 = v_offs[:, None] < V
            if EVEN_K:
                tl.store(s_ptrs, s_val, mask=vm2 & valid)
            else:
                tl.store(
                    s_ptrs, s_val, mask=vm2 & (k_offs[None, :] < K) & valid
                )

        if EVEN_V:
            tl.store(o_ptrs, o_val)
        else:
            tl.store(o_ptrs, o_val, mask=v_offs < V)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    B = mixed_qkv.shape[0]
    n_slots, HV, V, K = initial_state.shape
    q_dim = (mixed_qkv.shape[1] - HV * V) // 2
    H = q_dim // K
    ratio = HV // H

    BLOCK_K = max(16, triton.next_power_of_2(K))
    EVEN_K = BLOCK_K == K

    # BLOCK_V: narrow tiles keep per-thread ILP high, but below a few hundred
    # rows they start costing more than they save.  Target measurements (3
    # repeats each, spread < 0.2%) give BV2 for the single-sequence grid
    # (b1: 4.59 vs 4.95 us at BV4, 5.70 at BV8), BV4 for the small grids
    # (b8: 13.93 vs 15.04 at BV8; h4hv8-b32 13.89 vs 14.37) and BV8 for the
    # larger ones (b32 40.16 vs 40.15 tie, b64 72.21 vs 72.90, b128 136.43 vs
    # 137.43, b256 264.98 vs 267.34, b512 521.85 vs 526.42).  BV16/BV32 are
    # uniformly worse and are never selected.
    rows = B * HV
    if rows < 64:
        BLOCK_V = 2
    elif rows < 512:
        BLOCK_V = 4
    else:
        BLOCK_V = 8
    if BLOCK_V > V:
        BLOCK_V = max(1, triton.next_power_of_2(V))
    EVEN_V = V % BLOCK_V == 0

    # BLOCK_HV: only valid when the heads in one CTA share a key head
    # (ratio >= BLOCK_HV) and HV is divisible by it.  It halves the CTA count and
    # the per-key-head q/k + l2norm work; measured 4-10% on every shape with
    # >= 256 rows, and slower below that (b1: 6.05 vs 5.48 us at BH2).
    if ratio >= 2 and HV % 2 == 0 and rows >= 256:
        BLOCK_HV = 2
    else:
        BLOCK_HV = 1
    hv_groups = HV // BLOCK_HV

    nv = (V + BLOCK_V - 1) // BLOCK_V

    _gdn_packed_decode_kernel[(nv, B * hv_groups)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        scale,
        HV=HV,
        V=V,
        K=K,
        RATIO=ratio,
        Q_DIM=q_dim,
        HV_GROUPS=hv_groups,
        N_SLOTS=n_slots,
        STRIDE_QKV_TOK=mixed_qkv.stride(0),
        SS_SLOT=initial_state.stride(0),
        SS_HV=initial_state.stride(1),
        SS_V=initial_state.stride(2),
        SS_K=initial_state.stride(3),
        OS_N=out.stride(0),
        OS_HV=out.stride(2),
        OS_V=out.stride(3),
        BLOCK_V=BLOCK_V,
        BLOCK_HV=BLOCK_HV,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        EVEN_V=EVEN_V,
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=1,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
