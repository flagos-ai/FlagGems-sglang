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

"""Single-token fused recurrent gated delta rule decode (packed QKV) — Triton.

Reference semantics: vllm/model_executor/layers/fla/ops/fused_recurrent.py
::fused_recurrent_gated_delta_rule_packed_decode.

Design
------
grid = (NV, HV, B) with NV = cdiv(V, BLOCK_V).  Program (i_v, i_hv, i_n) owns
the state rows [i_v*BLOCK_V, (i_v+1)*BLOCK_V) of initial_state[slot, i_hv] and
the matching slice of out[i_n, 0, i_hv, :].  The whole [BLOCK_V, BLOCK_K] state
tile is held in registers in fp32 for a single read + single write pass;
BLOCK_K = next_pow2(K) so there is no K loop on the common shapes.

For V >= 128 the default is BLOCK_V = 128 (NV = 1) with num_warps = 2: each
program then owns one contiguous [V, K] state block per (slot, head), which is
the flat contiguous streaming pattern a copy kernel achieves, and both K-axis
contractions stay cheap because a full K row lives inside one warp's registers.
Grids that are too small for that tile (fewer than 512 programs, i.e. the light
batches) are latency/wave limited instead, and measure better with a smaller
tile and more programs.

Everything (state tile, q/k/v, reductions, rank-1 update) runs in fp32; the only
roundings are the two stores plus the deliberate beta round-trip through b's
dtype, exactly as the reference does.
"""

import triton
import triton.language as tl

_SOFTPLUS_THRESHOLD = 20.0


@triton.jit
def _gdn_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    stride_qkv_tok,
    q_dim,
    hv_dim,
    v_dim,
    k_dim,
    head_ratio,
    EXACT_V: tl.constexpr,
    EXACT_K: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)

    slot = tl.load(idx_ptr + i_n)
    valid = slot > 0
    slot_c = tl.where(valid, slot, 0)

    rows = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
    cols = tl.arange(0, BLOCK_K)
    # rmask/cmask are provably all-true for the shapes in this workload set
    # (V divisible by BLOCK_V, K a power of two).  Predicated loads/stores that
    # can never mask anything still cost vectorization and issue slots, so the
    # masks are only emitted when they can actually mask something.
    rmask = rows < v_dim
    cmask = cols < k_dim
    smask = rmask[:, None] & cmask[None, :]
    need_r = not EXACT_V
    need_c = not EXACT_K

    # grouped-value head mapping: value head i_hv reads q/k head i_hv // ratio
    ih = i_hv // head_ratio

    qk_base = mixed_qkv_ptr + i_n * stride_qkv_tok
    qk_off = ih * k_dim + cols
    if need_c:
        qv = tl.load(qk_base + qk_off, mask=cmask, other=0.0).to(tl.float32)
        kv = tl.load(qk_base + q_dim + qk_off, mask=cmask, other=0.0).to(
            tl.float32
        )
    else:
        qv = tl.load(qk_base + qk_off).to(tl.float32)
        kv = tl.load(qk_base + q_dim + qk_off).to(tl.float32)

    if USE_L2NORM:
        qv = qv / tl.sqrt(tl.sum(qv * qv, axis=0) + 1e-6)
        kv = kv / tl.sqrt(tl.sum(kv * kv, axis=0) + 1e-6)
    qv = qv * scale

    # fused gating (fp32)
    a_val = tl.load(a_ptr + i_n * hv_dim + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * hv_dim + i_hv).to(tl.float32)
    a_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dtb = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    x = a_val + dtb
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(a_log) * sp
    decay = tl.exp(g)
    # deliberate round-trip through b's element type, as in the reference kernel
    beta = tl.sigmoid(b_val).to(b_ptr.dtype.element_ty).to(tl.float32)

    s_base = (
        state_ptr + slot_c * (hv_dim * v_dim * k_dim) + i_hv * (v_dim * k_dim)
    )
    s_ptrs = s_base + rows[:, None] * k_dim + cols[None, :]

    if need_r or need_c:
        state = tl.load(s_ptrs, mask=smask, other=0.0).to(tl.float32) * decay
    else:
        state = tl.load(s_ptrs).to(tl.float32) * decay

    v_ptrs = qk_base + 2 * q_dim + i_hv * v_dim + rows
    if need_r:
        vv = tl.load(v_ptrs, mask=rmask, other=0.0).to(tl.float32)
    else:
        vv = tl.load(v_ptrs).to(tl.float32)

    pred = tl.sum(state * kv[None, :], axis=1)
    v_corr = (vv - pred) * beta
    state = state + v_corr[:, None] * kv[None, :]

    outv = tl.sum(state * qv[None, :], axis=1)

    o_ptrs = out_ptr + (i_n * hv_dim + i_hv) * v_dim + rows
    out_store = tl.where(valid, outv, 0.0).to(out_ptr.dtype.element_ty)
    if need_r:
        tl.store(o_ptrs, out_store, mask=rmask)
    else:
        tl.store(o_ptrs, out_store)
    if need_r or need_c:
        tl.store(
            s_ptrs, state.to(state_ptr.dtype.element_ty), mask=smask & valid
        )
    else:
        tl.store(
            s_ptrs,
            state.to(state_ptr.dtype.element_ty),
            mask=(rows[:, None] >= 0) & valid,
        )


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _pick_config(v_dim, batch, hv, elem_size):
    """Per-shape (BLOCK_V, num_warps) from the measured target sweeps.

    All sweeps were run on BW (80 CUs) with best-of-N CUDA-graph timing, which
    removes the measured ~8 us do_bench launch/measurement floor and is a much
    lower-variance signal for the light shapes.

    * Default (program count HV*B >= 1024): BLOCK_V = 64 with num_warps = 1.
      A single warp needs no cross-warp barriers for the two K-axis reductions,
      and the duplicate q/k traffic of the split (NV = 2) is negligible next to
      the state traffic.  Measured 1198 GB/s at b512, 1165 at h4hv8-b512, 1161
      at b128, 992 at h4hv8-b128 and 986 at b32, all above the corresponding
      BLOCK_V=128/num_warps=2 numbers (1195 / 1143 / 1143 / 961 / 975).
    * HV*B < 1024: only 32-256 programs exist and the grid is wave/latency
      limited, so a quarter-size tile with four warps is materially better
      (b1: 0.00789 vs 0.01287 ms; b8: 0.02245 vs 0.02348 ms; h4hv8-b32:
      0.02292 vs 0.02332 ms).
    * Four-byte inputs double the state bytes, so the row block is capped at 64
      with num_warps = 2, which keeps the fp32 register footprint in the same
      range as the 2-byte shapes and measured best there as well.
    """
    bv = min(128, _next_pow2(v_dim))
    if elem_size >= 4:
        return min(64, bv), 2
    if bv < 128:
        return bv, 4
    if hv * batch >= 1024:
        return 64, 1
    return 64, 4


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
    HV, V, K = initial_state.shape[-3:]

    qk_dim = mixed_qkv.shape[1] - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K
    ratio = HV // H

    block_v, num_warps = _pick_config(V, B, HV, mixed_qkv.element_size())
    nv = -(-V // block_v)

    _gdn_decode_kernel[(nv, HV, B)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0),
        q_dim,
        HV,
        V,
        K,
        ratio,
        EXACT_V=(V % block_v == 0),
        EXACT_K=(_next_pow2(K) == K),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        BLOCK_V=block_v,
        BLOCK_K=_next_pow2(K),
        num_warps=num_warps,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
