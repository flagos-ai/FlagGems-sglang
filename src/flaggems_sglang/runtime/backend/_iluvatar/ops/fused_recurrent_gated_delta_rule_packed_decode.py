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

"""Fused recurrent gated-delta-rule packed decode (single token) in Triton.

Semantics transcribed from
vllm/model_executor/layers/fla/ops/fused_recurrent.py
::fused_recurrent_gated_delta_rule_packed_decode_kernel.

One fused kernel covers the whole operator: gate computation, optional q/k
L2-normalization, the decay, the delta-rule correction, the rank-1 state
update, the read-out and the packed-slot in-place state write-back.

Performance notes (Iluvatar BI-V150, flagtree 0.6.0):
  * Each program owns one (sequence, value head) pair and walks NSUB narrow V
    strips (BLOCK_V rows each) sequentially.  The gate transcendentals and the
    q/k row loads plus their L2 norms are hoisted above that loop, so they run
    once per (n, hv) instead of once per V block, and the redundant re-read of
    the q/k rows by every V block disappears.
  * BLOCK_V is deliberately small: the live fp32 state tile dominates the
    register budget, so narrow strips raise occupancy (measured ~58 registers
    per thread instead of ~127 at BLOCK_V=32), which is what the memory
    latency hiding of this streaming kernel needs.
  * num_warps=1: a single warp per CTA keeps the K-axis row reductions
    warp-local (shuffle only, no cross-warp shared-memory staging).
  * K/V/H/HV and the last-axis strides of `initial_state`/`out` are compile
    time constants so the bounds masks fold away and the axis analysis can
    emit wide vector accesses for the state tile.  A non-dense stride still
    compiles correctly because the value itself is passed in.
  * The two K-axis row reductions reshape K into (K // CH, CH) and reduce the
    innermost axis first, which shortens the cross-lane butterfly.
"""

import triton
import triton.language as tl


@triton.jit
def _gdn_packed_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    initial_state_ptr,
    out_ptr,
    ssm_idx_ptr,
    scale,
    stride_mq_tok,
    stride_a_tok,
    stride_b_tok,
    stride_state_slot,
    stride_state_head,
    stride_out_tok,
    stride_out_head,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_out_v: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NSUB: tl.constexpr,
    CH: tl.constexpr,
    NEED_VMASK: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_vg = tl.program_id(0)
    i_hp = tl.program_id(1)
    i_n = i_hp // HV
    i_hv = i_hp % HV
    i_h = i_hv // (HV // H)

    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K

    # ---- fused gate (float32), computed once per (sequence, value head) --
    a_val = tl.load(a_ptr + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * stride_b_tok + i_hv)
    a_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    x = a_val + dt_b
    sp = tl.where(
        x <= SOFTPLUS_THRESHOLD,
        tl.log(1.0 + tl.exp(tl.minimum(x, SOFTPLUS_THRESHOLD))),
        x,
    )
    decay = tl.exp(-tl.exp(a_log) * sp)
    # beta is rounded through b's dtype, exactly as the reference does.
    beta = tl.sigmoid(b_val.to(tl.float32)).to(b_val.dtype).to(tl.float32)

    # ---- q / k rows (also hoisted) --------------------------------------
    row = mixed_qkv_ptr + i_n * stride_mq_tok
    q_vec = tl.load(row + i_h * K + offs_k, mask=kmask, other=0.0).to(
        tl.float32
    )
    k_vec = tl.load(
        row + (H * K) + i_h * K + offs_k, mask=kmask, other=0.0
    ).to(tl.float32)
    if USE_L2NORM:
        q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
        k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
    q_vec = q_vec * scale

    q3 = tl.reshape(q_vec, (BLOCK_K // CH, CH))
    k3 = tl.reshape(k_vec, (BLOCK_K // CH, CH))

    # ---- paged state slot -----------------------------------------------
    slot = tl.load(ssm_idx_ptr + i_n).to(tl.int32)
    valid = slot > 0
    slot_eff = tl.where(valid, slot, 0)
    sbase = (
        initial_state_ptr
        + slot_eff * stride_state_slot
        + i_hv * stride_state_head
    )
    obase = out_ptr + i_n * stride_out_tok + i_hv * stride_out_head

    # tl.range with software pipelining: the next strip's state tile is loaded
    # while the current one is being reduced, which the profiled 42% memory
    # dependency stall on an unrolled strip loop asks for.
    for s in tl.range(NSUB, num_stages=2):
        offs_v = (i_vg * NSUB + s) * BLOCK_V + tl.arange(0, BLOCK_V)
        if NEED_VMASK:
            vm = offs_v < V
            smask = vm[:, None] & kmask[None, :] & valid
            v_vec = tl.load(
                row + (2 * H * K) + i_hv * V + offs_v, mask=vm, other=0.0
            ).to(tl.float32)
        else:
            smask = kmask[None, :] & valid
            v_vec = tl.load(row + (2 * H * K) + i_hv * V + offs_v).to(
                tl.float32
            )

        ptrs = (
            sbase
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        S = tl.load(ptrs, mask=smask, other=0.0).to(tl.float32)

        # ---- decay, delta-rule correction, rank-1 update, read-out ------
        S3 = tl.reshape(S, (BLOCK_V, BLOCK_K // CH, CH))
        Sd3 = S3 * decay
        pred = tl.sum(tl.sum(Sd3 * k3[None, :, :], axis=2), axis=1)
        corr = (v_vec - pred) * beta
        S23 = Sd3 + corr[:, None, None] * k3[None, :, :]
        acc = tl.sum(tl.sum(S23 * q3[None, :, :], axis=2), axis=1)

        out_val = tl.where(valid, acc, 0.0).to(out_ptr.dtype.element_ty)
        if NEED_VMASK:
            tl.store(obase + offs_v * stride_out_v, out_val, mask=offs_v < V)
        else:
            tl.store(obase + offs_v * stride_out_v, out_val)
        tl.store(
            ptrs,
            tl.reshape(S23, (BLOCK_V, BLOCK_K)).to(
                initial_state_ptr.dtype.element_ty
            ),
            mask=smask,
        )


def _nsub_for(nblk, pairs):
    """Pick how many BLOCK_V strips each program walks.

    The gate transcendentals, the q/k rows and their L2 norms are hoisted out
    of the strip loop, so a larger strip count amortises them over more state
    bytes; but the CTA count must stay large enough to fill the machine.
    Measured on the target, every workload is within 1% of its own optimum for
    any grid in [1024, 16384] CTAs, so the largest divisor of `nblk` that lands
    in that window is taken, falling back to the largest divisor that still
    leaves 256 CTAs for tiny batches.  Dividing `nblk` exactly also lets the V
    guard fold away whenever V is a multiple of BLOCK_V * NSUB.
    """
    pairs = max(1, int(pairs))
    nblk = max(1, int(nblk))
    if nblk * pairs < 256:
        return 1
    best = 0
    for d in range(1, nblk + 1):
        if nblk % d:
            continue
        ctas = (nblk // d) * pairs
        if 1024 <= ctas <= 16384:
            best = d
    if best:
        return best
    for d in range(nblk, 0, -1):
        if nblk % d == 0 and (nblk // d) * pairs >= 256:
            return d
    return 1


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

    out_v = out.view(B, HV, V)
    V_i = int(V)
    K_i = int(K)

    BLOCK_K = max(triton.next_power_of_2(K_i), 16)
    BLOCK_V = min(8, max(triton.next_power_of_2(V_i), 1))
    nblk = triton.cdiv(V_i, BLOCK_V)
    nsub = _nsub_for(nblk, B * HV)
    grid = (triton.cdiv(V_i, BLOCK_V * nsub), B * HV)

    ch = 1
    for cand in (16, 8, 4, 2):
        if BLOCK_K % cand == 0:
            ch = cand
            break

    _gdn_packed_decode_kernel[grid](
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
        a.stride(0),
        b.stride(0),
        initial_state.stride(0),
        initial_state.stride(1),
        out_v.stride(0),
        out_v.stride(1),
        stride_state_v=initial_state.stride(2),
        stride_state_k=initial_state.stride(3),
        stride_out_v=out_v.stride(2),
        K=K_i,
        V=V_i,
        H=H,
        HV=HV,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        NSUB=nsub,
        CH=ch,
        NEED_VMASK=(V_i % (BLOCK_V * nsub) != 0),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=1,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
