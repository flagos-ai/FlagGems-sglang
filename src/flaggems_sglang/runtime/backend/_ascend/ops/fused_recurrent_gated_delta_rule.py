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

"""Gated delta-rule linear attention recurrence (fused_recurrent_gdn).

Sequential fp32 state-space scan.  One program owns the state tile
S[BLOCK_V, BLOCK_K] for one (b, hv, value-block) triple and keeps it resident
across the whole T loop.  The recurrence is independent per value row, so the
V-axis split is exact parallelism.  K is the contiguous tile axis.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    init_ptr,
    o_ptr,
    fin_ptr,
    scale,
    T,
    K,
    V,
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
    sib,
    sihv,
    siv,
    sik,
    sob,
    sot,
    soh,
    sov,
    sfb,
    sfhv,
    sfv,
    sfk,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    HAS_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    BETA_CHANNEL: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_b = tl.program_id(2)
    h = pid_hv // ratio

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    if HAS_INIT:
        S = tl.load(
            init_ptr
            + pid_b * sib
            + pid_hv * sihv
            + offs_v[:, None] * siv
            + offs_k[None, :] * sik,
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0,
        )
    else:
        S = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    q_base = q_ptr + pid_b * sqb + h * sqh
    k_base = k_ptr + pid_b * skb + h * skh
    v_base = v_ptr + pid_b * svb + pid_hv * svh
    g_base = g_ptr + pid_b * sgb + pid_hv * sgh
    b_base = beta_ptr + pid_b * sbb + pid_hv * sbh
    o_base = o_ptr + pid_b * sob + pid_hv * soh

    for t in range(0, T):
        eg = tl.exp(tl.load(g_base + t * sgt))

        kt = tl.load(
            k_base + t * skt + offs_k * skk, mask=mask_k, other=0.0
        ).to(tl.float32)
        qt = tl.load(
            q_base + t * sqt + offs_k * sqk, mask=mask_k, other=0.0
        ).to(tl.float32)

        if USE_L2NORM:
            qt = qt / tl.sqrt(tl.sum(qt * qt, axis=0) + 1e-6)
            kt = kt / tl.sqrt(tl.sum(kt * kt, axis=0) + 1e-6)
        qt = qt * scale

        S = S * eg

        pred = tl.sum(S * kt[None, :], axis=1)

        vt = tl.load(
            v_base + t * svt + offs_v * svv, mask=mask_v, other=0.0
        ).to(tl.float32)

        if BETA_CHANNEL:
            bt = tl.load(
                b_base + t * sbt + offs_v * sbv, mask=mask_v, other=0.0
            )
        else:
            bt = tl.load(b_base + t * sbt)
        c = (vt - pred) * bt

        S = S + c[:, None] * kt[None, :]

        ot = tl.sum(S * qt[None, :], axis=1)
        tl.store(o_base + t * sot + offs_v * sov, ot, mask=mask_v)

    if STORE_FINAL:
        tl.store(
            fin_ptr
            + pid_b * sfb
            + pid_hv * sfhv
            + offs_v[:, None] * sfv
            + offs_k[None, :] * sfk,
            S,
            mask=mask_v[:, None] & mask_k[None, :],
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
    V = v.shape[3]
    ratio = HV // H
    beta_channel = beta.dim() == v.dim()

    BLOCK_K = max(16, triton.next_power_of_2(K))
    # Per-program fp32 state-tile element budget.  8192 elements (32 KB per
    # fp32 full-tile temporary: S, S*kt, c[:,None]*kt[None,:], S*qt) is the
    # comfortable default and was worth +42% geo_mean over 4096 in round 2.
    # For the K=128 / V=128 Qwen3-Next family, BLOCK_V=64 (NV=2) still costs
    # twice its pure-state floor, so a target-side probe was run: at T=1 the
    # 16384 budget (BLOCK_V=128, NV=1) compiles, matches the shipped results,
    # and reaches 0.2539 ms on decode-b64 against 0.6151 ms at BLOCK_V=64 --
    # the pure-state access floor.  The T>1 shapes of that family still
    # overflow UB at 16384 (1851392 bits needed vs 1572864 available), so the
    # larger budget is gated on T == 1, where the loop body holds one step
    # only and both reductions are single-shot.  Because the launch key is
    # (BLOCK_K, BLOCK_V, ...), per-shape constants are legal: no kernel
    # recompilation happens for a given shape, and no shape reuses a launch
    # signature with a different budget.
    budget_elems = 16384 if (K == 128 and V == 128 and T == 1) else 8192
    BV = max(16, budget_elems // BLOCK_K)
    if BV > V:
        BV = V
    BLOCK_V = max(16, triton.next_power_of_2(BV))

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    has_init = initial_state is not None
    if output_final_state:
        fin = torch.empty((B, HV, V, K), dtype=torch.float32, device=v.device)
    else:
        fin = o
    ist = initial_state if has_init else o

    grid = (triton.cdiv(V, BLOCK_V), HV, B)
    _gdn_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        ist,
        o,
        fin,
        float(scale),
        T,
        K,
        V,
        ratio,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        beta.stride(3) if beta_channel else 0,
        ist.stride(0),
        ist.stride(1),
        ist.stride(2),
        ist.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        fin.stride(0),
        fin.stride(1),
        fin.stride(2),
        fin.stride(3),
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        HAS_INIT=has_init,
        STORE_FINAL=bool(output_final_state),
        BETA_CHANNEL=beta_channel,
    )
    return o, (fin if output_final_state else None)


__all__ = ["fused_recurrent_gated_delta_rule"]
