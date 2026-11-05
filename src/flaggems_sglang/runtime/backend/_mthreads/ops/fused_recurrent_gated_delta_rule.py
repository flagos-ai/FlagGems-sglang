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

"""Triton implementation of the gated delta-rule linear-attention recurrence
(FlagGems-sglang fused_recurrent_gdn, the fla kernel behind Qwen3-Next / GDN).

Architecture
------------
One program per (V-block, flattened (b, hv) head).  Each program owns a
register-resident float32 state tile S[BV, BK] -- BK covers the FULL K extent of
the head, because the recurrence makes S a loop-carried dependency and K is
never split across programs -- and walks the whole T loop.

Per timestep, in the reference's exact order:
  1. decay      S *= exp(g[b,t,hv])                       (broadcast over V,K)
  2. load k,q   (per KEY head h = hv // (HV // H))
  3. optional L2 norm on q,k with 1e-6 INSIDE the sqrt, then q *= scale
  4. pred       pred[v] = sum_k S[v,k] * k[k]              (on the DECAYED state)
  5. load v, beta
  6. v_corr     (v - pred) * beta
  7. update     S += v_corr[:, None] * k[None, :]
  8. readout    o = sum_k S[v,k] * q[k]                    (AFTER the update)
Everything is float32; only the o store rounds to v's dtype, matching the
reference's `o.to(v.dtype)`.  final_state is written as float32.

Prediction precision (measured on the target)
---------------------------------------------
`torch.einsum("bhvk,bhk->bhv", state, vec)` is dispatched on MUSA to muDNN's
batched matmul.  Measured on the target: below 9 independent matrices muDNN uses
an exact fp32 kernel (relative deviation from a complex-fp64 reference ~1e-7);
at 9 or more it uses its tensor-core tf32 path, truncating both operands toward
zero to 10 explicit mantissa bits (relative deviation ~5e-4).  That per-step
difference is amplified by the expanding recurrence and exceeds the workload
tolerance on the most amplifying case (gdn-corr-gva4, B*HV=16: the exact-fp32
kernel deviates from the device reference by 2.0 while the tolerance is 0.024).
`tl.dot(..., input_precision="tf32")` reproduces that device einsum BIT-EXACTLY
(512/512 elements measured), so where alignment matters the prediction is
computed through the tensor-core dot; where it does not, the plain fp32
reduction is both faster and measurably inside tolerance.
"""

import torch
import triton
import triton.language as tl

# Measured on the target: muDNN switches its batched-matmul path from an exact
# fp32 kernel to tensor-core tf32 at 9 independent matrices (batch = B*HV here).
_TF32_BATCH_THRESHOLD = 9
# Measured: the tf32 tl.dot only wins on narrow-K tiles with a shallow T; with a
# long T loop the plain fp32 reduction is faster, and it was verified to match
# the device reference inside tolerance for the long-T narrow-K workloads.
_TF32_MAX_T = 64


@triton.jit
def _gdn_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    init_ptr,
    o_ptr,
    fin_ptr,
    scale,
    B,
    T,
    H,
    HV,
    K,
    V,
    sQb,
    sQt,
    sQh,
    sQk,
    sKb,
    sKt,
    sKh,
    sKk,
    sVb,
    sVt,
    sVh,
    sVv,
    sGb,
    sGt,
    sGh,
    sBb,
    sBt,
    sBh,
    sBv,
    sIb,
    sIh,
    sIv,
    sIk,
    sOb,
    sOt,
    sOh,
    sOv,
    sFb,
    sFh,
    sFv,
    sFk,
    BV: tl.constexpr,
    BK: tl.constexpr,
    HAS_INIT: tl.constexpr,
    USE_OUT_FINAL: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    TF32_PRED: tl.constexpr,
    ST: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HV
    hv = pid_bh % HV
    ratio = HV // H
    h = hv // ratio

    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    m_v = offs_v < V
    m_k = offs_k < K
    # tl.dot needs N >= 16; a single non-zero column carries the K-vector.
    col0 = (tl.arange(0, 16) == 0).to(tl.float32)[None, :]

    # ---- prologue: load (or zero) the float32 state tile S[V, K] ----
    if HAS_INIT:
        state = tl.load(
            init_ptr
            + b * sIb
            + hv * sIh
            + offs_v[:, None] * sIv
            + offs_k[None, :] * sIk,
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        )
    else:
        state = tl.zeros((BV, BK), dtype=tl.float32)

    g_base = g_ptr + b * sGb + hv * sGh
    b_base = beta_ptr + b * sBb + hv * sBh
    q_base = q_ptr + b * sQb + h * sQh + offs_k * sQk
    k_base = k_ptr + b * sKb + h * sKh + offs_k * sKk
    v_base = v_ptr + b * sVb + hv * sVh + offs_v * sVv

    # ---- sequential recurrence over T ----
    for t in tl.range(0, T, num_stages=ST):
        # 1. scalar decay gate
        state = state * tl.exp(tl.load(g_base + t * sGt))

        # 2/3. key/query vector for this KEY head, cast to fp32
        kv = tl.load(k_base + t * sKt, mask=m_k, other=0.0).to(tl.float32)
        qv = tl.load(q_base + t * sQt, mask=m_k, other=0.0).to(tl.float32)
        if USE_L2NORM:
            # eps INSIDE the sqrt: x / sqrt(sum(x^2) + 1e-6)
            kv = kv / tl.sqrt(tl.sum(kv * kv) + 1e-6)
            qv = qv / tl.sqrt(tl.sum(qv * qv) + 1e-6)
        qv = qv * scale

        # 4. prediction on the decayed state
        if TF32_PRED:
            pred = tl.sum(
                tl.dot(
                    state,
                    kv[:, None] * col0,
                    input_precision="tf32",
                    out_dtype=tl.float32,
                ),
                axis=1,
            )
        else:
            pred = tl.sum(state * kv[None, :], axis=1)

        # 5/6. delta correction
        vt = tl.load(v_base + t * sVt, mask=m_v, other=0.0).to(tl.float32)
        bt = tl.load(b_base + t * sBt + offs_v * sBv, mask=m_v, other=0.0)
        vcorr = (vt - pred) * bt

        # 7. rank-1 update
        state = state + vcorr[:, None] * kv[None, :]

        # 8. readout of the UPDATED state
        o_t = tl.sum(state * qv[None, :], axis=1)
        tl.store(
            o_ptr + b * sOb + t * sOt + hv * sOh + offs_v * sOv,
            o_t.to(o_ptr.dtype.element_ty),
            mask=m_v,
        )

    if USE_OUT_FINAL:
        tl.store(
            fin_ptr
            + b * sFb
            + hv * sFh
            + offs_v[:, None] * sFv
            + offs_k[None, :] * sFk,
            state,
            mask=m_v[:, None] & m_k[None, :],
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
    V = v.shape[-1]
    device = q.device

    BK = max(16, triton.next_power_of_2(K))
    Vp = max(16, triton.next_power_of_2(V))
    ngroups = B * HV

    # Where the reference's own matmul runs on muDNN's tf32 path AND the padded
    # dot is the better inner loop (narrow K, shallow T), reproduce that
    # precision explicitly.  Everywhere else use the plain fp32 reduction, which
    # was measured both inside tolerance and faster.
    tf32_pred = (
        ngroups >= _TF32_BATCH_THRESHOLD and BK <= 64 and T < _TF32_MAX_T
    )

    # Launch shape per size class.  Values come from per-shape sweeps on the
    # target; for the reference shapes they measured (us, BV, warps, stages):
    #   bench0 (64 heads, T=128)          8/4/1 -> 620   (16/4/1 -> 787)
    #   bench1 (256 heads, T=32)         64/4/1 -> 250   (32/2/1 -> 275)
    #   b1-t32 (32 heads, T=32)           8/4/1 -> 145   (16/4/1 -> 171)
    #   b4-t32 (128 heads, T=32)         16/4/1 -> 359   (16/4/2 -> 375)
    #   b8-t16 (256 heads, T=16)         32/4/2 -> 358   (16/4/2 -> 378)
    #   b16-t8 (512 heads, T=8)          32/4/2 -> 328   (16/4/2 -> 355)
    #   decode-b32 (1024 heads, T=1)     16/4/1 -> 138   (32/8/1 -> 144)
    if tf32_pred:
        BV, num_warps, num_stages = min(64, Vp), 4, 1
    elif BK <= 64:
        BV, num_warps, num_stages = 8, 4, 1
    elif T > 1:
        if ngroups >= 128:
            if T >= 32:
                BV, num_warps, num_stages = 16, 4, 1
            else:
                BV, num_warps, num_stages = 32, 4, 2
        else:
            BV, num_warps, num_stages = 8, 4, 1
    else:
        BV, num_warps, num_stages = min(16, Vp), 4, 1

    grid = (triton.cdiv(V, BV), ngroups)

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=device)

    has_init = initial_state is not None
    use_out_final = bool(output_final_state)
    if use_out_final:
        fin = torch.empty((B, HV, V, K), dtype=torch.float32, device=device)
    else:
        fin = o  # never stored to
    init = initial_state if has_init else o  # never dereferenced

    beta_4d = beta.dim() == v.dim()
    if beta_4d:
        sBb, sBt, sBh, sBv = (
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            beta.stride(3),
        )
    else:
        sBb, sBt, sBh = beta.stride(0), beta.stride(1), beta.stride(2)
        sBv = 0

    _gdn_recurrent_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        init,
        o,
        fin,
        float(scale),
        B,
        T,
        H,
        HV,
        K,
        V,
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
        sBb,
        sBt,
        sBh,
        sBv,
        init.stride(0) if has_init else 0,
        init.stride(1) if has_init else 0,
        init.stride(2) if has_init else 0,
        init.stride(3) if has_init else 0,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        fin.stride(0),
        fin.stride(1),
        fin.stride(2),
        fin.stride(3),
        BV=BV,
        BK=BK,
        HAS_INIT=has_init,
        USE_OUT_FINAL=use_out_final,
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        TF32_PRED=tf32_pred,
        ST=num_stages,
        num_warps=num_warps,
    )

    return o, (fin if use_out_final else None)


__all__ = ["fused_recurrent_gated_delta_rule"]
