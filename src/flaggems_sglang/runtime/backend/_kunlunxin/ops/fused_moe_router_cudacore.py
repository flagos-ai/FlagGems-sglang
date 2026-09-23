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

"""Fused MoE router (Triton, two-kernel pipeline).

Reference semantics:
    logits = fp32(x) @ fp32(W).T            # [M, E]
    if softcap != 0: logits = tanh(logits / softcap) * softcap
    if bias is not None: logits = logits + bias
    probs  = softmax(logits, -1)
    v, ids = topk(logits, k, -1, largest=True, sorted=True)
    topk_weights = gather(probs, -1, ids)   # raw probabilities, no renormalization
    return topk_weights (fp32 [M,k]), ids (int32 [M,k])

Kernel 1 (rank-2 MMA) does the bf16 tensor-core matmul x @ W.T into a padded
fp32 logits scratch buffer of row width WSTRIDE; the expert axis is split over
several programs so the weight matrix is streamed with better occupancy.
Kernel 2 does bias/softcap, the row softmax and the top-k selection.  Two forms
are used: a rank-2 row-block form (one program per BLOCK_M rows, only max/sum
reductions) and a rank-1 per-row form for padded expert counts or softcapped
inputs.

Target back end constraints observed on this device: compares/selects are
legalized only on rank-1 tiles (a predicate broadcast over a rank-2 tile aborts
the pipeline), tensor/tensor division only outside the MMA output layout, the
min reduction returns wrong values on mostly-padded rows (a max reduction is
used instead), `tl.exp` of very large arguments is disproportionately slow (the
selection mask is therefore built with min/mul/subtract instead of exp), and
int32 ids must stay native integers throughout.
"""

import torch
import triton
import triton.language as tl

_WSTRIDE = 256  # scratch row width (also the fast-path expert count)


@triton.jit
def _router_logits_kernel(
    x_ptr,
    w_ptr,
    l_ptr,
    M,
    K,
    E,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WSTRIDE: tl.constexpr,
    NEED_M: tl.constexpr,
    NEED_K: tl.constexpr,
    NEED_E: tl.constexpr,
):
    """l[m, e] = sum_k x[m, k] * W[e, k] (bf16 tensor-core matmul, fp32 acc)."""
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + offs_k
        if NEED_M and NEED_K:
            a = tl.load(
                x_ptr + offs_m[:, None] * K + ks[None, :],
                mask=(offs_m[:, None] < M) & (ks[None, :] < K),
                other=0.0,
            )
        elif NEED_M:
            a = tl.load(
                x_ptr + offs_m[:, None] * K + ks[None, :],
                mask=offs_m[:, None] < M,
                other=0.0,
            )
        elif NEED_K:
            a = tl.load(
                x_ptr + offs_m[:, None] * K + ks[None, :],
                mask=ks[None, :] < K,
                other=0.0,
            )
        else:
            a = tl.load(x_ptr + offs_m[:, None] * K + ks[None, :])
        if NEED_E and NEED_K:
            w = tl.load(
                w_ptr + offs_e[:, None] * K + ks[None, :],
                mask=(offs_e[:, None] < E) & (ks[None, :] < K),
                other=0.0,
            )
        elif NEED_E:
            w = tl.load(
                w_ptr + offs_e[:, None] * K + ks[None, :],
                mask=offs_e[:, None] < E,
                other=0.0,
            )
        elif NEED_K:
            w = tl.load(
                w_ptr + offs_e[:, None] * K + ks[None, :],
                mask=ks[None, :] < K,
                other=0.0,
            )
        else:
            w = tl.load(w_ptr + offs_e[:, None] * K + ks[None, :])
        acc = tl.dot(a, tl.trans(w), acc)

    tl.store(l_ptr + offs_m[:, None] * WSTRIDE + offs_e[None, :], acc)


@triton.jit
def _router_topk_rows_kernel(
    l_ptr,
    bias_ptr,
    ow_ptr,
    oi_ptr,
    M,
    NTOPK,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WSTRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Row softmax + top-k over a block of token rows (rank-2 tile)."""
    BIG: tl.constexpr = 1.0e30
    SUME: tl.constexpr = WSTRIDE * (WSTRIDE - 1) // 2
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, WSTRIDE)
    offs_e_f = offs_e.to(tl.float32)

    x = tl.load(l_ptr + offs_m[:, None] * WSTRIDE + offs_e[None, :])
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_e)
        x = x + b[None, :]

    row_max = tl.max(x, axis=1)
    denom = tl.sum(tl.exp(x - row_max[:, None]), axis=1)

    for j in tl.static_range(TOPK):
        sel = tl.max(x, axis=1)
        # 0 exactly at the row maximum, 1 everywhere else (no exp, no compare)
        pen = tl.minimum((sel[:, None] - x) * BIG, 1.0)
        # sum of the expert index over the one-hot of the selected lane
        idx = (SUME - tl.sum(pen * offs_e_f[None, :], axis=1)).to(tl.int32)
        store_mask = offs_m < M
        tl.store(
            ow_ptr + offs_m * NTOPK + j,
            tl.exp(sel - row_max) / denom,
            mask=store_mask,
        )
        tl.store(oi_ptr + offs_m * NTOPK + j, idx, mask=store_mask)
        x = x + (pen - 1.0) * BIG


@triton.jit
def _router_topk_row_kernel(
    l_ptr,
    bias_ptr,
    ow_ptr,
    oi_ptr,
    E,
    NTOPK,
    inv_softcap: tl.constexpr,
    softcap: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    WSTRIDE: tl.constexpr,
    E_EXACT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    """Bias/softcap, row softmax and top-k selection for a single token row."""
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    x = tl.load(l_ptr + pid * WSTRIDE + offs_e)

    # tanh(z) = 1 - 2 / (exp(2z) + 1): stable for both signs, no comparisons.
    if HAS_SOFTCAP:
        z = x * inv_softcap
        x = (1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)) * softcap
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_e, mask=offs_e < E, other=0.0)
        x = x + b
    if not E_EXACT:
        # Padded lanes must not reach the denominator nor the selection.
        x = tl.where(offs_e < E, x, -1.0e30)

    row_max = tl.max(x)
    denom = tl.sum(tl.exp(x - row_max))

    for j in tl.static_range(TOPK):
        sel = tl.max(x)
        hit = x == sel
        # Highest (1<<20) - lane among the maxima == lowest selected lane index.
        idx = (1 << 20) - tl.max(tl.where(hit, (1 << 20) - offs_e, 0))
        tl.store(ow_ptr + pid * NTOPK + j, tl.exp(sel - row_max) / denom)
        tl.store(oi_ptr + pid * NTOPK + j, idx)
        x = tl.where(offs_e == idx, -1.0e30, x)


def _logits_config(m):
    """(BLOCK_M, max BLOCK_E, BLOCK_K, num_warps) for the logits kernel."""
    if m <= 16:
        return 16, 32, 512, 2
    if m <= 128:
        return 64, 32, 512, 8
    if m <= 512:
        return 128, 128, 256, 8
    return 256, 128, 256, 16


def fused_moe_router_cudacore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    x = x.contiguous()
    router_weight = router_weight.contiguous()
    if correction_bias is not None:
        correction_bias = correction_bias.contiguous()
    k_top = int(topk)
    softcap = (
        0.0
        if moe_softcapping is None
        else (0.0 if moe_softcapping is None else float(moe_softcapping))
    )
    has_softcap = softcap != 0.0
    has_bias = correction_bias is not None

    M = x.shape[0]
    K = x.shape[1]
    E = router_weight.shape[0]

    BLOCK_M, max_be, cfg_bk, num_warps = _logits_config(M)
    BLOCK_E = max(16, min(max_be, triton.next_power_of_2(E)))
    BLOCK_K = min(cfg_bk, max(16, triton.next_power_of_2(K)))
    M_PAD = triton.cdiv(M, BLOCK_M) * BLOCK_M
    E_BLOCKS = triton.cdiv(E, BLOCK_E)
    WSTRIDE = max(_WSTRIDE, BLOCK_E, triton.next_power_of_2(E))

    logits = torch.empty(
        (M_PAD, WSTRIDE), dtype=torch.float32, device=x.device
    )
    topk_weights = torch.empty(
        (M, k_top), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((M, k_top), dtype=torch.int32, device=x.device)
    bias_ptr = correction_bias if has_bias else topk_weights

    _router_logits_kernel[(M_PAD // BLOCK_M, E_BLOCKS)](
        x,
        router_weight,
        logits,
        M,
        K,
        E,
        BLOCK_M=BLOCK_M,
        BLOCK_E=BLOCK_E,
        BLOCK_K=BLOCK_K,
        WSTRIDE=WSTRIDE,
        NEED_M=(M_PAD != M),
        NEED_K=(K % BLOCK_K != 0),
        NEED_E=(E_BLOCKS * BLOCK_E != E),
        num_warps=num_warps,
        num_stages=3 if BLOCK_M >= 256 else 2,
    )

    if E == WSTRIDE and not has_softcap:
        rows = 16 if M_PAD < 256 else 32
        _router_topk_rows_kernel[(M_PAD // rows,)](
            logits,
            bias_ptr,
            topk_weights,
            topk_ids,
            M,
            k_top,
            TOPK=k_top,
            BLOCK_M=rows,
            WSTRIDE=WSTRIDE,
            HAS_BIAS=has_bias,
            num_warps=2,
        )
    else:
        _router_topk_row_kernel[(M,)](
            logits,
            bias_ptr,
            topk_weights,
            topk_ids,
            E,
            k_top,
            inv_softcap=(1.0 / softcap if has_softcap else 1.0),
            softcap=softcap,
            TOPK=k_top,
            BLOCK_E=WSTRIDE,
            WSTRIDE=WSTRIDE,
            E_EXACT=(E == WSTRIDE),
            HAS_BIAS=has_bias,
            HAS_SOFTCAP=has_softcap,
            num_warps=2,
        )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_cudacore"]
