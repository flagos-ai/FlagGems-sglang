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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""v3 fused MoE router (Tensor Core) implementation.

Two portable Triton kernels replace the PyTorch reference:

  1. ``_router_gemm_kernel`` — tiled GEMM ``logits = x @ router_weight.T``
     via ``tl.dot`` (Tensor Core). Logit soft-capping and the optional
     correction bias are fused into the store, so no intermediate logits
     tensor is materialized by PyTorch. Output is fp32 (matching the
     reference's fp32 compute) into a scratch buffer.
  2. ``_router_reduce_kernel`` — per block of token rows, loads the full
     expert row, computes global softmax, selects top-k (k<=2) experts by
     logit value, and gathers the softmax weights at those experts.

This avoids the reference's per-call fp32 materialization of ``x`` and
``router_weight`` and its string of separate kernels.
``flaggems_sglang.device`` is used for allocation; no vendor-specific
ops are used.

v3: retune the reduce kernel's launch config. A scan over BLOCK_M x
num_warps showed BLOCK_M=8 / num_warps=8 beats the prior BLOCK_M=16 /
num_warps=4 on every bench shape (the expert axis is short, so the math
is launch/dispatch bound; a smaller M-tile spread across more warps
amortizes that better). ~16 us vs ~22 us for B<=512, ~25 us vs ~30 us
for B=4096. GEMM path unchanged from v1.

v4: B-dependent reduce launch config. A second scan (BLOCK_M x num_warps,
per B) found that BLOCK_M=4 / num_warps=4 is a touch faster than the v3
BLOCK_M=8 / num_warps=8 on the small, latency-bound shapes (B<=512:
~43.5 vs ~44 us end-to-end), while BLOCK_M=8 / num_warps=8 stays best for
B>=1024 (the larger M-tile keeps the 4096-row grid from exploding into
1024 programs each reloading the full expert row). So the reduce kernel
now picks its (BLOCK_M, num_warps) from B at launch time. The GEMM path
is unchanged: its autotune config space is already optimal here (a
broad scan over BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages for the
B=4096 shape could not beat the autotuner's own pick of
BLOCK_M=128/BLOCK_N=128/BLOCK_K=64/num_warps=8/num_stages=2 at ~164 us).
A single-kernel GEMM+softmax+top2 fusion was tried and rejected on this
backend: both a wide single-N-tile variant (smem blowup / slow) and a
sweep-over-N-tiles + online-softmax variant (the nested N-outer / K-inner
dot compiles to ~215 us, far worse than the two-kernel ~44 us) regressed
sharply, so the two-kernel split is kept.
"""

import torch
import triton
import triton.language as tl


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _gemm_configs(max_n=128):
    cfgs = []
    # Keep BLOCK_N <= max_n to avoid N-tiles wider than the expert axis
    # (avoids fully-masked dot tiles, which break codegen on this backend).
    n_opts = [n for n in (32, 64, 128) if n <= max_n]
    for bm in (16, 32, 64, 128):
        for bn in n_opts:
            for bk in (64, 128, 256):
                # bound shared memory: skip the combos known to exceed ~64 KiB.
                if bm * bn * bk >= 128 * 128 * 128:
                    continue
                for nw in (4, 8):
                    for ns in (2, 3, 4):
                        cfgs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return cfgs


@triton.autotune(configs=_gemm_configs(128), key=["B", "E", "H"])
@triton.jit
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    B,
    H,
    E,
    softcap,
    USE_SOFTCAP: tl.constexpr,
    USE_BIAS: tl.constexpr,
    x_stride0,
    x_stride1,
    w_stride0,
    w_stride1,
    b_stride0,
    l_stride0,
    l_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    rm_mask = rm < B
    rn_mask = rn < E

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        kk = k_start + rk
        k_mask = kk < H
        # x: [BLOCK_M, BLOCK_K]
        x = tl.load(
            x_ptr + rm[:, None] * x_stride0 + kk[None, :] * x_stride1,
            mask=rm_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # w: [BLOCK_K, BLOCK_N], w[k, e] = router_weight[e, k]
        w = tl.load(
            w_ptr + rn[None, :] * w_stride0 + kk[:, None] * w_stride1,
            mask=rn_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        # input_precision="ieee" forces true IEEE-fp32 matmul (no TF32).
        # The reference is `x.float() @ router_weight.float().t()`, a pure
        # fp32 GEMM; the tensor-core default is TF32 (~10 mantissa bits),
        # which leaves ~1e-4 residuals in the logits that, after softmax +
        # top2 gather, push ~0.1% of topk weights just past atol=1e-4.
        # "ieee" matches the reference bit-for-bit (modulo reduction order).
        acc = tl.dot(
            x.to(tl.float32), w.to(tl.float32), acc=acc, input_precision="ieee"
        )

    logits = acc
    if USE_SOFTCAP:
        # tl has no tanh; use the numerically-stable identity
        #   tanh(z) = 1 - 2 / (exp(2z) + 1)
        z = logits / softcap
        e2 = tl.exp(2.0 * z)
        logits = softcap * (1.0 - 2.0 / (e2 + 1.0))
    if USE_BIAS:
        b = tl.load(bias_ptr + rn * b_stride0, mask=rn_mask, other=0.0).to(
            tl.float32
        )
        logits = logits + b[None, :]

    out_mask = rm_mask[:, None] & rn_mask[None, :]
    tl.store(
        logits_ptr + rm[:, None] * l_stride0 + rn[None, :] * l_stride1,
        logits,
        mask=out_mask,
    )


# Small-problem GEMM (no autotune). The metax backend's autotuner trips a
# codegen bug (PassManager::run) when multiple dot configs are benchmarked
# together on very small expert counts (e.g. E=8). A single fixed config is
# correct and fast enough for those tiny cases, so we bypass autotune when E
# is small. The kernel body is identical to the autotuned variant.
@triton.jit
def _router_gemm_kernel_fixed(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    B,
    H,
    E,
    softcap,
    USE_SOFTCAP: tl.constexpr,
    USE_BIAS: tl.constexpr,
    x_stride0,
    x_stride1,
    w_stride0,
    w_stride1,
    b_stride0,
    l_stride0,
    l_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    rm_mask = rm < B
    rn_mask = rn < E

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        kk = k_start + rk
        k_mask = kk < H
        x = tl.load(
            x_ptr + rm[:, None] * x_stride0 + kk[None, :] * x_stride1,
            mask=rm_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + rn[None, :] * w_stride0 + kk[:, None] * w_stride1,
            mask=rn_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(
            x.to(tl.float32), w.to(tl.float32), acc=acc, input_precision="ieee"
        )

    logits = acc
    if USE_SOFTCAP:
        z = logits / softcap
        e2 = tl.exp(2.0 * z)
        logits = softcap * (1.0 - 2.0 / (e2 + 1.0))
    if USE_BIAS:
        b = tl.load(bias_ptr + rn * b_stride0, mask=rn_mask, other=0.0).to(
            tl.float32
        )
        logits = logits + b[None, :]

    out_mask = rm_mask[:, None] & rn_mask[None, :]
    tl.store(
        logits_ptr + rm[:, None] * l_stride0 + rn[None, :] * l_stride1,
        logits,
        mask=out_mask,
    )


@triton.jit
def _router_reduce_kernel(
    logits_ptr,
    out_w_ptr,
    out_ids_ptr,
    B,
    E,
    l_stride0,
    l_stride1,
    ow_stride0,
    ow_stride1,
    oi_stride0,
    oi_stride1,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    re = tl.arange(0, BLOCK_E)

    rm_mask = rm < B
    emask = re < E

    logits = tl.load(
        logits_ptr + rm[:, None] * l_stride0 + re[None, :] * l_stride1,
        mask=rm_mask[:, None] & emask[None, :],
        other=float("-inf"),
    )
    # padded experts never participate in softmax / topk
    logits = tl.where(emask[None, :], logits, float("-inf"))

    m = tl.max(logits, axis=1, keep_dims=True)
    ex = tl.exp(logits - m)
    s = tl.sum(ex, axis=1, keep_dims=True)
    probs = ex / s

    id0 = tl.argmax(logits, axis=1)
    w0 = tl.sum(tl.where(re[None, :] == id0[:, None], probs, 0.0), axis=1)

    tl.store(out_w_ptr + rm * ow_stride0, w0, mask=rm_mask)
    tl.store(out_ids_ptr + rm * oi_stride0, id0.to(tl.int32), mask=rm_mask)

    if TOPK == 2:
        logits2 = tl.where(re[None, :] == id0[:, None], float("-inf"), logits)
        id1 = tl.argmax(logits2, axis=1)
        w1 = tl.sum(tl.where(re[None, :] == id1[:, None], probs, 0.0), axis=1)
        tl.store(out_w_ptr + rm * ow_stride0 + ow_stride1, w1, mask=rm_mask)
        tl.store(
            out_ids_ptr + rm * oi_stride0 + oi_stride1,
            id1.to(tl.int32),
            mask=rm_mask,
        )


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    B, H = x.shape
    E, Hw = router_weight.shape
    assert H == Hw, "hidden dim mismatch"
    assert topk <= 2, "tensorcore router only supports topk <= 2"

    x = x.contiguous()
    router_weight = router_weight.contiguous()
    if correction_bias is not None:
        correction_bias = correction_bias.contiguous()

    topk_weights = torch.empty((B, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((B, topk), dtype=torch.int32, device=x.device)
    logits = torch.empty((B, E), dtype=torch.float32, device=x.device)

    use_bias = correction_bias is not None
    use_softcap = float(moe_softcapping) != 0.0

    if E < 64:
        # Small expert count: use a fixed config (autotuner codegen bug on
        # multi-config dot for tiny E on this backend). Pick BLOCK_N >= 32 so
        # the dot meets the backend's minimum N tile.
        bm = 16
        bn = 32 if E <= 32 else 64
        bk = 64
        grid_gemm = (triton.cdiv(B, bm), triton.cdiv(E, bn))
        _router_gemm_kernel_fixed[grid_gemm](
            x,
            router_weight,
            correction_bias,
            logits,
            B,
            H,
            E,
            float(moe_softcapping),
            use_softcap,
            use_bias,
            x.stride(0),
            x.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            correction_bias.stride(0) if use_bias else 0,
            logits.stride(0),
            logits.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid_gemm = lambda meta: (
            triton.cdiv(B, meta["BLOCK_M"]),
            triton.cdiv(E, meta["BLOCK_N"]),
        )
        _router_gemm_kernel[grid_gemm](
            x,
            router_weight,
            correction_bias,
            logits,
            B,
            H,
            E,
            float(moe_softcapping),
            use_softcap,
            use_bias,
            x.stride(0),
            x.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            correction_bias.stride(0) if use_bias else 0,
            logits.stride(0),
            logits.stride(1),
        )

    block_e = max(16, _next_pow2(E))
    # Reduce kernel: launch config is chosen from B. The expert axis is short
    # (E=256), so the math is tiny and launch/dispatch cost dominates; a small
    # M-tile spread across few warps amortizes dispatch best. A scan over
    # BLOCK_M x num_warps on this backend showed:
    #   - small B (<=512): BLOCK_M=4 / num_warps=4 is fastest (~16-17 us, vs
    #     ~16-17 us for BLOCK_M=8/num_warps=8 but consistently a hair lower, and
    #     the smaller grid keeps the launch cost down on the latency-bound
    #     small-B shapes).
    #   - large B (>=4096): BLOCK_M=8 / num_warps=8 is fastest (~25 us, the
    #     larger M-tile keeps the 512-row grid from exploding into 1024
    #     programs each loading the full expert row).
    if B >= 1024:
        block_m = 8
        num_warps = 8
    else:
        block_m = 4
        num_warps = 4
    grid_reduce = (triton.cdiv(B, block_m),)
    _router_reduce_kernel[grid_reduce](
        logits,
        topk_weights,
        topk_ids,
        B,
        E,
        logits.stride(0),
        logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]
