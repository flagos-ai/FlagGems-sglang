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

"""Fused MoE router (top-k on logits + full-softmax weights) for MUSA / Triton.

Reference semantics (torch):
    logits = x.float() @ W.float().T          # [M, E] fp32
    if softcap != 0: logits = tanh(logits / softcap) * softcap
    if bias is not None: logits = logits + bias
    probs  = softmax(logits, -1)
    ids    = topk(logits, k, -1).indices      # ON LOGITS, descending
    weights = gather(probs, -1, ids)          # NOT renormalised, fp32
    return weights [M, k] fp32, ids.to(int32) [M, k]

Measured MUSA/Triton characteristics that drive the design:
  * tl.dot reaches ~48-55 TFLOPS on this N=256 shape, but a single CTA can only
    pull ~5-7 GB/s from HBM, so the GEMM stage needs >= ~64 concurrent CTAs.
    That motivates an E-split (BLOCK_N < E) and, for small M, a K-split.
  * Row reductions over a multi-warp tile are extremely expensive here
    (~1.3 cycles per tile element, dominated by cross-warp communication).
    Giving one token row to exactly one warp (one row per program, num_warps=1)
    turns every reduction into an intra-warp shuffle and is 3-4x faster in both
    latency and throughput than the equivalent wider CTA.

So the router is split into two device kernels:
  1) `_logits_kernel` : tiled x @ W^T partial sums into an fp32 workspace
                        [KSPLIT, M, E]; grid = M-tiles * E-tiles * KSPLIT.
  2) `_router_kernel` : one warp per row; sums the KSPLIT partials in fp32,
                        applies softcap then bias, softmax, and an unrolled
                        argmax/exclusion top-k; stores fp32 weights and int32 ids.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _logits_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    K,
    E,
    KCHUNK,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_pk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(E, BLOCK_N)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_k = pid // (num_m * num_n)
    rem = pid % (num_m * num_n)
    pid_m = rem // num_n
    pid_n = rem % num_n

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K
    )
    rmask = rows < M
    cmask = cols < E
    kbase = pid_k * KCHUNK
    klim = tl.minimum(kbase + KCHUNK, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, KCHUNK, BLOCK_K):
        ks = kbase + i + rk
        kmask = ks < klim
        xt = tl.load(
            x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk,
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        wt = tl.load(
            w_ptr + cols[:, None] * stride_we + ks[None, :] * stride_wk,
            mask=cmask[:, None] & kmask[None, :],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(wt), acc, out_dtype=tl.float32)

    tl.store(
        part_ptr + pid_k * stride_pk + rows[:, None] * E + cols[None, :],
        acc,
        mask=rmask[:, None] & cmask[None, :],
    )


@triton.jit
def _router_kernel(
    part_ptr,
    bias_ptr,
    w_out_ptr,
    ids_out_ptr,
    M,
    E,
    softcap,
    stride_b,
    KSPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    rows = tl.program_id(0) + tl.arange(0, 1)
    eo = tl.arange(0, BLOCK_E)
    emask = eo < E

    acc = tl.zeros((1, BLOCK_E), dtype=tl.float32)
    for kk in tl.static_range(KSPLIT):
        acc += tl.load(
            part_ptr + kk * (M * E) + rows[:, None] * E + eo[None, :],
            mask=emask[None, :],
            other=0.0,
        )

    if HAS_SOFTCAP:
        # tanh(z) * c  with z = logits / c, evaluated exactly in fp32
        z = acc / softcap
        acc = (1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)) * softcap

    if HAS_BIAS:
        b = tl.load(bias_ptr + eo * stride_b, mask=emask, other=0.0)
        acc = acc + b[None, :]

    ni = float("-inf")
    logits = tl.where(emask[None, :], acc, ni)

    # The first top-k pass returns the row maximum, which is exactly the shift
    # the reference softmax subtracts, so it is reused instead of running a
    # separate full-width max reduction.  exp(v - v) == 1, so the top weight is
    # 1/denominator, identical to the reference's exp(max - max)/denominator.
    shift, idx = tl.max(logits, axis=1, return_indices=True)
    inv_den = 1.0 / tl.sum(tl.exp(logits - shift[:, None]), axis=1)
    tl.store(w_out_ptr + rows * TOPK, inv_den)
    tl.store(ids_out_ptr + rows * TOPK, idx.to(tl.int32))
    work = tl.where(eo[None, :] == idx[:, None], ni, logits)
    for j in range(1, TOPK):
        val, idx = tl.max(work, axis=1, return_indices=True)
        tl.store(w_out_ptr + rows * TOPK + j, tl.exp(val - shift) * inv_den)
        tl.store(ids_out_ptr + rows * TOPK + j, idx.to(tl.int32))
        work = tl.where(eo[None, :] == idx[:, None], ni, work)


def _pick_config(M, K, E):
    """Launch parameters measured on the MTT S5000 target."""
    e_pow = max(16, triton.next_power_of_2(E))
    if M >= 1024:
        cfg = dict(BM=128, BN=min(128, e_pow), BK=64, KS=2, NW=16, NS=2)
    elif M >= 128:
        cfg = dict(BM=128, BN=min(64, e_pow), BK=64, KS=8, NW=16, NS=2)
    elif M >= 32:
        cfg = dict(BM=32, BN=min(32, e_pow), BK=128, KS=16, NW=4, NS=2)
    else:
        cfg = dict(BM=16, BN=min(32, e_pow), BK=128, KS=16, NW=4, NS=3)

    ks = cfg["KS"]
    while ks > 1 and triton.cdiv(K, ks) < 16:
        ks //= 2
    cfg["KS"] = ks
    chunk = triton.cdiv(K, ks)
    cfg["BK"] = min(cfg["BK"], max(16, triton.next_power_of_2(chunk)))
    return cfg


def fused_moe_router_cudacore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    M, K = x.shape
    E = router_weight.shape[0]
    topk = int(topk)
    dev = x.device

    w_out = torch.empty((M, topk), device=dev, dtype=torch.float32)
    ids_out = torch.empty((M, topk), device=dev, dtype=torch.int32)
    if M == 0:
        return w_out, ids_out

    has_softcap = (
        0.0 if moe_softcapping is None else float(moe_softcapping)
    ) != 0.0
    softcap = (
        (0.0 if moe_softcapping is None else float(moe_softcapping))
        if has_softcap
        else 1.0
    )

    has_bias = correction_bias is not None
    if has_bias:
        if E == 0 or correction_bias.numel() != E:
            has_bias = False
        else:
            correction_bias = correction_bias.reshape(-1)
    if not has_bias:
        correction_bias = x

    cfg = _pick_config(M, K, E)
    block_m = cfg["BM"]
    block_n = cfg["BN"]
    block_k = cfg["BK"]
    ksplit = cfg["KS"]
    kchunk = triton.cdiv(K, ksplit)

    part = torch.empty((ksplit, M, E), device=dev, dtype=torch.float32)

    grid = (triton.cdiv(M, block_m) * triton.cdiv(E, block_n) * ksplit,)
    _logits_kernel[grid](
        x,
        router_weight,
        part,
        M,
        K,
        E,
        kchunk,
        x.stride(0),
        x.stride(1),
        router_weight.stride(0),
        router_weight.stride(1),
        M * E,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=cfg["NW"],
        num_stages=cfg["NS"],
    )

    _router_kernel[(M,)](
        part,
        correction_bias,
        w_out,
        ids_out,
        M,
        E,
        softcap,
        correction_bias.stride(0),
        KSPLIT=ksplit,
        TOPK=topk,
        HAS_BIAS=has_bias,
        HAS_SOFTCAP=has_softcap,
        BLOCK_E=max(16, triton.next_power_of_2(E)),
        num_warps=1,
        num_stages=1,
    )
    return w_out, ids_out


__all__ = ["fused_moe_router_cudacore"]
