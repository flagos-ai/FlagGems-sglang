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

"""Fused MoE router (tensor-core) for the MUSA / Triton target.

logits = x @ W^T                     (bf16 operands, fp32 accumulate)
logits = tanh(logits / softcap) * softcap      (only when softcap != 0)
logits = logits + bias                          (only when bias is not None)
probs  = softmax(logits)
(vals, ids) = topk(logits, k)
weights = gather(probs, ids)

Measured target facts that drive the design:
  * a kernel launch costs ~11 us on this backend, so the number of launches is
    a first-order cost for the small-M workloads;
  * a single CTA sustains only ~25-30 GB/s, while ~60 CTAs reach ~1.5 TB/s, so
    small-M workloads (which own only M/BLOCK_M work units) are latency bound
    and must split the K axis to get parallelism;
  * Triton software pipelining (num_stages > 1) is counter-productive on this
    backend (measured 3-5x slower), so every kernel runs with num_stages=1;
  * the "swap" orientation dot(W_tile, trans(x_tile)) is faster than
    dot(x_tile, trans(W_tile)) because the big operand W is then the A operand.

Dispatch:
  small/medium M with a long K  -> two kernels: a K-split partial GEMM plus a
                                   per-row reduce/softcap/softmax/top-k kernel
  large M                       -> one fully fused kernel over row blocks
"""

import torch
import triton
import triton.language as tl

_SPLIT_M_MAX = 1024
_SPLIT_K_MIN = 1024
_MAX_BLOCK_E = 256


@triton.jit
def _router_fused(
    x_ptr,
    w_ptr,
    bias_ptr,
    w_out_ptr,
    id_out_ptr,
    M,
    K,
    E,
    sxm,
    sxk,
    swe,
    swk,
    som,
    sok,
    sim,
    sik,
    softcap,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TOPK: tl.constexpr,
    EVEN_E: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    e = tl.arange(0, BLOCK_E)

    acc = tl.zeros((BLOCK_E, BLOCK_M), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        if EVEN_E:
            a = tl.load(
                w_ptr + e[:, None] * swe + kk[None, :] * swk,
                mask=(kk < K)[None, :],
                other=0.0,
            )
        else:
            a = tl.load(
                w_ptr + e[:, None] * swe + kk[None, :] * swk,
                mask=(e < E)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
        xt = tl.load(
            x_ptr + rows[:, None] * sxm + kk[None, :] * sxk,
            mask=rmask[:, None] & (kk < K)[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(xt), acc, out_dtype=tl.float32)

    logits = acc
    if HAS_SOFTCAP:
        logits = (tl.sigmoid(logits * (2.0 / softcap)) * 2.0 - 1.0) * softcap
    if HAS_BIAS:
        if EVEN_E:
            bias = tl.load(bias_ptr + e)
        else:
            bias = tl.load(bias_ptr + e, mask=e < E, other=0.0)
        logits = logits + bias[:, None]
    if not EVEN_E:
        logits = tl.where((e < E)[:, None], logits, float("-inf"))

    rowmax = tl.max(logits, axis=0)
    rowmax = tl.where(rmask, rowmax, 0.0)
    p = tl.exp(logits - rowmax[None, :])
    probs = p / tl.sum(p, axis=0)[None, :]
    for j in tl.static_range(K_TOPK):
        idx = tl.argmax(logits, axis=0)
        wv = tl.sum(tl.where(e[:, None] == idx[None, :], probs, 0.0), axis=0)
        tl.store(w_out_ptr + rows * som + j * sok, wv, mask=rmask)
        tl.store(
            id_out_ptr + rows * sim + j * sik, idx.to(tl.int32), mask=rmask
        )
        logits = tl.where(e[:, None] == idx[None, :], float("-inf"), logits)


@triton.jit
def _router_partial(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    K,
    E,
    sxm,
    sxk,
    swe,
    swk,
    spe,
    KS,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_E: tl.constexpr,
):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    e = tl.arange(0, BLOCK_E)

    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    k_lo = sid * KS
    for k0 in tl.range(k_lo, k_lo + KS, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x_ptr + rows[:, None] * sxm + kk[None, :] * sxk,
            mask=rmask[:, None] & (kk < K)[None, :],
            other=0.0,
        )
        if EVEN_E:
            b = tl.load(w_ptr + e[:, None] * swe + kk[None, :] * swk)
        else:
            b = tl.load(
                w_ptr + e[:, None] * swe + kk[None, :] * swk,
                mask=(e < E)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    if EVEN_E:
        tl.store(
            part_ptr + rows[:, None] * spe + sid * BLOCK_E + e[None, :],
            acc,
            mask=rmask[:, None],
        )
    else:
        tl.store(
            part_ptr + rows[:, None] * spe + sid * BLOCK_E + e[None, :],
            acc,
            mask=rmask[:, None] & (e < E)[None, :],
        )


@triton.jit
def _router_reduce(
    part_ptr,
    bias_ptr,
    w_out_ptr,
    id_out_ptr,
    E,
    spe,
    som,
    sok,
    sim,
    sik,
    softcap,
    BLOCK_E: tl.constexpr,
    NSP: tl.constexpr,
    K_TOPK: tl.constexpr,
    EVEN_E: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.arange(0, NSP)
    e = tl.arange(0, BLOCK_E)

    tile = tl.load(
        part_ptr + row * spe + s[:, None] * BLOCK_E + e[None, :],
        mask=(e < E)[None, :],
        other=0.0,
    )
    logits = tl.sum(tile, axis=0)
    if HAS_SOFTCAP:
        logits = (tl.sigmoid(logits * (2.0 / softcap)) * 2.0 - 1.0) * softcap
    if HAS_BIAS:
        if EVEN_E:
            logits = logits + tl.load(bias_ptr + e)
        else:
            logits = logits + tl.load(bias_ptr + e, mask=e < E, other=0.0)
    if not EVEN_E:
        logits = tl.where(e < E, logits, float("-inf"))

    rowmax = tl.max(logits)
    p = tl.exp(logits - rowmax)
    probs = p / tl.sum(p)
    for j in tl.static_range(K_TOPK):
        idx = tl.argmax(logits, axis=0)
        wv = tl.sum(tl.where(e == idx, probs, 0.0))
        tl.store(w_out_ptr + row * som + j * sok, wv)
        tl.store(id_out_ptr + row * sim + j * sik, idx.to(tl.int32))
        logits = tl.where(e == idx, float("-inf"), logits)


def _pick_split(M, K, block_e):
    """Return (BLOCK_M, NSPLIT, BLOCK_K, num_warps) or None if the split is unusable."""
    if M <= 16:
        bm, ns, bk, nw = 32, 8, 256, 8
    elif M <= 128:
        bm, ns, bk, nw = 16, 16, 128, 4
    else:
        bm, ns, bk, nw = 64, 4, 128, 16
    if K % ns:
        return None
    ks = K // ns
    bk = min(bk, ks)
    while bk > 16 and (ks % bk or K % bk):
        bk //= 2
    if ks % bk or K % bk or bk < 16:
        return None
    if triton.cdiv(M, bm) * ns > 4096:
        return None
    return bm, ns, bk, nw


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    if correction_bias is not None:
        correction_bias = correction_bias.contiguous()
    M, K = x.shape
    E = router_weight.shape[0]
    k = int(topk)
    softcap = 0.0 if moe_softcapping is None else float(moe_softcapping)
    has_softcap = softcap != 0.0
    has_bias = correction_bias is not None

    topk_weights = torch.empty((M, k), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((M, k), dtype=torch.int32, device=x.device)

    block_e = max(16, triton.next_power_of_2(E))
    even_e = block_e == E
    sxm, sxk = x.stride(0), x.stride(1)
    swe, swk = router_weight.stride(0), router_weight.stride(1)
    som, sok = topk_weights.stride(0), topk_weights.stride(1)
    sim, sik = topk_ids.stride(0), topk_ids.stride(1)
    bias_arg = correction_bias if has_bias else x

    split = None
    if M <= _SPLIT_M_MAX and K >= _SPLIT_K_MIN and block_e <= _MAX_BLOCK_E:
        split = _pick_split(M, K, block_e)

    if split is not None:
        bm, ns, bk, nw = split
        part = torch.empty(
            (M, ns, block_e), dtype=torch.float32, device=x.device
        )
        spe = ns * block_e
        _router_partial[(triton.cdiv(M, bm), ns)](
            x,
            router_weight,
            part,
            M,
            K,
            E,
            sxm,
            sxk,
            swe,
            swk,
            spe,
            K // ns,
            BLOCK_M=bm,
            BLOCK_E=block_e,
            BLOCK_K=bk,
            EVEN_E=even_e,
            num_warps=nw,
            num_stages=1,
        )
        _router_reduce[(M,)](
            part,
            bias_arg,
            topk_weights,
            topk_ids,
            E,
            spe,
            som,
            sok,
            sim,
            sik,
            softcap,
            BLOCK_E=block_e,
            NSP=ns,
            K_TOPK=k,
            EVEN_E=even_e,
            HAS_BIAS=has_bias,
            HAS_SOFTCAP=has_softcap,
            num_warps=4,
            num_stages=1,
        )
        return topk_weights, topk_ids

    bm = 32 if M > 32 else 16
    bk = min(64, max(16, triton.next_power_of_2(min(K, 64))))
    if K % bk:
        bk = 16
    _router_fused[(triton.cdiv(M, bm),)](
        x,
        router_weight,
        bias_arg,
        topk_weights,
        topk_ids,
        M,
        K,
        E,
        sxm,
        sxk,
        swe,
        swk,
        som,
        sok,
        sim,
        sik,
        softcap,
        BLOCK_M=bm,
        BLOCK_E=block_e,
        BLOCK_K=bk,
        K_TOPK=k,
        EVEN_E=even_e,
        HAS_BIAS=has_bias,
        HAS_SOFTCAP=has_softcap,
        num_warps=8,
        num_stages=1,
    )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]
