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

"""Triton top_k_renorm.

The cutoff is the k-th largest value of the row and the mask is ``probs
>= cutoff``, so entries tied at the cutoff are all kept (a row may keep
more than k entries); this mirrors sglang's ``top_k_renorm_probs``. The
output is fp32 and rows whose retained mass is zero stay zero. ``top_k``
accepts an int, a scalar tensor, or one int per row.

Probabilities are non-negative, so their fp32 bit patterns are
order-monotone and the pivot is the largest bit pattern whose retained
count reaches k. The pivot search is a bisection in the integer domain:
pure integer comparisons, no data-dependent sorting.

Two regimes:

* ``vocab <= 32768``: the whole row lives in one CTA's registers. One
  load, ~30 integer bisection steps in registers, one store: 1R+1W.
* ``vocab > 32768``: radix select spread over SMs. K1 atomic-histograms
  each row's top 16 value bits (65536 bins); K2 locates the pivot bucket
  and the in-bucket rank with one cumsum; K3 gathers the bucket into a
  bounded per-row scratch; K4 bisects the exact pivot inside the bucket
  and atomically accumulates the retained mass; K5 re-bisects the pivot
  (a few microseconds from the scratch) and writes kept/den. 4R+1W,
  every row pass fully parallel. Degenerate buckets (count > scratch
  size) fall back to an exact in-kernel bisection so results stay
  correct for any input. Scratch buffers are cached per (batch, device).
"""

import torch
import triton
import triton.language as tl

_REG_LIMIT = 32768
_SCRATCH_T = 4096
_SCRATCH_CACHE = {}


@triton.jit
def _bisect_scratch(S, c, kp, T: tl.constexpr):
    """Largest bit pattern in the gathered bucket with count(>= piv) >= kp."""
    offs = tl.arange(0, T)
    bits = tl.load(S + offs, mask=offs < c, other=-2147483648)
    lo = tl.full((), 0, tl.int32)
    hi = tl.max(bits, 0)
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        cnt = tl.sum((bits >= mid).to(tl.int32), 0)
        take = cnt >= kp
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    return lo


@triton.jit
def _topk_renorm_reg(
    P,
    K,
    O,
    SP: tl.constexpr,
    SO: tl.constexpr,
    V: tl.constexpr,
    TOP_K: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    KS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    m = off < V
    x = tl.load(
        P + row.to(tl.int64) * SP + off, mask=m, other=float("-inf")
    ).to(tl.float32)
    bits = x.to(tl.int32, bitcast=True)

    if IS_TENSOR:
        k = tl.load(K + row.to(tl.int64) * KS).to(tl.int32)
        kk = tl.minimum(tl.maximum(k, 1), V)
    else:
        kk = tl.minimum(tl.maximum(TOP_K, 1), V)

    lo = tl.full((), 0, tl.int32)
    hi = tl.max(bits, 0)
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        cnt = tl.sum((bits >= mid).to(tl.int32), 0)
        take = cnt >= kk
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)

    kept = tl.where(bits >= lo, x, 0.0)
    denom = tl.sum(kept, 0)
    denom = tl.where(denom > 0, denom, 1.0)
    tl.store(O + row.to(tl.int64) * SO + off, kept / denom, mask=m)


@triton.jit
def _k1_hist(P, H, V, SP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    m = cols < V
    x = tl.load(P + row.to(tl.int64) * SP + cols, mask=m, other=0.0)
    key = (x.to(tl.int32, bitcast=True) >> 16) & 0xFFFF
    tl.atomic_add(H + row * 65536 + key, 1, mask=m)


@triton.jit
def _k2_bucket(
    H,
    K,
    BS,
    KP,
    V,
    TOP_K: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    KS: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, 65536)
    h = tl.load(H + row * 65536 + bins)

    if IS_TENSOR:
        k = tl.load(K + row.to(tl.int64) * KS).to(tl.int32)
        kk = tl.minimum(tl.maximum(k, 1), V)
    else:
        kk = tl.minimum(tl.maximum(TOP_K, 1), V)

    prefix = tl.cumsum(h, 0)
    # The kk-th largest (counted from the top) falls in the first bucket
    # whose ascending prefix exceeds V - kk; the count of prefix entries
    # <= V - kk is that bucket index. Elements strictly above the bucket
    # number V - prefix[bs], so the in-bucket rank from the top is
    # kp = kk - V + prefix[bs], which lies in [1, h[bs]].
    n = tl.sum((prefix <= V - kk).to(tl.int32), 0)
    bs = tl.minimum(n, 65535)
    pref_b = tl.sum(tl.where(bins == bs, prefix, 0))
    tl.store(BS + row, bs)
    tl.store(KP + row, kk - V + pref_b)


@triton.jit
def _k3_gather(
    P,
    BS,
    CNT,
    S,
    V,
    SP: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    bs = tl.load(BS + row)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    m = cols < V
    x = tl.load(P + row.to(tl.int64) * SP + cols, mask=m, other=0.0)
    bits = x.to(tl.int32, bitcast=True)
    hit = m & (((bits >> 16) & 0xFFFF) == bs)
    pos = tl.atomic_add(CNT + row + cols * 0, 1, mask=hit)
    ok = hit & (pos < T)
    tl.store(S + row.to(tl.int64) * T + pos, bits, mask=ok)


@triton.jit
def _k4_den(
    P,
    CNT,
    KP,
    S,
    DEN,
    T: tl.constexpr,
    SP: tl.constexpr,
    V,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    c_raw = tl.load(CNT + row)
    kp = tl.load(KP + row)
    c = tl.minimum(c_raw, T)
    ok = (c_raw <= T) & (c_raw > 0) & (kp >= 1)
    if ok:
        piv = _bisect_scratch(S + row.to(tl.int64) * T, c, kp, T)
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        m = cols < V
        x = tl.load(P + row.to(tl.int64) * SP + cols, mask=m, other=0.0)
        b = x.to(tl.int32, bitcast=True)
        den = tl.sum(tl.where(b >= piv, x.to(tl.float32), 0.0), 0)
        lane = tl.arange(0, BLOCK)
        tl.atomic_add(DEN + row + cols * 0, den, mask=lane == 0)


@triton.jit
def _k5_write(
    P,
    CNT,
    KP,
    S,
    DEN,
    O,
    K,
    T: tl.constexpr,
    SP: tl.constexpr,
    SO: tl.constexpr,
    V,
    TOP_K: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    KS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    c_raw = tl.load(CNT + row)
    kp = tl.load(KP + row)
    c = tl.minimum(c_raw, T)
    ok = (c_raw <= T) & (c_raw > 0) & (kp >= 1)
    base = P + row.to(tl.int64) * SP
    ob = O + row.to(tl.int64) * SO
    off = tl.arange(0, BLOCK)

    if ok:
        piv = _bisect_scratch(S + row.to(tl.int64) * T, c, kp, T)
        den = tl.load(DEN + row)
        den = tl.where(den > 0, den, 1.0)
        cols = chunk * BLOCK + off
        m = cols < V
        x = tl.load(base + cols, mask=m, other=0.0)
        b = x.to(tl.int32, bitcast=True)
        kept = tl.where(b >= piv, x.to(tl.float32), 0.0)
        tl.store(ob + cols, kept / den, mask=m, cache_modifier=".cs")
    elif chunk == 0:
        # Rare fallback (degenerate bucket, non-finite dtypes): exact
        # full-row pipeline inside one CTA.
        if IS_TENSOR:
            k = tl.load(K + row.to(tl.int64) * KS).to(tl.int32)
        else:
            k = TOP_K
        kk = tl.minimum(tl.maximum(k, 1), V)
        hi = tl.full((), -2147483648, tl.int32)
        for s in range(0, V, BLOCK):
            cols = s + off
            b = tl.load(
                base + cols, mask=cols < V, other=float("-inf")
            ).to(tl.float32).to(tl.int32, bitcast=True)
            hi = tl.maximum(hi, tl.max(b, 0))
        lo = tl.full((), 0, tl.int32)
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            cnt = 0
            for s in range(0, V, BLOCK):
                cols = s + off
                b = tl.load(
                    base + cols, mask=cols < V, other=float("-inf")
                ).to(tl.float32).to(tl.int32, bitcast=True)
                cnt += tl.sum((b >= mid).to(tl.int32), 0)
            take = cnt >= kk
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        piv = lo
        denom = tl.zeros((), dtype=tl.float32)
        for s in range(0, V, BLOCK):
            cols = s + off
            x = tl.load(base + cols, mask=cols < V, other=0.0).to(tl.float32)
            b = x.to(tl.int32, bitcast=True)
            denom += tl.sum(tl.where(b >= piv, x, 0.0), 0)
        denom = tl.where(denom > 0, denom, 1.0)
        for s in range(0, V, BLOCK):
            cols = s + off
            x = tl.load(base + cols, mask=cols < V, other=0.0).to(tl.float32)
            b = x.to(tl.int32, bitcast=True)
            kept = tl.where(b >= piv, x, 0.0)
            tl.store(
                ob + cols,
                kept / denom,
                mask=cols < V,
                cache_modifier=".cs",
            )


def top_k_renorm(probs, top_k):
    if probs.dim() != 2:
        raise ValueError(
            f"top_k_renorm expects a 2-D probs tensor, got {probs.dim()}-D"
        )
    batch, vocab = probs.shape

    if isinstance(top_k, torch.Tensor):
        ks = top_k.to(device=probs.device, dtype=torch.int64).reshape(-1)
        if ks.numel() not in (1, batch):
            raise ValueError(
                "top_k tensor must be a scalar or one value per row: "
                f"got {ks.numel()} for batch {batch}"
            )
    elif isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an int or a tensor, got {top_k!r}")

    if batch == 0 or vocab == 0:
        return probs.float()

    x = probs.contiguous().float()
    if isinstance(top_k, torch.Tensor):
        if ks.numel() == 1:
            k_arg, k_stride = ks, 0
        else:
            k_arg, k_stride = ks.contiguous(), ks.stride(0)
        is_tensor, scalar = True, 0
    else:
        k_arg, k_stride, is_tensor, scalar = x, 0, False, int(top_k)

    out = torch.empty_like(x)
    if vocab <= _REG_LIMIT:
        block = max(16, triton.next_power_of_2(vocab))
        warps = min(16, max(1, block // 512))
        _topk_renorm_reg[(batch,)](
            x,
            k_arg,
            out,
            x.stride(0),
            out.stride(0),
            vocab,
            scalar,
            is_tensor,
            k_stride,
            block,
            num_warps=warps,
        )
        return out

    dev = x.device
    bufs = _SCRATCH_CACHE.get((batch, dev))
    if bufs is None:
        hist = torch.zeros((batch, 65536), dtype=torch.int32, device=dev)
        cnt = torch.zeros((batch,), dtype=torch.int32, device=dev)
        den = torch.zeros((batch,), dtype=torch.float32, device=dev)
        bs = torch.empty((batch,), dtype=torch.int32, device=dev)
        kp = torch.empty((batch,), dtype=torch.int32, device=dev)
        scratch = torch.empty((batch, _SCRATCH_T), dtype=torch.int32,
                              device=dev)
        bufs = (hist, cnt, den, bs, kp, scratch)
        _SCRATCH_CACHE[(batch, dev)] = bufs
    hist, cnt, den, bs, kp, scratch = bufs
    hist.zero_()
    cnt.zero_()
    den.zero_()

    small_grid = batch * triton.cdiv(vocab, 4096) < 256
    block = 4096 if small_grid else 8192
    warps = 4 if small_grid else 8
    grid = (batch, triton.cdiv(vocab, block))

    _k1_hist[grid](x, hist, vocab, x.stride(0), block, num_warps=warps)
    _k2_bucket[(batch,)](
        hist, k_arg, bs, kp, vocab, scalar, is_tensor, k_stride,
        num_warps=16,
    )
    _k3_gather[grid](
        x, bs, cnt, scratch, vocab, x.stride(0), _SCRATCH_T, block,
        num_warps=warps,
    )
    _k4_den[grid](
        x, cnt, kp, scratch, den, _SCRATCH_T, x.stride(0), vocab, block,
        num_warps=warps,
    )
    _k5_write[grid](
        x, cnt, kp, scratch, den, out, k_arg, _SCRATCH_T, x.stride(0),
        out.stride(0), vocab, scalar, is_tensor, k_stride, block,
        num_warps=warps,
    )
    return out


__all__ = ["top_k_renorm"]
