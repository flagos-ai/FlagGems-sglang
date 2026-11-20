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

import torch
import triton
import triton.language as tl


@triton.jit
def _short(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, W)
    x = tl.load(P + row * V + c, c < V, 0.0)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    if not KT and KV == 1:
        pivot = tl.max(x, 0)
    elif not KT and KV == V:
        pivot = tl.full((), 0.0, tl.float32)
    else:
        ranks = tl.sum(
            ((x[None, :] > x[:, None]) & (c[None, :] < V)).to(tl.int32), 1
        )
        pivot = tl.min(tl.where((c < V) & (ranks < k), x, float("inf")), 0)
    kept = tl.where(x >= pivot, x, 0.0)
    total = tl.sum(kept, 0)
    tl.store(O + row * V + c, kept / total, c < V)


@triton.jit
def _renorm(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, B)
    if KT:
        ki = 0 if SINGLE else row * KS
        k = tl.load(K + ki).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    lo = tl.full((B,), 2147483647, tl.int32)
    hi = tl.full((B,), 0, tl.int32)
    for start in range(V // B):
        c = start * B + a
        x = tl.load(P + row * V + c, True, 0)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        lo = tl.minimum(lo, tl.where(True, bits, 2147483647))
        hi = tl.maximum(hi, bits)
    if V % B != 0:
        start = tl.full((), V // B, tl.int32)
        c = start * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        lo = tl.minimum(lo, tl.where(c < V, bits, 2147483647))
        hi = tl.maximum(hi, bits)
    lo = tl.min(lo, 0)
    hi = tl.max(hi, 0)
    if k == 1:
        lo = hi
    if k == V:
        hi = lo
    nlo = tl.full((), V, tl.int32)
    nhi = tl.full((), 0, tl.int32)
    step = tl.full((), 0, tl.int32)
    if V > 65536:
        for warm in range(2):
            if lo < hi:
                lf = lo.to(tl.float32, bitcast=True)
                hf = hi.to(tl.float32, bitcast=True)
                fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
                guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
                mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
                counts = tl.full((B,), 0, tl.int16)
                for start in range(V // B):
                    c = start * B + a
                    x = tl.load(P + row * V + c)
                    bits = x.to(tl.int32, bitcast=True) & 2147483647
                    counts += (bits >= mid).to(tl.int16)
                if V % B != 0:
                    c = V // B * B + a
                    x = tl.load(P + row * V + c, c < V, 0.0)
                    bits = x.to(tl.int32, bitcast=True) & 2147483647
                    counts += ((c < V) & (bits >= mid)).to(tl.int16)
                count = tl.sum(counts.to(tl.int32), 0)
                lo = tl.where(count >= k, mid, lo)
                hi = tl.where(
                    count <= k, tl.where(count == k, mid, mid - 1), hi
                )
                nlo = tl.where(count >= k, count, nlo)
                nhi = tl.where(count < k, count, nhi)
    while lo < hi:
        lf = lo.to(tl.float32, bitcast=True)
        hf = hi.to(tl.float32, bitcast=True)
        fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
        guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
        mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
        mid = tl.where(step % 4 == 3, lo + (hi - lo + 1) // 2, mid)
        counts = tl.full((B,), 0, tl.int32)
        lower = tl.full((B,), 0, tl.int32)
        upper = tl.full((B,), 2147483647, tl.int32)
        for start in range(V // B):
            c = start * B + a
            x = tl.load(P + row * V + c, True, 0)
            bits = x.to(tl.int32, bitcast=True) & 2147483647
            above = True & (bits >= mid)
            counts += above.to(tl.int32)
            lower = tl.maximum(lower, tl.where(True & (bits < mid), bits, 0))
            upper = tl.minimum(upper, tl.where(above, bits, 2147483647))
        if V % B != 0:
            start = tl.full((), V // B, tl.int32)
            c = start * B + a
            x = tl.load(P + row * V + c, c < V, 0)
            bits = x.to(tl.int32, bitcast=True) & 2147483647
            above = (c < V) & (bits >= mid)
            counts += above.to(tl.int32)
            lower = tl.maximum(
                lower, tl.where((c < V) & (bits < mid), bits, 0)
            )
            upper = tl.minimum(upper, tl.where(above, bits, 2147483647))
        lower = tl.max(lower, 0)
        upper = tl.min(upper, 0)
        count = tl.sum(counts, 0)
        lo = tl.where(count >= k, upper, lo)
        hi = tl.where(count <= k, tl.where(count == k, upper, lower), hi)
        nlo = tl.where(count >= k, count, nlo)
        nhi = tl.where(count < k, count, nhi)
        step += 1
    pivot = lo.to(tl.float32, bitcast=True)
    sums = tl.full((B,), 0.0, tl.float32)
    for start in range(V // B):
        c = start * B + a
        x = tl.load(P + row * V + c, True, 0)
        sums += tl.where(x >= pivot, x, 0.0)
    if V % B != 0:
        start = tl.full((), V // B, tl.int32)
        c = start * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        sums += tl.where(x >= pivot, x, 0.0)
    denominator = tl.sum(sums, 0)
    for start in range(V // B):
        c = start * B + a
        x = tl.load(P + row * V + c, True, 0)
        out = tl.where(x >= pivot, x / denominator, 0.0)
        tl.store(O + row * V + c, out, True)
    if V % B != 0:
        start = tl.full((), V // B, tl.int32)
        c = start * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        out = tl.where(x >= pivot, x / denominator, 0.0)
        tl.store(O + row * V + c, out, c < V)


@triton.jit
def _renorm_wide(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, B)
    if KT:
        ki = 0 if SINGLE else row * KS
        k = tl.load(K + ki).to(tl.int64)
    else:
        k = tl.full((), KV, tl.int64)
    lo = tl.full((B,), 2147483647, tl.int32)
    hi = tl.full((B,), 0, tl.int32)
    for start in range(triton.cdiv(V, B)):
        c = start.to(tl.int64) * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        lo = tl.minimum(lo, tl.where(c < V, bits, 2147483647))
        hi = tl.maximum(hi, bits)
    lo = tl.min(lo, 0)
    hi = tl.max(hi, 0)
    if k == 1:
        lo = hi
    if k == V:
        hi = lo
    nlo = tl.full((), V, tl.int64)
    nhi = tl.full((), 0, tl.int64)
    step = tl.full((), 0, tl.int32)
    while lo < hi:
        lf = lo.to(tl.float32, bitcast=True)
        hf = hi.to(tl.float32, bitcast=True)
        fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
        guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
        mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
        mid = tl.where(step % 4 == 3, lo + (hi - lo + 1) // 2, mid)
        counts = tl.full((B,), 0, tl.int64)
        lower = tl.full((B,), 0, tl.int32)
        upper = tl.full((B,), 2147483647, tl.int32)
        for start in range(triton.cdiv(V, B)):
            c = start.to(tl.int64) * B + a
            x = tl.load(P + row * V + c, c < V, 0)
            bits = x.to(tl.int32, bitcast=True) & 2147483647
            above = (c < V) & (bits >= mid)
            counts += above.to(tl.int32)
            lower = tl.maximum(
                lower, tl.where((c < V) & (bits < mid), bits, 0)
            )
            upper = tl.minimum(upper, tl.where(above, bits, 2147483647))
        lower = tl.max(lower, 0)
        upper = tl.min(upper, 0)
        count = tl.sum(counts, 0)
        lo = tl.where(count >= k, upper, lo)
        hi = tl.where(count <= k, tl.where(count == k, upper, lower), hi)
        nlo = tl.where(count >= k, count, nlo)
        nhi = tl.where(count < k, count, nhi)
        step += 1
    pivot = lo.to(tl.float32, bitcast=True)
    sums = tl.full((B,), 0.0, tl.float32)
    correction = tl.full((B,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, B)):
        c = start.to(tl.int64) * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        value = tl.where(x >= pivot, x, 0.0) - correction
        updated = sums + value
        correction = updated - sums - value
        sums = updated
    denominator = tl.sum(sums, 0)
    for start in range(triton.cdiv(V, B)):
        c = start.to(tl.int64) * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        out = tl.where(x >= pivot, x / denominator, 0.0)
        tl.store(O + row * V + c, out, c < V)


@triton.jit
def _copy_complete(P, Out, N: tl.constexpr, W: tl.constexpr):
    c = tl.program_id(0).to(tl.int64) * W + tl.arange(0, W)
    x = tl.load(P + tl.minimum(c, N - 1))
    tl.store(Out + c, x, c < N)


MAX_BATCH = 65535


def top_k_renorm(probs, top_k):
    n, v = probs.shape
    out = torch.empty_like(probs)
    if n > MAX_BATCH:
        for start in range(0, n, MAX_BATCH):
            end = min(start + MAX_BATCH, n)
            current_k = (
                top_k[start:end]
                if isinstance(top_k, torch.Tensor) and top_k.numel() > 1
                else top_k
            )
            current = top_k_renorm(probs[start:end], current_k)
            _copy_complete[triton.cdiv(current.numel(), 4096),](
                current,
                out[start:end],
                current.numel(),
                4096,
                num_warps=4,
                num_stages=1,
            )
        return out
    kt = isinstance(top_k, torch.Tensor)
    keys = top_k if kt else None
    ks = (top_k.stride(0) if top_k.numel() > 1 else 0) if kt else 0
    kv = 0 if kt else int(top_k)
    if v <= 128 or (kv == v and v <= 8192):
        _packed.run(
            probs,
            keys,
            out,
            (v, ks, kv),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(n,),
            warmup=False,
        )
    else:
        _packed[n,](
            probs,
            keys,
            out,
            (v, ks, kv),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _packed(P, K, Out, META: tl.constexpr):
    V: tl.constexpr = META[0]
    KS: tl.constexpr = META[1]
    KV: tl.constexpr = META[2]
    KT: tl.constexpr = KV == 0
    SINGLE: tl.constexpr = KS == 0
    W: tl.constexpr = triton.next_power_of_2(V)
    B: tl.constexpr = 4096 if V > 4096 else W
    if V <= 128:
        _short(P, K, Out, V, KS, KT, SINGLE, KV, W)
    elif V > 1048576:
        _renorm_wide(P, K, Out, V, KS, KT, SINGLE, KV, 2048)
    else:
        _renorm(P, K, Out, V, KS, KT, SINGLE, KV, B)


__all__ = ["top_k_renorm"]
