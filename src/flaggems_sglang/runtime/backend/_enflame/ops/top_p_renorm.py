# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# GCU (L600) rewrite of the generic top_p_renorm kernel.
#
# Divergence from the generic implementation (kept untouched upstream as the
# comparison baseline):
#   1. Two-phase search schedule: the first MID iterations run a mass-only
#      bit-pattern bisection (lo=mid+1 / hi=mid, one tree reduction per pass);
#      only the remaining iterations use data-value clamping via lower/upper.
#      The clamping is what makes the search converge in ~10-14 passes, but
#      it costs three tree reductions per pass, so cheap prepasses first narrow
#      the bracket and the expensive clamped passes finish the job.
#   2. lower/upper hoisted to (BLOCK,) accumulators (max_hi/min_hi) with a
#      single tree reduction per pass after the chunk loop, instead of a full
#      reduction per chunk (9 -> 3 reductions per pass at V=131072).
#   3. Linear-in-value interpolation proposal instead of sqrt-space.
#   4. Host: BLOCK=16384 for vocab > 32768 (hoisted accumulators stay within
#      the DSM budget), num_warps=8 for large blocks.
# Accuracy: the search predicate (cumulative mass at a pivot) and all fp
# accumulation orders are unchanged, so the selected pivot is the same data
# value as the generic kernel (verified bit-identical outputs on the test
# cases). The interpolation only steers proposals and cannot change the
# result. The generic file also carried a dead `_add_pairs` helper which is
# not reproduced here.

import torch
import triton
import triton.language as tl


@triton.jit
def _top_p_select_interp(
    P,
    T,
    O,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV: tl.constexpr,
    BLOCK: tl.constexpr,
    MID: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if IS_TENSOR:
        ti = 0 if SINGLE else row * TS
        p = tl.load(T + ti).to(tl.float32)
    else:
        p = tl.full((), PV, tl.float32)
    target = 1.0 - p
    low = tl.full((), float("inf"), tl.float32)
    high = tl.full((), 0.0, tl.float32)
    mass_hi = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        low = tl.minimum(low, tl.min(tl.where(valid, x, float("inf")), 0))
        high = tl.maximum(high, tl.max(tl.where(valid, x, 0.0), 0))
        mass_hi = mass_hi + tl.where(valid, x, 0.0)
    mass = tl.sum(mass_hi, 0)
    cdf_lo = tl.full((), 0.0, tl.float32)
    cdf_hi = mass
    iteration = tl.full((), 0, tl.int32)
    extra = False
    lo = low.to(tl.int32, bitcast=True)
    lo = tl.where(lo == -2147483648, 0, lo)
    hi = high.to(tl.int32, bitcast=True)
    if target > 0:
        while (lo < hi) & (iteration < MID):
            mid = lo + (hi - lo) // 2
            if ((iteration & 1) == 0) | (((iteration & 3) == 1) & extra):
                fraction = tl.minimum(
                    1.0,
                    tl.maximum(
                        0.0,
                        (target - cdf_lo)
                        / tl.maximum(cdf_hi - cdf_lo, 1.0e-30),
                    ),
                )
                low_value = lo.to(tl.float32, bitcast=True)
                high_value = hi.to(tl.float32, bitcast=True)
                proposal = (
                    low_value + (high_value - low_value) * fraction
                ).to(tl.int32, bitcast=True)
                mid = tl.minimum(hi - 1, tl.maximum(lo, proposal))
            pivot = mid.to(tl.float32, bitcast=True)
            mass_hi = tl.full((BLOCK,), 0.0, tl.float32)
            for start in range(triton.cdiv(V, BLOCK)):
                cols = start * BLOCK + offsets
                valid = cols < V
                x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(
                    tl.float32
                )
                mass_hi = mass_hi + tl.where(valid & (x <= pivot), x, 0.0)
            mass = tl.sum(mass_hi, 0)
            enough = mass >= target
            hi = tl.where(enough, mid, hi)
            lo = tl.where(enough, lo, mid + 1)
            extra = tl.abs(mass - target) <= 0.5 * tl.minimum(
                target - cdf_lo, cdf_hi - target
            )
            cdf_lo = tl.where(enough, cdf_lo, mass)
            cdf_hi = tl.where(enough, mass, cdf_hi)
            iteration += 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if ((iteration & 1) == 0) | (((iteration & 3) == 1) & extra):
                fraction = tl.minimum(
                    1.0,
                    tl.maximum(
                        0.0,
                        (target - cdf_lo)
                        / tl.maximum(cdf_hi - cdf_lo, 1.0e-30),
                    ),
                )
                low_value = lo.to(tl.float32, bitcast=True)
                high_value = hi.to(tl.float32, bitcast=True)
                proposal = (
                    low_value + (high_value - low_value) * fraction
                ).to(tl.int32, bitcast=True)
                mid = tl.minimum(hi - 1, tl.maximum(lo, proposal))
            pivot = mid.to(tl.float32, bitcast=True)
            mass_hi = tl.full((BLOCK,), 0.0, tl.float32)
            max_hi = tl.full((BLOCK,), 0.0, tl.float32)
            min_hi = tl.full((BLOCK,), float("inf"), tl.float32)
            for start in range(triton.cdiv(V, BLOCK)):
                cols = start * BLOCK + offsets
                valid = cols < V
                x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(
                    tl.float32
                )
                below = valid & (x <= pivot)
                selected = tl.where(below, x, 0.0)
                mass_hi = mass_hi + selected
                max_hi = tl.maximum(max_hi, selected)
                min_hi = tl.minimum(
                    min_hi, tl.where(valid & (x > pivot), x, float("inf"))
                )
            mass = tl.sum(mass_hi, 0)
            lower = tl.max(max_hi, 0)
            upper = tl.min(min_hi, 0)
            enough = mass >= target
            hi = tl.where(enough, lower.to(tl.int32, bitcast=True), hi)
            lo = tl.where(enough, lo, upper.to(tl.int32, bitcast=True))
            extra = tl.abs(mass - target) <= 0.5 * tl.minimum(
                target - cdf_lo, cdf_hi - target
            )
            cdf_lo = tl.where(enough, cdf_lo, mass)
            cdf_hi = tl.where(enough, mass, cdf_hi)
            iteration += 1
    pivot = lo.to(tl.float32, bitcast=True)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        total += tl.where(valid & (x >= pivot), x, 0.0)
    denom = tl.sum(total, 0)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        y = tl.where(x >= pivot, x / denom, 0.0)
        tl.store(O + row * V + cols, y, valid)


def top_p_renorm(probs, top_p):
    batch, vocab = probs.shape
    out = torch.empty((batch, vocab), dtype=torch.float32, device=probs.device)
    is_tensor = isinstance(top_p, torch.Tensor)
    if is_tensor:
        values = top_p.to(device=probs.device).reshape(-1)
        single = values.numel() == 1
        stride = values.stride(0)
        scalar = 0.0
    else:
        values = probs
        single = True
        stride = 0
        scalar = float(top_p)
    if batch > 0 and vocab > 0:
        if vocab > 32768:
            block = 16384
        else:
            block = min(32768, triton.next_power_of_2(vocab))
        _top_p_select_interp[(batch,)](
            probs,
            values,
            out,
            vocab,
            probs.stride(0),
            probs.stride(1),
            stride,
            is_tensor,
            single,
            scalar,
            block,
            8,
            num_warps=8 if block >= 16384 else min(16, max(4, block // 1024)),
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["top_p_renorm"]
