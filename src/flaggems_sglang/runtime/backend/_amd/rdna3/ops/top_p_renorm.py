# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""top_p_renorm for RDNA3 (gfx1100): exact top-p truncation and
renormalization (Triton).

Same semantics as the generic op: the pivot is the smallest probability
whose mass from below reaches 1 - p, and every probability not below the
pivot is kept. As there, one program handles one row and each pass over the
row evaluates one candidate pivot, so the time is set by the number of
passes. The generic search alternates interpolation and bisection over the
float bit patterns from [min, max]. Here the first pass also takes the mass
below three probes (1/8, 1 and 8 times 1/V), which usually brackets the
pivot within a factor of eight; the search then interpolates linearly with
the Anderson-Bjorck correction and stops as soon as an end of the bracket is
known to be the pivot. On softmax rows this takes 5.4 search passes on
average where the generic search takes 13.6, and at most 12 against 22. From
the sixth step on, every other step bisects, so the worst case stays bounded
like the generic one. Each pass accumulates per element and reduces once, in
one combined reduction. Rows longer than 65536 take 32 warps while there are
fewer rows than WGPs.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator — the whole computation is done
inside the Triton kernel.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.utils.device_info import get_sm_count

_BISECT_FROM = 6


@triton.jit
def _stats_combine(s0, h0, a0, b0, c0, s1, h1, a1, b1, c1):
    return s0 + s1, tl.maximum(h0, h1), a0 + a1, b0 + b1, c0 + c1


@triton.jit
def _pass_combine(m0, lo0, up0, m1, lo1, up1):
    return m0 + m1, tl.maximum(lo0, lo1), tl.minimum(up0, up1)


@triton.jit
def _finish(lo, hi, cdf_lo, cdf_hi, lo_el, hi_el, target, done, ans):
    # Once lo is an element, mass(x < lo) < target, so lo is the pivot if
    # mass(x < lo) + lo reaches the target. Once hi is an element,
    # mass(x <= hi) - hi bounds mass(x < hi), so hi is the pivot if that
    # stays under the target.
    at_lo = (
        (done == 0)
        & lo_el
        & (cdf_lo + lo.to(tl.float32, bitcast=True) >= target)
    )
    ans = tl.where(at_lo, lo, ans)
    done = tl.where(at_lo, 1, done)
    at_hi = (
        (done == 0)
        & hi_el
        & (cdf_hi - hi.to(tl.float32, bitcast=True) < target)
    )
    ans = tl.where(at_hi, hi, ans)
    done = tl.where(at_hi, 1, done)
    met = (done == 0) & (lo >= hi)
    ans = tl.where(met, hi, ans)
    done = tl.where(met, 1, done)
    return done, ans


@triton.jit
def _top_p_renorm_kernel(
    P,
    T,
    O,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV,  # not constexpr: Triton 3.6 fails to compile the search folded at p=1
    BLOCK: tl.constexpr,
    BISECT_FROM: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    base = P + row * S0
    if IS_TENSOR:
        ti = 0 if SINGLE else row * TS
        p = tl.load(T + ti).to(tl.float32)
    else:
        p = PV.to(tl.float32)
    target = 1.0 - p
    q0 = tl.full((), 0.125 / V, tl.float32)
    q1 = tl.full((), 1.0 / V, tl.float32)
    q2 = tl.full((), 8.0 / V, tl.float32)

    total_v = tl.zeros((BLOCK,), tl.float32)
    high_v = tl.zeros((BLOCK,), tl.float32)
    m0_v = tl.zeros((BLOCK,), tl.float32)
    m1_v = tl.zeros((BLOCK,), tl.float32)
    m2_v = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + offsets
        x = tl.load(base + cols * S1, cols < V, other=0.0).to(tl.float32)
        total_v += x
        high_v = tl.maximum(high_v, x)
        m0_v += tl.where(x <= q0, x, 0.0)
        m1_v += tl.where(x <= q1, x, 0.0)
        m2_v += tl.where(x <= q2, x, 0.0)
    total, high, m0, m1, m2 = tl.reduce(
        (total_v, high_v, m0_v, m1_v, m2_v), 0, _stats_combine
    )

    # Bracket: mass(x < lo) = cdf_lo < target <= mass(x <= hi) = cdf_hi.
    lo = tl.full((), 0, tl.int32)
    hi = high.to(tl.int32, bitcast=True)
    cdf_lo = tl.full((), 0.0, tl.float32)
    cdf_hi = total
    lo_el = tl.full((), 0, tl.int1)
    hi_el = tl.full((), 1, tl.int1)
    qb0 = q0.to(tl.int32, bitcast=True)
    qb1 = q1.to(tl.int32, bitcast=True)
    qb2 = q2.to(tl.int32, bitcast=True)
    take = (m0 < target) & (qb0 + 1 > lo)
    lo = tl.where(take, qb0 + 1, lo)
    cdf_lo = tl.where(take, m0, cdf_lo)
    take = (m1 < target) & (qb1 + 1 > lo)
    lo = tl.where(take, qb1 + 1, lo)
    cdf_lo = tl.where(take, m1, cdf_lo)
    take = (m2 < target) & (qb2 + 1 > lo)
    lo = tl.where(take, qb2 + 1, lo)
    cdf_lo = tl.where(take, m2, cdf_lo)
    take = (m2 >= target) & (qb2 < hi)
    hi = tl.where(take, qb2, hi)
    cdf_hi = tl.where(take, m2, cdf_hi)
    hi_el = hi_el & ~take
    take = (m1 >= target) & (qb1 < hi)
    hi = tl.where(take, qb1, hi)
    cdf_hi = tl.where(take, m1, cdf_hi)
    hi_el = hi_el & ~take
    take = (m0 >= target) & (qb0 < hi)
    hi = tl.where(take, qb0, hi)
    cdf_hi = tl.where(take, m0, cdf_hi)
    hi_el = hi_el & ~take

    done = tl.where(target > 0, 0, 1)
    ans = lo
    done, ans = _finish(
        lo, hi, cdf_lo, cdf_hi, lo_el, hi_el, target, done, ans
    )
    w_lo = tl.full((), 1.0, tl.float32)
    w_hi = tl.full((), 1.0, tl.float32)
    last = tl.full((), 0, tl.int32)
    step = tl.full((), 0, tl.int32)
    while done == 0:
        low_value = lo.to(tl.float32, bitcast=True)
        high_value = hi.to(tl.float32, bitcast=True)
        f_lo = (cdf_lo - target) * w_lo
        f_hi = (cdf_hi - target) * w_hi
        fraction = tl.minimum(
            1.0, tl.maximum(0.0, -f_lo / tl.maximum(f_hi - f_lo, 1.0e-30))
        )
        mid = (low_value + (high_value - low_value) * fraction).to(
            tl.int32, bitcast=True
        )
        bisect = (step >= BISECT_FROM) & (((step - BISECT_FROM) & 1) == 1)
        mid = tl.where(bisect, lo + (hi - lo) // 2, mid)
        mid = tl.minimum(hi - 1, tl.maximum(lo, mid))
        pivot = mid.to(tl.float32, bitcast=True)

        mass_v = tl.zeros((BLOCK,), tl.float32)
        lower_v = tl.zeros((BLOCK,), tl.float32)
        upper_v = tl.full((BLOCK,), float("inf"), tl.float32)
        for start in range(0, V, BLOCK):
            cols = start + offsets
            x = tl.load(base + cols * S1, cols < V, other=float("inf"))
            x = x.to(tl.float32)
            below = x <= pivot
            selected = tl.where(below, x, 0.0)
            mass_v += selected
            lower_v = tl.maximum(lower_v, selected)
            upper_v = tl.minimum(upper_v, tl.where(below, float("inf"), x))
        mass, lower, upper = tl.reduce(
            (mass_v, lower_v, upper_v), 0, _pass_combine
        )

        # Anderson-Bjorck: when the same end moves twice in a row, scale the
        # other end's residual so the next interpolation leans towards it.
        enough = mass >= target
        side = tl.where(enough, 1, -1)
        same = side == last
        residual = mass - target
        d_lo = cdf_lo - target
        d_hi = cdf_hi - target
        m_hi = 1.0 - residual / tl.where(d_lo != 0, d_lo, -1.0)
        m_hi = tl.where(m_hi > 0, m_hi, 0.5)
        m_lo = 1.0 - residual / tl.where(d_hi != 0, d_hi, 1.0)
        m_lo = tl.where((m_lo > 0) & (d_hi != 0), m_lo, 0.5)
        w_hi = tl.where(enough, 1.0, tl.where(same, w_hi * m_hi, w_hi))
        w_lo = tl.where(enough, tl.where(same, w_lo * m_lo, w_lo), 1.0)
        last = side
        hi = tl.where(enough, lower.to(tl.int32, bitcast=True), hi)
        cdf_hi = tl.where(enough, mass, cdf_hi)
        hi_el = hi_el | enough
        lo = tl.where(enough, lo, upper.to(tl.int32, bitcast=True))
        cdf_lo = tl.where(enough, cdf_lo, mass)
        lo_el = lo_el | ~enough
        step += 1
        done, ans = _finish(
            lo, hi, cdf_lo, cdf_hi, lo_el, hi_el, target, done, ans
        )

    pivot = ans.to(tl.float32, bitcast=True)
    total_v = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + offsets
        x = tl.load(base + cols * S1, cols < V, other=0.0).to(tl.float32)
        total_v += tl.where(x >= pivot, x, 0.0)
    denom = tl.sum(total_v, 0)
    for start in range(0, V, BLOCK):
        cols = start + offsets
        valid = cols < V
        x = tl.load(base + cols * S1, valid, other=0.0).to(tl.float32)
        tl.store(
            O + row * V + cols, tl.where(x >= pivot, x / denom, 0.0), valid
        )


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
        if vocab > 65536 and batch < get_sm_count():
            # Each program has a WGP to itself: 32 warps hide more of the
            # latency of a long row.
            block, warps = 4096, 32
        else:
            block = min(8192, triton.next_power_of_2(vocab))
            warps = min(16, max(4, block // 512))
        _top_p_renorm_kernel[(batch,)](
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
            _BISECT_FROM,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["top_p_renorm"]
