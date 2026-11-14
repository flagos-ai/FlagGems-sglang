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

"""Triton implementation of diffusion/ltx2_split_rotary (GCU-adapted).

Semantics (mirrors the reference): half-split rotation over
``x[B, T, H*2D]`` with ``cos/sin[B, H, T, D]``::

    out[b,t,h*2D+d]     = bf16(a * cos[b,h,t,d]) - z * sin[b,h,t,d]
    out[b,t,h*2D+D+d]   = bf16(z * cos[b,h,t,d]) + a * sin[b,h,t,d]
    a = x[b,t,h*2D+d],  z = x[b,t,h*2D+D+d]

with the deliberate bf16 rounding of the first product (the reference
PyTorch order) and everything else in float32.

GCU adaptation notes (2026-11-08 toolchain, 2D-affine band added 2026-11-13):
* The generic flat kernel dispatches ceil(N/512) CTAs, each a 512-wide
  masked 2-byte computed-index op behind a 6-deep div/mod chain.  At
  T=1024 (H=16, D=64) that is 2048 CTAs whose dispatch alone costs ~105us
  (empty-kernel probe); the kernel is CTA-dispatch bound at ~100GB/s.
* Small-T fast path (bf16, contiguous, D a power of two): 1D row-batch
  rewrite (``_ltx2_rot_rows``), kept unchanged.  Each program owns TPP
  consecutive tokens x HB consecutive heads with shift-only computed
  indices (div/mod silently miscompiles); the tail token block anchors at
  ``min(pid*TPP, T-TPP)`` (idempotent overlap, no masks anywhere); the
  deliberate product rounding uses native bf16 multiplies.
* 2D-affine band (``_ltx2_rot_tiles``, added 2026-11-13, T >= 256,
  H <= 128, D <= 64): measured on the same shapes, the row-batch
  computed-index kernel is limited by two effects the small-T sweep could
  not see: (a) CTA dispatch is paid per 48-CTA wave -- an empty kernel
  costs ~2.4us for grid<=48 but ~7.4us at grid 128 (3 waves); (b) 1D
  computed-index memory ops trail 2D affine tiles with identical segment
  layout (x/out 8MB copy: 18.6us computed vs 12.9us affine at grid 32).
  The tile kernel loads/stores a/z/c/s/y as [TPP, D] 2D affine tiles
  (one head per program, HB-style head batching removed -- HB=2 measured
  +20%), num_warps=1 (nw2 +20%, nw4 +2.7x at T=1024), TPP clamped to
  [64, 512] with grid ~(32-64) CTAs.  Joint [TPP, 2D] 256B-row loads with
  in-register deinterleave (reshape/permute/split or join) were tried and
  are catastrophically slow on this backend (333-386us, ~15x) -- register
  layout conversion does not follow the cheap-tl.trans dot-tile
  precedent; the 128B-row [TPP, D] form is the structural floor for
  half-split pairing without data movement.
* Dispatch (kernel mode, median, H=16 D=64): T=64 4.6us / T=512 9.6us
  (-32%) / T=1024 13.8us (-38% vs 22.3) / T=2048 23.4us (1.58x) /
  T=4096 35.2us (1.92x) / T=8192 63.6us (2.03x vs 129).  T <= 128 keeps
  the row-batch kernel (tile per-op overhead dominates tiny T: T=64
  affine 6.4us vs row-batch 4.6us).
* Everything else (other dtypes, non-contiguous inputs, non-power-of-two
  D, oversized tensors) falls back to the original generic kernel below.
"""

import torch
import triton
import triton.language as tl

BLK = 512


@triton.jit
def _ltx2_rot_rows(
    X,
    C,
    S,
    O,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    TPP: tl.constexpr,
    HB: tl.constexpr,
    LOG_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = tl.program_id(2)
    N_M: tl.constexpr = TPP * D
    XT: tl.constexpr = H * 2 * D
    t0 = pid_t * TPP
    t0 = tl.minimum(t0, T - TPP)
    m = tl.arange(0, N_M)
    tloc = m >> LOG_D
    xoff = m + tloc * (XT - D)
    h0 = pid_h * HB
    xb = X + (b * T + t0) * XT + h0 * (2 * D)
    ob = O + (b * T + t0) * XT + h0 * (2 * D)
    cb = C + (b * H + h0) * T * D + t0 * D
    sb = S + (b * H + h0) * T * D + t0 * D
    for hh in tl.static_range(HB):
        a = tl.load(xb + hh * (2 * D) + xoff)
        z = tl.load(xb + hh * (2 * D) + D + xoff)
        c = tl.load(cb + hh * (T * D) + m)
        s = tl.load(sb + hh * (T * D) + m).to(tl.float32)
        r1 = a * c
        r2 = z * c
        y1 = r1.to(tl.float32) - z.to(tl.float32) * s
        y2 = r2.to(tl.float32) + a.to(tl.float32) * s
        tl.store(ob + hh * (2 * D) + xoff, y1)
        tl.store(ob + hh * (2 * D) + D + xoff, y2)


@triton.jit
def _ltx2_rot_tiles(
    X,
    C,
    S,
    O,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    TPP: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    XT: tl.constexpr = H * 2 * D
    t0 = pid_t * TPP
    t0 = tl.minimum(t0, T - TPP)
    tt = t0 + tl.arange(0, TPP)
    dd = tl.arange(0, D)
    xh = X + b * T * XT + h * (2 * D) + tt[:, None] * XT
    ch = C + (b * H + h) * T * D + tt[:, None] * D
    sh = S + (b * H + h) * T * D + tt[:, None] * D
    a = tl.load(xh + dd[None, :])
    z = tl.load(xh + D + dd[None, :])
    c = tl.load(ch + dd[None, :])
    s = tl.load(sh + dd[None, :]).to(tl.float32)
    r1 = a * c
    r2 = z * c
    y1 = r1.to(tl.float32) - z.to(tl.float32) * s
    y2 = r2.to(tl.float32) + a.to(tl.float32) * s
    oh = O + b * T * XT + h * (2 * D) + tt[:, None] * XT
    tl.store(oh + dd[None, :], y1)
    tl.store(oh + D + dd[None, :], y2)


@triton.jit
def _rot(
    X,
    C,
    S,
    O,
    N: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    q = i // D
    d = i % D
    h = q % H
    row = q // H
    b = row // T
    t = row % T
    xo = q * (2 * D) + d
    co = ((b * H + h) * T + t) * D + d
    a = tl.load(X + xo, m, eviction_policy="evict_last").to(tl.float32)
    z = tl.load(X + xo + D, m, eviction_policy="evict_last").to(tl.float32)
    c = tl.load(C + co, m, eviction_policy="evict_last").to(tl.float32)
    s = tl.load(S + co, m, eviction_policy="evict_last").to(tl.float32)
    y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
    y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
    tl.store(O + xo, y1, m, eviction_policy="evict_first")
    tl.store(O + xo + D, y2, m, eviction_policy="evict_first")


def _prev_pow2(value):
    return 1 << (value.bit_length() - 1)


def ltx2_split_rotary(x, cos, sin):
    b, h, t, d = cos.shape
    out = torch.empty_like(x)
    n = b * t * h * d
    if n:
        if (
            x.dtype is torch.bfloat16
            and d & (d - 1) == 0
            and x.numel() < (1 << 31)
            and h <= 256
            and b <= 65535
            and x.is_contiguous()
            and cos.is_contiguous()
            and sin.is_contiguous()
        ):
            if t >= 256 and h <= 128 and d <= 64:
                tpp_t = min(max(_prev_pow2(t) // 2, 64), 512)
                ntb_t = (t + tpp_t - 1) // tpp_t
                if ntb_t <= 65535 and ntb_t * h * b >= 16:
                    _ltx2_rot_tiles[(ntb_t, h, b)](
                        x,
                        cos,
                        sin,
                        out,
                        t,
                        H=h,
                        D=d,
                        TPP=tpp_t,
                        num_warps=1,
                    )
                    return out
            tpp = min(max(4096 // d, 1), _prev_pow2(t))
            hb = 2 if h % 2 == 0 else 1
            ntb = (t + tpp - 1) // tpp
            if ntb * (h // hb) * b < 32:
                hb = 1
            while ntb * h * b < 32 and tpp > 1:
                tpp //= 2
                ntb = (t + tpp - 1) // tpp
            if ntb <= 65535:
                _ltx2_rot_rows[(ntb, h // hb, b)](
                    x,
                    cos,
                    sin,
                    out,
                    t,
                    H=h,
                    D=d,
                    TPP=tpp,
                    HB=hb,
                    LOG_D=d.bit_length() - 1,
                    num_warps=4,
                )
                return out
        _rot[((n + BLK - 1) // BLK,)](
            x, cos, sin, out, n, t, h, d, BLOCK=BLK, num_warps=2
        )
    return out


__all__ = ["ltx2_split_rotary"]
