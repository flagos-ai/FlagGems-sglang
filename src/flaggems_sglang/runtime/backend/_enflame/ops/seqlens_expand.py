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

"""Expand per-request (qo_len, kv_len) pairs into per-query-token KV lengths.

    out[offset[i] + j] = max(kv_len[i] - qo_len[i] + 1 + j, 0),  j < qo_len[i]
    offset = exclusive_cumsum(extend_seq_lens)

The ``clamp(..., min=0)`` is load-bearing: DP-padded / idle rows may have
``kv_len < qo_len`` and downstream consumers read these lengths as uint32,
so a negative value would turn into ~4e9 tokens and cause illegal access.

Pure Triton, portable across backends (device is taken from the input
tensors, following the runtime-selected ``flaggems_sglang.device``). The
reference implementation is a Python loop with a device sync per request
(``int(extend_seq_lens[i])``) plus 2-3 kernel launches per request; this
implementation is always a single launch.

Kernel selection by N (latencies: do_bench warmup 1000 ms / rep 100 ms
median on gcu400, 2026-11-13; an empty one-store launch floors at
2.44/2.46/2.71/3.06/4.73 us for grid 1/8/32/48/64 — the device retires
at most 48 programs per wave, so any grid above that buys a full extra
wave of latency):

* N == 1 / N == 2: dedicated one-warp scalar kernels with the rows
  unrolled and offsets carried in-register — 3.07 us at N=1 against a
  2.44 us pure launch floor; every wider form measured 4+ us higher.
* 3 <= N <= 8: per-request kernel (grid=(N,)). Each program re-derives
  its exclusive offset with ONE wide [NB] load of all query lengths plus
  a masked sum — no prefix loop, no scan intrinsic, no indicator matrix
  (measured on gcu400: the flat kernel with a dummy store still costs
  5.8/6.5 us at n=8/64 while its real work is only ~0.2/1.4 us; the
  fixed part is ``tl.cumsum`` plus the FL x R indicator structure).
* 9 <= N <= 128: pair kernel (grid=(ceil(N/2),)) — two consecutive
  requests per program. The exclusive offset still comes from one wide
  [NB] masked sum, but BOTH rows are written by a single fused [2*BQ]
  store over their contiguous span, row membership picked by the
  scalar compare ``j < qo0`` — still no cumsum, no indicator matrix.
  Pairing matters because of the 48-per-wave retire limit: the
  one-program-per-request form pays a second wave from n=49 on, while
  the second row's scalar loads and the wider store ride the first
  row's latency chain almost for free. Measured: 6.4 -> 4.7 us at
  n=64, 8.7 -> 4.7 us at n=96 and 8.6 -> 6.4 us at n=128 (ties below
  n=32, hence the 8-boundary of this arm). Small-R flat variants are
  NOT the tool in this range: ``tl.cumsum`` over fewer than 16 lanes
  lowers to a pathological path on this backend — flat R in
  {1,2,4,8} measured 15-100 us at n in {8..512} where R in {16,32}
  measured 7.3-8.9 us.
* N > 128: flat-window kernel. As in earlier versions each program owns
  ``R`` consecutive requests and reduces ``qo[0 : pid*R]`` in CBP-wide
  masked chunks for its exclusive offset, but the output side is ONE
  contiguous 1D store of the program's whole span: the effective
  regions of consecutive requests are contiguous (offsets are
  monotonic), so a window vector ``f`` of FL = R * BQ lanes starting
  at the program's carry lands in exactly one request's row; that row
  is found with a broadcast indicator compare against the in-register
  window-local offsets, and the whole span is written by a single
  coalesced ``out[carry + f]`` store. Killing the 2D ragged store pays
  off across the board (ablation at N = 4096: compute-only 49.7 us vs
  123.3 us with the row-tile store).
  * R tiers 16 / 32 at N <= 512 / above, re-scanned 2026-11-13 after
    the BQ floor dropped 16 -> 8 (2026-11-08 halved every row's lane
    count but kept the old 32/64 tiers): with BQ=8 the FL x R
    indicator is the dominant per-program cost, so the old tiers
    bought a small grid while paying a 4x-bigger matrix per program.
    Measured: 8.9 -> 7.9 us at n=512 (R 32 -> 16), 9.6 vs 13.8/16.0 us
    at n=1024 and 17.5 vs 22.4/30.1 us at n=2048 (R 32 beats both R 16
    and the old R 64), and 47.3 -> 30.3 us at n=4096 (R 64 -> 32).
    num_warps=1 everywhere (2 warps measured slower at every form:
    perreq/pair/flat all degrade, e.g. flatR16@512 7.9 -> 12.8 us);
    R = 128 exceeds the 1.5 MB local-memory limit.
  * FL is sized from the benchmark row-width regime; a wide
    ``max_q_len`` shrinks R through the register-lane budget so the
    FL x R intermediate stays compilable. On top of that, the FL x R
    indicator matrix is capped at 32768 int32 elements by an explicit
    DSM budget: measured on gcu400, FL*R = 32768 compiles at
    dsm 657408 B (254 KB under the 917504 B limit) while FL*R = 65536
    requests 917632 B and fails to compile (DSM use tracks FL*R only —
    R, FL and CBP splits at the same matrix size measure the same), so
    R is halved until R * R * BQ fits. Every previously-compiling
    config keeps its exact launch shape.

BQ is ``max(next_pow2(max_q_len), 8)`` — the floor was lowered from 16
to 8 after measuring that padding q_len<=8 rows to 16 lanes doubles the
indicator work for nothing: with bench-core inputs (max_q = 8) the flat
kernel drops 10.4 -> 8.6 us at n = 128, 11.1 -> 8.9 us at n = 512 and
38.5 -> 30.1 us at n = 4096, and the n == 1 scalar kernel
2.97 -> 2.74 us. 8 is kept as the floor because narrower stores were
never exercised and 8 lanes already cover every real extend length
regime seen in the eval shapes.

Exhaustive alternates measured slower on the eval backend and rejected:
two-launch scan+fill (extra launch > prefix savings at every N), serial
per-row 1D stores inside the program (row issue latency adds ~1 us/row),
output-parallel binary search (scalar gathers ~10x slower), clamped
unmasked prefix loads (~3x the prefix cost at N = 4096), deriving the
row scalars from the wide load with a where+sum reduction instead of two
scalar loads (4.2 -> 4.8 us at n=8 perreq, 8.8 us at n=64), and
num_warps 2/4 on every arm (perreq@64 6.4 -> 8.0/13.7 us, pair@64
4.7 -> 5.4 us, flatR16@512 7.9 -> 12.8 us).

Dtype handling: int32 inputs go straight to the kernel. int64 inputs are
narrowed with a host-side value cast (``.to(torch.int32)``) — exact for
the length domain (non-negative < 2**31) and independent of backend
64-bit word layout; only the correctness-only int64 path pays the extra
launches, benchmark shapes are int32.

No ``@triton.autotune``: the autotuner's per-call key lookup costs ~9 us
of host time here, which is a large fraction of the budget of this
launch-latency-bound op. Launch shapes are picked by plain size branches
instead.
"""

import torch
import triton
import triton.language as tl

# Cap on the prefix-reduction chunk width (int32 lanes held at once).
_MAX_PREFIX_CHUNK = 4096
# Window register-lane budget: FL = R * BQ lanes per program must stay at
# or below this, so a large max_q_len shrinks the requests-per-program
# width (the FL x R indicator intermediate is the real pressure point).
_WINDOW_ELEMS = 8192
# DSM budget for the FL x R indicator matrix (int32 intermediates),
# measured on gcu400: FL*R = 32768 -> dsm 657408 B (fits 917504 B with
# ~254 KB margin); FL*R = 65536 -> dsm 917632 B > 917504 B, a hard
# compile-time OutOfResources. DSM tracks the FL*R product only (R/FL/CBP
# splits at equal matrix size measure identically), so halving R until
# R * R * BQ fits is the minimal-intervention fix.
_IND_MATRIX_ELEMS = 32768


@triton.jit
def _seqlens_expand_one_kernel(
    qo_ptr,  # int32 [N] query lengths
    kv_ptr,  # int32 [N] kv lengths
    out_ptr,  # int32 [total_len] output
    BQ: tl.constexpr,  # power-of-2 >= max_q_len, row write width
):
    """Single-request case: one program, one warp, scalar offsets."""
    qo = tl.load(qo_ptr)
    kv = tl.load(kv_ptr)
    j = tl.arange(0, BQ)
    vals = tl.maximum(kv - qo + 1 + j, 0)
    tl.store(out_ptr + j, vals, mask=j < qo)


@triton.jit
def _seqlens_expand_two_kernel(
    qo_ptr,  # int32 [2] query lengths
    kv_ptr,  # int32 [2] kv lengths
    out_ptr,  # int32 [total_len] output
    BQ: tl.constexpr,  # power-of-2 >= max_q_len, row write width
):
    """Two-request case: one program, one warp, both rows unrolled."""
    qo0 = tl.load(qo_ptr)
    kv0 = tl.load(kv_ptr)
    j = tl.arange(0, BQ)
    tl.store(out_ptr + j, tl.maximum(kv0 - qo0 + 1 + j, 0), mask=j < qo0)
    qo1 = tl.load(qo_ptr + 1)
    kv1 = tl.load(kv_ptr + 1)
    tl.store(out_ptr + qo0 + j, tl.maximum(kv1 - qo1 + 1 + j, 0), mask=j < qo1)


@triton.jit
def _seqlens_expand_perreq_kernel(
    qo_ptr,  # int32 [N] query lengths
    kv_ptr,  # int32 [N] kv lengths
    out_ptr,  # int32 [total_len] output
    n,  # number of requests
    NB: tl.constexpr,  # power-of-2 >= n, whole-array load width
    BQ: tl.constexpr,  # power-of-2 >= max_q_len, row write width
    NOMASK: tl.constexpr,  # NB == n: the whole-array load needs no mask
):
    """One program per request: the exclusive offset comes from a single
    wide [NB] load of all query lengths masked-summed below ``pid`` — no
    prefix loop, no scan intrinsic, no indicator matrix, so the kernel's
    fixed cost stays at the plain load/store floor. When n is a power of
    two the load is emitted unmasked (a true mask is not folded away by
    the backend and measurably slows the kernel: 4.6 -> 4.1 us at n=8)."""
    i = tl.program_id(0)
    idx = tl.arange(0, NB)
    if NOMASK:
        allqo = tl.load(qo_ptr + idx)
    else:
        allqo = tl.load(qo_ptr + idx, mask=idx < n, other=0)
    pre = tl.sum(tl.where(idx < i, allqo, 0))
    qo = tl.load(qo_ptr + i)
    kv = tl.load(kv_ptr + i)
    j = tl.arange(0, BQ)
    tl.store(out_ptr + pre + j, tl.maximum(kv - qo + 1 + j, 0), mask=j < qo)


@triton.jit
def _seqlens_expand_pair_kernel(
    qo_ptr,  # int32 [N] query lengths
    kv_ptr,  # int32 [N] kv lengths
    out_ptr,  # int32 [total_len] output
    n,  # number of requests
    NB: tl.constexpr,  # power-of-2 >= n, whole-array load width
    BQ: tl.constexpr,  # power-of-2 >= max_q_len, one row's write width
    BQ2: tl.constexpr,  # 2 * BQ, fused two-row store width
    NOMASK: tl.constexpr,  # NB == n: the whole-array load needs no mask
):
    """Two consecutive requests per program. The exclusive offset comes
    from the same single wide [NB] masked sum as the per-request kernel;
    both rows are then written by ONE fused [BQ2] store over their
    contiguous span — lane j < qo0 belongs to row i, the rest to row
    i+1. This halves the program count without any scan or indicator
    matrix, which matters because the device retires at most 48
    programs per wave: the one-program-per-request form pays a second
    wave from n=49 on, while the second row's scalar loads and the
    wider store ride the first row's latency chain almost for free
    (6.4 -> 4.7 us at n=64, do_bench median, gcu400). For odd N the
    last program's second row is masked out (live1)."""
    i = 2 * tl.program_id(0)
    idx = tl.arange(0, NB)
    if NOMASK:
        allqo = tl.load(qo_ptr + idx)
    else:
        allqo = tl.load(qo_ptr + idx, mask=idx < n, other=0)
    pre = tl.sum(tl.where(idx < i, allqo, 0))
    live1 = i + 1 < n
    qo0 = tl.load(qo_ptr + i)
    kv0 = tl.load(kv_ptr + i)
    qo1 = tl.load(qo_ptr + i + 1, mask=live1, other=0)
    kv1 = tl.load(kv_ptr + i + 1, mask=live1, other=0)
    j = tl.arange(0, BQ2)
    inr0 = j < qo0
    v0 = tl.maximum(kv0 - qo0 + 1 + j, 0)
    v1 = tl.maximum(kv1 - qo1 + 1 + (j - qo0), 0)
    tl.store(out_ptr + pre + j, tl.where(inr0, v0, v1), mask=j < qo0 + qo1)


@triton.jit
def _seqlens_expand_flat_kernel(
    qo_ptr,  # int32 [N] query lengths
    kv_ptr,  # int32 [N] kv lengths
    out_ptr,  # int32 [total_len] output
    n,  # number of requests
    R: tl.constexpr,  # requests per program (power of 2)
    FL: tl.constexpr,  # flat window width = R * BQ (power of 2)
    CBP: tl.constexpr,  # prefix-reduction chunk width (power of 2)
):
    pid = tl.program_id(0)
    my_start = pid * R
    # Exclusive offset of my first request = sum(qo[0 : my_start]), reduced
    # in CBP-wide chunks so any n works without a scratch cumsum buffer.
    carry = tl.zeros((), dtype=tl.int32)
    pa = tl.arange(0, CBP)
    for start in range(0, my_start, CBP):
        x = tl.load(qo_ptr + start + pa, mask=(start + pa) < my_start, other=0)
        carry += tl.sum(x)
    # My chunk's requests and their window-local exclusive offsets.
    idx = my_start + tl.arange(0, R)
    live = idx < n
    qo = tl.load(qo_ptr + idx, mask=live, other=0)
    kv = tl.load(kv_ptr + idx, mask=live, other=0)
    off = tl.cumsum(qo, 0) - qo
    # Flat window: lane f of the program's span lands in the unique
    # request whose [off, off + qo) contains it — one broadcast compare
    # against the window-local offsets, then ONE contiguous store.
    f = tl.arange(0, FL)
    fm = f[:, None]
    oml = off[None, :]
    qom = qo[None, :]
    kvm = kv[None, :]
    ind = (fm >= oml) & (fm < oml + qom)
    vals = tl.sum(tl.where(ind, tl.maximum(kvm - qom + 1 + fm - oml, 0), 0), 1)
    span = tl.sum(qo)
    tl.store(out_ptr + carry + f, vals, mask=f < span)


def _next_pow2(x):
    return max(triton.next_power_of_2(max(int(x), 1)), 8)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    out = torch.empty(
        total_len, dtype=torch.int32, device=extend_seq_lens.device
    )
    n = extend_seq_lens.numel()
    if n == 0 or total_len <= 0:
        return out

    if extend_seq_lens.dtype == torch.int64 or seq_lens.dtype == torch.int64:
        # Value-preserving narrow: exact for the length domain (0 <= v <
        # 2**31) and independent of backend 64-bit storage layout.
        qo = extend_seq_lens.to(torch.int32)
        kv = seq_lens.to(torch.int32)
    else:
        qo = extend_seq_lens
        kv = seq_lens

    bq = _next_pow2(max_q_len)
    if n == 1:
        # Pure launch floor: one program, one warp, scalar offsets.
        _seqlens_expand_one_kernel[(1,)](qo, kv, out, BQ=bq, num_warps=1)
        return out
    if n == 2:
        # Same launch-floor treatment: both rows unrolled, offsets carried
        # in-register.
        _seqlens_expand_two_kernel[(1,)](qo, kv, out, BQ=bq, num_warps=1)
        return out

    if n <= 8:
        # Per-request arm for the sizes where both forms tie: one tiny
        # program per request, single wide [NB] masked-sum prefix (the
        # flat kernel's cumsum + indicator fixed cost dominates its
        # useful work here; dummy-store floor 5.8/6.5 us at n=8/64 vs
        # ~0.2/1.4 us of real work).
        nb = _next_pow2(n)
        _seqlens_expand_perreq_kernel[(n,)](
            qo,
            kv,
            out,
            n,
            NB=nb,
            BQ=bq,
            NOMASK=(nb == n),
            num_warps=1,
        )
        return out

    if n <= 128 and 2 * bq <= _WINDOW_ELEMS:
        # Pair arm (see kernel docstring): halving the program count cuts
        # whole waves off the 48-per-wave retire limit (one wave up to
        # n=96, two at n=128 where per-request needs three); wins from
        # n=32 up, ties below. The 2*BQ window cap keeps extreme
        # max_q_len shapes on the flat arm. The O(N^2) qo re-read is
        # bounded by 128 x 128 int32 = 64 KB total traffic.
        nb = _next_pow2(n)
        _seqlens_expand_pair_kernel[(triton.cdiv(n, 2),)](
            qo,
            kv,
            out,
            n,
            NB=nb,
            BQ=bq,
            BQ2=2 * bq,
            NOMASK=(nb == n),
            num_warps=1,
        )
        return out

    # Flat-window arm. R tiers 16 / 32 at n <= 512 / above: with BQ=8
    # the FL x R indicator is the dominant per-program cost, so keep the
    # matrix small and let the grid grow (up to 48 programs retire per
    # wave; measured 8.9 -> 7.9 us at n=512 and 47.3 -> 30.3 us at
    # n=4096 against the old 32/64 tiers, re-scanned 2026-11-13 after
    # the BQ floor dropped 16 -> 8). R = 128 exceeds the
    # local-memory limit.
    if n <= 512:
        r = 16
    else:
        r = 32
    # Keep the FL = R * BQ window within the register-lane budget for wide
    # rows (FL x R intermediates are the local-memory pressure point).
    r = max(1, min(r, _WINDOW_ELEMS // bq, _next_pow2(n)))
    # Keep the FL x R indicator matrix inside the DSM budget: beyond
    # 32768 int32 elements the gcu400 compile fails with OutOfResources
    # (dsm 917632 B > 917504 B). Halving R quarters the matrix and only
    # touches configs that could not compile at all before.
    while r > 1 and r * r * bq > _IND_MATRIX_ELEMS:
        r //= 2
    cbp = min(_next_pow2(n), _MAX_PREFIX_CHUNK)
    grid = (triton.cdiv(n, r),)
    _seqlens_expand_flat_kernel[grid](
        qo,
        kv,
        out,
        n,
        R=r,
        FL=r * bq,
        CBP=cbp,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["seqlens_expand"]
