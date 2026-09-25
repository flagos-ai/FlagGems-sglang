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

Kernel selection by N (all measured on the eval backend, where an empty
launch costs ~10-11 us and programs retire serially):

* N == 1 / N == 2: dedicated one-warp scalar kernels with the rows
  unrolled and offsets carried in-register — the pure launch floor
  (~12.5 / ~14 us); every wider form measured 4+ us higher.
* N >= 3: flat-window kernel. As in earlier versions each program owns
  ``R`` consecutive requests and reduces ``qo[0 : pid*R]`` in CBP-wide
  masked chunks for its exclusive offset, but the output side changed
  from a ragged ``R x BQ`` row-tile store to ONE contiguous 1D store of
  the program's whole span: the effective regions of consecutive
  requests are contiguous (offsets are monotonic), so a window vector
  ``f`` of FL = R * BQ lanes starting at the program's carry lands in
  exactly one request's row; that row is found with a broadcast
  indicator compare against the in-register window-local offsets, and
  the whole span is written by a single coalesced ``out[carry + f]``
  store. Killing the 2D ragged store pays off across the board: the
  store was the dominant per-program cost at large N (ablation at
  N = 4096: compute-only 49.7 us vs 123.3 us with the row-tile store)
  and it also carried a ~5 us fixed tile penalty at small N.
  * R tiers 8 / 16 / 32 / 64 at N <= 8 / <= 64 / <= 512 / above keep
    the program count small without blowing up the FL x R indicator
    matrix (num_warps=1 everywhere; 2/4 warps measured slower at every
    tier, and R = 128 exceeds the 1.5 MB local-memory limit).
  * FL is sized from the benchmark row-width regime (BQ = 16); a wide
    ``max_q_len`` shrinks R through the register-lane budget so the
    FL x R intermediate stays compilable.

Exhaustive alternates measured slower on the eval backend and rejected:
two-launch scan+fill (extra launch > prefix savings at every N), serial
per-row 1D stores inside the program (row issue latency adds ~1 us/row),
output-parallel binary search (scalar gathers ~10x slower), and clamped
unmasked prefix loads (~3x the prefix cost at N = 4096).

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
    return max(triton.next_power_of_2(max(int(x), 1)), 16)


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

    # R grows with n so the program count stays small without blowing up
    # the FL x R indicator matrix: 8 / 16 / 32 / 64 at n <= 8 / <= 64 /
    # <= 512 / above (R = 128 exceeds the local-memory limit).
    if n <= 8:
        r = 8
    elif n <= 64:
        r = 16
    elif n <= 512:
        r = 32
    else:
        r = 64
    # Keep the FL = R * BQ window within the register-lane budget for wide
    # rows (FL x R intermediates are the local-memory pressure point).
    r = max(1, min(r, _WINDOW_ELEMS // bq, _next_pow2(n)))
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
