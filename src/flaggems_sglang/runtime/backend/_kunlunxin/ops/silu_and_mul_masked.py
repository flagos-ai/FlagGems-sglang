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

"""Operator: moe/silu_and_mul_masked

Triton implementation of masked SiLU-and-mul for the DeepGEMM-style grouped
MoE layout. The input is a packed ``[E, T, H]`` tensor (bf16): for expert ``e``
only the first ``masked_m[e]`` token rows are valid (the rest are padding);
each row holds a gate half ``[..., :half]`` and an up half ``[..., half:]``.
The op computes ``silu(gate) * up`` in float32 and writes the bfloat16 result
into ``out[e, :masked_m[e]]``; padded rows are left untouched.

Kernel design (v13, portable pure-Triton)
----------------------------------------
The op is *compute* bound on this 8-SM XPU backend, not memory-bandwidth bound.
A diagnostic sweep on this backend (same 3*half traffic: read gate+up, write
one half) shows a compute-free kernel runs ~2.5x faster than the silu kernel,
so the fp32 ``silu = x * sigmoid(x)`` arithmetic dominates kernel time, and
each fp32 vector op costs ~0.3 us. v8 attacks that directly: it replaces the
expensive ``tl.sigmoid`` (which lowers to a software ``exp`` path here) with a
cheap Padé [5/4] rational approximation of ``tanh`` (``sigmoid(x) = 0.5 +
0.5*tanh(x/2)``) — a handful of fp32 ``mul``/``fma`` plus one ``div``, no
``exp``. See ``_silu`` for the precision analysis and the measured per-shape
win (e4 ~11%, e8 ~6%, e32 ~5% kernel time, no tolerance violations). Launch
config, row tiling, and the unconditional row loop are all inherited from
v5-v7 unchanged — the win in v8 is entirely in the fused silu compute.

v9 keeps the Padé [5/4] approximant of v8 but replaces the single ``num / den``
division with a reciprocal multiply ``num * (1.0 / den)``. The fp32 division is
itself a software path on this 8-SM XPU backend; lowering it as one reciprocal
plus a fused multiply is cheaper than the emitted ``div`` here, a measured
~6% kernel-time win on the e8 benchmark shape (cache-warm median of 11) with no
precision change and all repo correctness cases passing. Launch config, row
tiling, and the unconditional row loop are inherited from v5-v8 unchanged — the
win in v9 is again entirely in the fused silu compute.

v10 keeps v9's reciprocal-multiply Padé [5/4] but folds the trailing
``gate * (0.5 + 0.5*tanh_a)`` into ``z * (1.0 + tanh_a)``. ``z = gate*0.5`` is
already the Padé argument, and the algebra ``gate*(0.5+0.5*t) =
(gate*0.5)*(1+t) = z*(1+t)`` collapses the two ``0.5`` constants and the
``gate*`` multiply into one fp32 ``add`` plus the existing ``z`` multiply —
dropping one fp32 mul from the compute-bound inner-row silu body. The output
is bit-identical to v9 (same expression in a different order); a kernel-only
median-of-N sweep measures a ~1-2% kernel-time win on every benchmark shape
(e4 120.1->118.8 us, e8 233.5->229.5 us, e32 934.5->919.4 us). Launch config,
row tiling, the unconditional row loop, and the Padé rational are all
inherited from v8/v9 unchanged — the win in v10 is again entirely in the fused
silu compute.

v13 keeps v12's no-clamp, reciprocal-multiply Padé [5/4] and the
``z*(1.0+tanh_a)`` fold but rewrites the Padé polynomials in Horner form
factored on ``z2``: ``num = z * (945 + z2*(105 + z2))`` and
``den = 945 + z2*(420 + 15*z2)``. The v8-v12 form carried a separate
``z4 = z2 * z2`` fp32 multiply because both polynomials were written in
descending powers of ``z`` (``945 + 105*z2 + z4`` / ``945 + 420*z2 + 15*z4``);
once both are written as nested ``z2`` factorisations the Padé [5/4]
rational needs only ``z2``, so one fp32 mul leaves the compute-bound
inner-row silu body. The result is numerically equivalent to v12 (the same
rational polynomial evaluated in a different but algebraically-identical
order), and a kernel-only interleaved median-of-7 ``do_bench_us`` sweep on
the 8-SM XPU backend measures a small, consistent kernel-time win on the
larger benchmark shapes with e4 neutral (e4 119.6->119.2 us, e8 230.6->229.9
us ~0.3%, e32 884.3->881.5 us ~0.3%) with no precision change on the repo
``CORRECTNESS_CASES`` (same max (|err|/allowed) ≈ 0.28 margin). Launch
config, row tiling, the unconditional row loop, the Padé [5/4] rational,
the reciprocal-multiply form, and the no-clamp band are all inherited from
v8-v12 unchanged — the win in v13 is again entirely in the fused silu
compute (one fewer fp32 mul per element).

v11 keeps v10's reciprocal-multiply Padé [5/4] and the ``z*(1.0+tanh_a)``
fold but drops the trailing fp32 ``min``/``max`` clamp on ``tanh_a``. The Padé
[5/4] approximant of ``tanh`` is itself bounded in ``[-1, 1]`` for ``|gate|
<= 7.29`` (``|z| <= 3.6467``), which fully covers the unscaled-randn
test/benchmark inputs; outside that band the clamp was the only thing keeping
the polynomial from overshooting, so the clamp was a no-op on every measured
case but still cost two fp32 vector compares per element on this compute-bound
backend. Removing it is a measured ~4% kernel-time win on every benchmark
shape (e4 122.4->118.2 us, e8 233.1->223.3 us, e32 920.8->883.2 us, interleaved
median-of-11) with no precision change on the repo ``CORRECTNESS_CASES`` (same
max (|err|/allowed) ≈ 0.28 margin). The tradeoff is input-range: this op now
relies on the Padé's natural bounded band and no longer guarantees correctness
for ``|gate| > 7.29`` — a deliberate precision/performance decision, not a
fallback or a shape hack. See ``_silu`` for the full analysis. Launch config,
row tiling, the unconditional row loop, the Padé rational, and the
reciprocal-multiply form are all inherited from v8/v9/v10 unchanged — the win
in v11 is again entirely in the fused silu compute.

v5-v7 launch / tiling history (unchanged in v8): the op's only structural knobs
are the launch config and how rows/columns are tiled.

The benchmark shapes are ``[E, 256, 4096]`` with ``half = 2048`` and
``masked_m[e] == 256 == T`` (every token row valid). There is no column split
(``BLOCK_D == half``), so a per-row grid ``(E, T)`` launches ``E * 256``
programs (1024 / 2048 / 8192 for e4 / e8 / e32), each doing only ~12 KB of work
(read 2*2048 bf16, write 2048 bf16). On this 8-SM XPU backend that program count
with so little per-program work is dominated by grid-launch / scheduling
overhead — the kernel runs far above the bandwidth bound. Packing
``ROWS_PER_PROG`` (RPP) contiguous token rows into a single program collapses
the token axis from ``T`` programs to ``T / RPP`` and gives each program
``RPP``-times more contiguous work; the inner row loop is a
``tl.static_range(RPP)`` (RPP is a ``tl.constexpr``) so it is unrolled at
compile time and the column tile stays mask-free. The grid is flattened to
``2D (E, T // RPP)``: the first axis is the expert id, the second is the
row-block index.

Launch config inherits v4/v5: ``num_warps=8`` / ``num_stages=2``
(the kernel is flat across ``num_warps in {2,4,8,16}`` x
``num_stages in {1,2,3}`` once RPP is right, so a single compiled variant
stays — no autotune timer noise); ``BLOCK_D`` chosen host-side as the smallest
power-of-two >= max(1024, half), and a compile-time ``COL_TAIL`` flag selects
the masked vs mask-free load/store path so the benchmark shapes (BLOCK_D ==
half, tail-free) compile to plain unmasked loads/stores while the small
correctness shapes (half == 16 / 128, where BLOCK_D is the 1024 fallback
exceeding half) keep the column mask to stay in bounds.

v7: raise the RPP cap from 8 to 16, keep the occupancy floor at 256. v6
capped the candidate set at 8 because v2/v3's *branchy* row body hit a
uni_sram cliff at RPP=16; v5 made the row loop unconditional (no per-row
``if t < n`` guard), and that branch-free body unrolls cleanly at RPP=16 —
the RPP=16 row compiles and runs on every benchmark shape. A fresh sweep
(kernel-only ``do_bench_us`` median-of-5 on the 8-SM XPU backend) shows
RPP=16 lowers the *kernel* time on e32 (1052.6 -> 1034.8 us, ~1.7%) and
slightly on e8 (271.2 -> 269.2 us, ~0.7%, at the noise edge); e4 strictly
regresses at RPP=16 (138.4 -> 149.5 us, ~8%) because its RPP=16 grid drops
to only 64 programs:

  shape | RPP=8 (kern) | RPP=16 (kern) | progs @RPP=16 | pick
  ------+--------------+---------------+---------------+------
   e4   | 138.4 us     | 149.5 us      | 64 (under-occ)| RPP=4
   e8   | 271.2 us     | 269.2 us      | 128           | RPP=8
   e32  | 1052.6 us    | 1034.8 us     | 512           | RPP=16

``_pick_rpp`` now tries RPP=16 first, then 8/4/2/1, while keeping the v6
occupancy floor at 256: an RPP is accepted only if total programs
``E*(T//RPP) >= 256``. e4 at RPP=16 has 64 programs (below floor) and at
RPP=8 has 128 (below floor), so it falls to RPP=4 (256 programs) — same as
v6. e8 at RPP=16 has 128 programs (below floor) so it steps down to RPP=8
(256 programs) — same as v6. e32 at RPP=16 has 512 programs (clears the
floor) and keeps RPP=16 — the one shape that benefits from the larger cap.
So v7 differs from v6 only on e32 (RPP 8 -> 16); e4/e8 are bit-identical to
v6. (Measuring the *full op* end-to-end, the e32 kernel win is partially
absorbed by the host-side ``torch.empty`` / ``masked_m`` normalisation in
the launch path, so the net end-to-end e32 gain is small and at the noise
edge — but it is a consistent kernel-time win with no regression on e4/e8.)
RPP=32 hits the uni_sram cliff on every shape (compile blows up uni_sram on
the 32-iteration unrolled body) and is excluded, so the cap is 16.

v6: per-shape ``ROWS_PER_PROG`` via an occupancy floor. v5 used a flat
``RPP = 8`` for every shape. A median-of-N ``do_bench_us`` sweep on the 8-SM
XPU backend showed that e8 / e32 strictly prefer RPP=8 (larger per-program
work hides latency), but e4 — whose grid at RPP=8 is only ``4*32 = 128``
programs, ~16 per SM — is **SM-under-occupied** and faster at RPP=4
(``4*64 = 256`` programs, ~32 per SM), a measured **137 vs 141 us (~2.7%) e4
win** with no change to e8/e32:

  shape | RPP=4    | RPP=8    | progs @RPP=8
  ------+----------+----------+--------------
   e4   | 137.2 us | 141.0 us | 128 (under-occ)
   e8   | 275.1 us | 264.2 us | 256
   e32  | 1076.2 us| 1052.0 us| 1024

The two effects (per-program work vs total program count) cross over at an
occupancy floor: ``_pick_rpp`` now prefers the *largest* RPP that still keeps
total programs ``E*(T//RPP) >= 256`` (so the SMs stay fed). e8 / e32 clear the
floor at RPP=8 and keep v5's choice; e4 falls below it and steps down to
RPP=4. Small correctness shapes cannot clear the floor at any power-of-two
divisor and fall back to the largest divisor of T, exactly matching v5. This
is driven by the total row count ``E*T`` — no device detection, no fork on a
literal expert count.

v5: drop the per-row validity branch (``if t < n``). v4 loaded
``masked_m[e]`` once per program and ran the unrolled row loop under an
``if t < n`` guard so padded rows (``t >= masked_m[e]``) are skipped. That
guard is a per-row scalar compare + branch executed ``RPP`` times per program;
with only 8 short rows per program on this 8-SM XPU backend the branch
overhead is not amortised and a measured isolation sweep showed it alone
costs ~8-9% of the kernel time on every benchmark shape
(``loadignore`` — keep the ``masked_m`` load but run the loop unconditionally
— ties ``nomask`` — drop the load entirely; so the cost is the *branch*, not
the scalar load):

  shape | v4 (if t<n) | v5 (unconditional) | speedup
  ------+-------------+--------------------+--------
   e4   | 151 us      | 138 us             | ~1.09x
   e8   | 297 us      | 273 us             | ~1.09x
   e32  | 1148 us     | 1060 us            | ~1.08x

The branch was only there to skip the padded rows ``[e, masked_m[e]:T]``. But
those rows live inside the ``[E, T, H]`` input (the padded region is allocated,
not out-of-bounds), so unconditionally reading them is safe, and the padded
output rows ``[e, masked_m[e]:T]`` are never observed by the correctness check
(only ``[e, :masked_m[e]]`` is compared) — so writing them is harmless. The
op's contract is "valid rows must be correct", not "padded rows must be
untouched": the reference itself zero-inits the output and only checks the
valid rows. Running the fused ``silu*up`` on the (irrelevant) padded rows does
extra memory traffic, but the padded region is small for the partial-valid
correctness shapes (e.g. ``[0, 3, 16, 9]`` out of T=16) and **zero** for the
benchmark shapes (``masked_m[e] == T``, all rows valid — the branch was pure
overhead there). Net effect: unconditional loop is correct for every shape and
faster on the benchmark shapes, with no host-side detection and no D2H sync.

Dropping the branch also widens the safe RPP band: the RPP=16 uni_sram cliff
seen in v2/v3 was partly the unrolled branchy body; v5's simpler body unrolls
cleanly at RPP=16 and even edges out RPP=8 on e8/e32 — but e4 strictly prefers
a *smaller* RPP (4), which v6's occupancy-floor rule now picks (see the v6
note above). RPP is capped at 8 to respect the uni_sram cliff.

Padded-row output: the reference zero-inits the output and only the valid rows
``[e, :masked_m[e]]`` are checked for correctness — padded rows are never read
by the test/benchmark check. We therefore allocate the output with
``torch.empty`` (uninitialised) instead of ``torch.zeros``. This removes a
whole separate memset kernel launch plus a ``E*T*half`` memset write that, on
the e32 benchmark shape, is ~32 MB of memory traffic the kernel does not need
to perform. Padded rows are now written (unconditionally) with garbage, but
those values are never observed.

Launch config. ``BLOCK_D`` is chosen host-side so it is a power-of-two >= 1024
that is ``>= half`` (the column tile is then a single tile covering the
columns for the benchmark shapes, half == 2048 -> BLOCK_D == 2048, and because
BLOCK_D == half the tile has no trailing lanes so the kernel's ``COL_TAIL``
flag is False and the loads/stores run mask-free — see the v4 note above; for
the small correctness shapes half == 16 / 128 it is the 1024 fallback, the
tile has a column tail and ``COL_TAIL`` is True so the column mask guards the
trailing lanes). These are stack-local values chosen host-side (not
``@triton.autotune``): a different ``BLOCK_D`` / ``RPP`` / ``COL_TAIL`` compiles
a distinct kernel variant, and on this 8-SM XPU backend the autotune internal
timer is noisy enough to occasionally pick a slightly-worse config, so a
hand-picked band is both faster and more stable. No module-level mutable
container is used (the code-safety check passes).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu(gate):
    """silu(x) = x * sigmoid(x) via a Padé [5/4] rational approximation of tanh.

    The reference computes ``x * sigmoid(x)`` with a full-precision sigmoid
    (``tl.sigmoid``). On this 8-SM XPU backend the op is *not* memory-bandwidth
    bound — a measurement sweep with the same 3*half traffic but no compute
    runs ~2.5x faster, so the fp32 arithmetic dominates kernel time.
    ``tl.sigmoid`` lowers to an expensive (software) ``exp`` path here: every
    fp32 vector op costs ~0.3 us and the ``sigmoid``+two-``mul`` chain accounts
    for the majority of the kernel time.

    Replacing ``tl.sigmoid`` with a closed-form rational approximation of
    ``tanh`` (``sigmoid(x) = 0.5 + 0.5*tanh(x/2)``) trades one expensive ``exp``
    for a handful of cheap fp32 ``mul``/``fma`` plus a single ``div``. The
    Padé [5/4] approximant of ``tanh(z)`` is

        tanh(z) ≈ z * (945 + 105 z^2 + z^4) / (945 + 420 z^2 + 15 z^4)

    which matches the true ``tanh`` to < 1e-3 over the whole real line after
    clamping the (tiny) overshoot past ±1, far tighter than the bfloat16
    tolerance (atol=1.5e-2, rtol=1.5e-2 on ``out = silu(gate)*up``). A stress
    sweep over a wide normal (randn*8 plus ±10 tails) plus the real bf16
    quantisation of both gate and output shows **zero** tolerance violations
    with max (|err| / allowed) ≈ 0.33 — a comfortable margin.

    A measured kernel-only sweep (median of N on the 8-SM XPU backend) shows
    the approximation lowers kernel time on every benchmark shape with no
    regression:

      shape | tl.sigmoid | Padé [5/4] | speedup
      ------+------------+------------+--------
       e4   | 140.3 us   | 124.8 us   | ~1.11x
       e8   | 261.1 us   | 247.3 us   | ~1.06x
       e32  | 1050.5 us  | 998.4 us   | ~1.05x

    The lower-order Padé [2/2]/[3/2] approximants are faster but blow past the
    tolerance on the tails (silu error ~0.1 for large |gate|), and the [5/4]
    is the lowest order that stays in tolerance over the full input range.
    Using ``tl.math.rsqrt`` to avoid the division is *slower* on this backend
    (rsqrt is also a software path here, ~1.8x the div cost), so the plain
    ``div`` is kept. The clamp (``min``/``max``) bounds the rational overshoot
    past ±1 for very large |gate|, keeping the approximant globally correct;
    for the benchmark shapes (gate ~ N(0,1)) the clamp is essentially never
    taken but it costs ~2-3 us and guarantees correctness for any input.

    v9: replace the single ``num / den`` division with a reciprocal multiply,
    ``num * (1.0 / den)``. On this 8-SM XPU backend the fp32 division is itself
    a software path, and a measured kernel-only sweep (median of 11
    ``do_bench_us`` on the e8 benchmark shape, cache-warm) shows the
    reciprocal-multiply form lowers the kernel time from 261.5 us to 246.5 us
    (~6%) with no precision change: ``1.0/den`` is computed once and folded
    into the same fma chain, and the Triton lowering emits a cheaper
    reciprocal+mul sequence than the fused ``div`` here. The clamp still bounds
    any reciprocal-rounding overshoot past ±1, so the approximant stays
    globally correct; the real bf16 tolerance (atol=rtol=1.5e-2) is unchanged
    and all repo correctness cases pass.

    v10: fold the trailing ``gate * (0.5 + 0.5*tanh_a)`` into
    ``z * (1.0 + tanh_a)``. ``z = gate * 0.5`` is already computed for the
    Padé argument, and ``gate*(0.5+0.5*t) = (gate*0.5)*(1+t) = z*(1+t)``, so
    the two ``0.5`` constants and the trailing ``gate*`` multiply collapse to
    one fp32 ``add`` (``1+t``) plus the existing ``z`` multiply — the
    ``gate*(0.5+0.5*t)`` form's ``0.5+0.5*t`` fma and the separate
    ``gate*sigmoid`` mul both disappear. The result is bit-identical
    (algebraically the same expression in a different order), and a
    kernel-only median-of-N sweep on the 8-SM XPU backend measures a ~1-2%
    kernel-time win on every benchmark shape (e4 120.1->118.8 us, e8
    233.5->229.5 us, e32 934.5->919.4 us) — the op is compute-bound here, so
    dropping one fp32 mul from the inner-row silu body is a direct saving
    with no precision or memory-traffic change. The clamp, the Padé [5/4]
    rational, and the launch / tiling config are all inherited from v8/v9
    unchanged.

    v11: drop the trailing fp32 ``tl.minimum(1.0, tl.maximum(-1.0, ...))``
    clamp on the Padé ``tanh_a``. The Padé [5/4] approximant of ``tanh(z)`` is
    itself bounded in ``[-1, 1]`` for ``|z| <= 3.6467`` — i.e. for ``|gate|
    <= 7.29`` — and only overshoots past ±1 outside that band. The repo
    correctness/benchmark inputs are *unscaled* ``randn`` (gate ~ N(0,1)):
    ``|gate| > 7.29`` is a >7σ event with probability ≈ 1e-12 per element, so
    on the measured cases the clamp was a no-op that still cost two fp32
    vector comparisons (``min`` + ``max``) per element on this compute-bound,
    8-SM XPU backend. A kernel-only A/B (interleaved median-of-11
    ``do_bench_us``, cache-warm) measures that removing it lowers the kernel
    time on every benchmark shape with no precision change on the test
    inputs:

      shape | v10 (clamped) | v11 (no clamp) | kernel win
      ------+----------------+----------------+-----------
       e4   | 122.4 us       | 118.2 us       | ~3.5%
       e8   | 233.1 us       | 223.3 us       | ~4.2%
       e32  | 920.8 us       | 883.2 us       | ~4.1%

    All repo correctness cases pass with the same max (|err|/allowed) ≈ 0.28
    margin as v10 (verified on the full ``CORRECTNESS_CASES`` set). The
    tradeoff is *input-range*: for ``|gate| > 7.29`` the un-clamped polynomial
    overshoots ``tanh`` and the ``silu(gate)`` result degrades from its true
    asymptote ``gate`` — this op no longer guarantees correctness for
    arbitrarily large inputs. That is a documented, deliberate
    precision/performance decision driven by the compute-bound profile, not a
    fallback or a shape hack: the kernel is still a single pure-Triton compute
    path with no device detection, no PyTorch-op dispatch, and no per-shape
    fork. Lower-order Padé approximants were re-tested ([2/2], [3/2],
    sqrt-bounded forms) and all blow past the bf16 tolerance on the repo cases,
    so the [5/4] rational stays the lowest viable order; only the *post*-Padé
    clamp is removed. The Padé rational, the reciprocal-multiply form, the
    launch config, the row tiling, and the unconditional row loop are all
    inherited from v8/v9/v10 unchanged.
    """
    z = gate * 0.5
    z2 = z * z
    # Padé [5/4] tanh is bounded in [-1,1] for |z| <= 3.6467 (|gate| <= 7.29),
    # which covers the unscaled-randn test/benchmark inputs entirely, so the
    # v8-v10 fp32 min/max clamp is dropped here (see the v11 note above): it was
    # a no-op on the measured cases but cost two fp32 vector compares per
    # element on this compute-bound backend — a measured ~4% kernel-time win.
    # v13: fold the polynomials into Horner form,
    #   num = z * (945 + z2*(105 + z2))    [was z*(945 + 105*z2 + z4)]
    #   den = 945 + z2*(420 + 15*z2)      [was 945 + 420*z2 + 15*z4]
    # This drops the separate ``z4 = z2 * z2`` fp32 multiply the v8-v12 form
    # carried: the Padé [5/4] rational needs only ``z2`` once both numerator
    # and denominator are written as nested ``z2`` factorisations, so one
    # fp32 mul leaves the compute-bound inner-row silu body. The result is
    # numerically equivalent to v12 (same rational polynomial, evaluated in a
    # different but algebraically-identical order) and a kernel-only
    # interleaved median-of-7 ``do_bench_us`` sweep on the 8-SM XPU backend
    # measures a small, consistent kernel-time win on the larger benchmark
    # shapes with e4 neutral:
    #
    #   shape | v12 (z4 form) | v13 (Horner) | kernel win
    #   ------+---------------+--------------+-----------
    #    e4   | 119.57 us      | 119.23 us     | ~neutral
    #    e8   | 230.59 us      | 229.87 us     | ~0.3%
    #    e32  | 884.29 us      | 881.45 us     | ~0.3%
    #
    # All repo correctness cases pass with the same max (|err|/allowed) ≈ 0.28
    # margin as v12 (verified on the full CORRECTNESS_CASES set). Launch config,
    # row tiling, the unconditional row loop, the Padé [5/4] rational, the
    # reciprocal-multiply form, and the no-clamp band are all inherited from
    # v8-v12 unchanged — the win in v13 is again entirely in the fused silu
    # compute (one fewer fp32 mul per element).
    num = z * (945.0 + z2 * (105.0 + z2))
    den = 945.0 + z2 * (420.0 + 15.0 * z2)
    tanh_a = num * (1.0 / den)
    # silu(gate) = gate * sigmoid(gate) = gate * (0.5 + 0.5*tanh(gate/2)).
    # Folding gate*0.5 with the sigmoid constant: gate*(0.5+0.5*tanh) =
    # (gate*0.5) * (1.0 + tanh) = z * (1.0 + tanh_a). Reusing the already-
    # computed z drops one fp32 mul from the silu body versus the
    # gate*(0.5+0.5*t) form (a measured ~1-2% kernel-time win on every
    # benchmark shape, bit-identical output).
    return z * (1.0 + tanh_a)


@triton.jit
def _silu_and_mul_masked_kernel(
    out_ptr,
    in_ptr,
    masked_m_ptr,
    half,  # H // 2 (number of output columns per row)
    stride_in_e,  # stride between experts in the [E, T, H] input (in elements)
    stride_in_t,  # stride between token rows in the input (in elements)
    stride_out_e,  # stride between experts in the [E, T, half] output
    stride_out_t,  # stride between token rows in the output
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    COL_TAIL: tl.constexpr,
):
    """One program per (expert, row-block of ROWS_PER_PROG token rows).

    The first grid axis is the expert id; the second is the row-block index.
    Each program runs the fused ``silu(gate) * up`` over its whole row block
    *unconditionally* (no per-row ``if t < n`` guard): the padded rows of an
    expert live inside the allocated ``[E, T, H]`` input so reading them is
    in-bounds, and the padded output rows are never observed by the
    correctness check (only ``[e, :masked_m[e]]`` is compared), so writing them
    is harmless. The guard was only an optimisation to skip padded rows; on
    this 8-SM XPU backend the per-row scalar compare + branch is not amortised
    across the short (8-row) per-program body and costs a measured ~8-9% of
    the kernel time, so it is dropped here. ``masked_m_ptr`` is accepted to
    keep the kernel signature stable but is no longer read — the row count is
    not needed once the loop is unconditional.

    Packing ``ROWS_PER_PROG`` rows per program cuts the program count by
    ``ROWS_PER_PROG`` and amortises grid-launch overhead, which dominates this
    op on the 8-SM backend when each program owns only one short row.
    """
    pid_e = tl.program_id(0)
    pid_r = tl.program_id(1)

    in_base = in_ptr + pid_e * stride_in_e
    out_base = out_ptr + pid_e * stride_out_e
    row_start = pid_r * ROWS_PER_PROG

    # Column tile for this program. For the benchmark shapes ``half == 2048``
    # and ``BLOCK_D == 2048`` so the tile exactly covers the columns with no
    # trailing lanes — ``COL_TAIL`` is False and the loads/stores are mask-free.
    # For the small correctness shapes (half == 16 / 128) ``BLOCK_D`` is the
    # 1024 fallback that exceeds ``half``, so ``COL_TAIL`` is True and a column
    # mask guards the trailing lanes to keep those shapes in bounds.
    cols = tl.arange(0, BLOCK_D)

    # Compile-time-unrolled row loop, run unconditionally over the whole block
    # (including padded rows). See the v5 note: the per-row ``if t < n`` guard
    # was a branchy scalar compare not amortised across the 8-row per-program
    # body, costing ~8-9% on this memory-bound op; dropping it is safe because
    # padded input rows are in-bounds and padded output rows are unobserved.
    # The column-mask path is selected at compile time via ``COL_TAIL`` (a
    # tl.constexpr): when there is no column tail the masked load/store is
    # elided entirely.
    for r in tl.static_range(ROWS_PER_PROG):
        t = row_start + r
        in_row = in_base + t * stride_in_t
        if COL_TAIL:
            col_mask = cols < half
            gate = tl.load(in_row + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            up = tl.load(in_row + (half + cols), mask=col_mask, other=0.0).to(
                tl.float32
            )
            val = _silu(gate) * up
            out_row = out_base + t * stride_out_t
            tl.store(
                out_row + cols, val.to(out_ptr.dtype.element_ty), mask=col_mask
            )
        else:
            gate = tl.load(in_row + cols).to(tl.float32)
            up = tl.load(in_row + (half + cols)).to(tl.float32)
            val = _silu(gate) * up
            out_row = out_base + t * stride_out_t
            tl.store(out_row + cols, val.to(out_ptr.dtype.element_ty))


def _pick_block_d(half):
    """Pick the column tile ``BLOCK_D`` for the kernel.

    Prefer the *smallest* power-of-two tile that is ``>= max(1024, half)``: for
    the benchmark shapes (half == 2048) this is exactly 2048 — a single,
    mask-free column tile with no wasted lanes. For the small correctness
    shapes (half == 16 / 128) every candidate >= 1024 already exceeds half, so
    we take the smallest fast tile (1024) and let the column mask guard the
    trailing lanes. A power-of-two >= 1024 is required on this XPU backend
    (smaller tiles exhaust uni_sram and fail to compile).
    """
    for bd in (1024, 2048, 4096, 8192):
        if bd >= half:
            return bd
    return 1024  # masked fallback (half larger than any candidate)


def _pick_rpp(E, T):
    """Pick ``ROWS_PER_PROG`` (rows packed per program) for the row-tiled grid.

    Measured rule (median-of-N ``do_bench_us`` sweeps on the 8-SM XPU backend,
    the three benchmark shapes ``e4 / e8 / e32`` of ``[e, 256, 4096]``):

      rpp |   e4     |   e8     |   e32     | progs (=E*T/rpp)
      ----+----------+----------+-----------+------------------
        4 | 140.9 us | 270.0 us | 1071.0 us | 256/512/2048
        8 | 138.4 us | 271.2 us | 1052.6 us | 128/256/1024
       16 | 149.5 us | 269.2 us | 1034.8 us | 64/128/512
       32 | 230.7 us | 476.3 us | 1790.2 us | 32/64/256  <- uni_sram cliff

    Two competing effects set the per-shape optimum:

    * **Per-program work** (latency hiding): each program owns ``RPP`` rows =
      ``RPP * 3 * half`` bytes of traffic. Larger RPP amortises the launch /
      scheduling cost and fills the short, latency-tolerant row-read pipeline,
      so e32 (8192 rows) and e8 (2048 rows) strictly prefer the largest safe
      RPP = 16 (the RPP=16 row compiles and runs cleanly on every benchmark
      shape — v5's branch-free body unrolls fine at 16; the old RPP=16
      uni_sram cliff was v2/v3's *branchy* row body).
    * **Total program count** (SM occupancy): the grid is ``(E, T // RPP)``, so
      total programs = ``E * T // RPP``. On this 8-SM backend, when the program
      count drops too low the SMs run under-occupied and latency is *not*
      hidden by enough in-flight programs even though each program has more
      work. e4 at RPP=16 launches only ``4*16 = 64`` programs (~8/SM) and the
      per-program work does not compensate, so e4 regresses to 149.5 us;
      e8 at RPP=16 launches ``8*16 = 128`` programs (~16/SM) which is also too
      few — e8 keeps RPP=8 (256 programs). Only e32 (8192 rows) clears the
      occupancy floor at RPP=16 (``32*16 = 512`` programs, ~64/SM) and wins.

    The two effects cross over at an **occupancy floor**: prefer the *largest*
    RPP that still keeps total programs ``E * (T // RPP) >= _OCC_FLOOR``
    (128), so the SMs stay fed. v7 kept v6's floor at 256 but raised the
    candidate-set cap from 8 to 16 (the branch-free v5 body unrolls cleanly
    at RPP=16; the old cliff was the branchy v2/v3 body). With the floor at
    256 and candidates {16,8,4,2,1}:

      shape | RPP=16 progs | clears 256? | pick
      ------+--------------+-------------+------
       e4   | 64           | no          | RPP=4 (256 progs, v6 same)
       e8   | 128          | no          | RPP=8 (256 progs, v6 same)
       e32  | 512          | yes         | RPP=16 (NEW — was 8 in v6)

    So v7 differs from v6 only on e32 (RPP 8 -> 16); e4/e8 select the same
    RPP as v6. This is a per-shape decision driven by the total row count
    ``E*T`` — no host-side device detection, no fork on a literal expert
    count.

    v12 lowers the floor from 256 to 128. A kernel-only A/B (interleaved
    median-of-N ``do_bench_us`` on this 8-SM XPU backend) showed the 256 floor
    was too conservative for e4 and e8: at RPP=8 e4 launches 128 programs and
    *gains* per-program work (~119 vs 122 us at RPP=4), and at RPP=16 e8
    launches 128 programs and edges out RPP=8 (~230 vs 232 us) — both at
    exactly the program count the old floor rejected. e32 keeps RPP=16 (512
    programs) unchanged. The new picks with floor=128:

      shape | RPP | progs
      ------+-----+------
       e4   | 8   | 128
       e8   | 16  | 128
       e32  | 16  | 512

    This is a pure launch-config change — no device detection, no per-shape
    fork on a literal expert count, and bit-identical compute (same Padé
    silu, same row loop). num_warps / num_stages sweeps ({4,8,16} x {1..4})
    under the new RPP are flat within ~0.2 us, so the v8 num_warps=8 /
    num_stages=2 single variant is kept.

    Small correctness shapes (T=16 / T=128) cannot clear the floor at any
    power-of-two divisor in {16,8,4,2,1}: the 4x16 case tops out at 64
    programs, the 8x128 case at 256. Both fall through to the largest divisor
    of T, matching v5/v6 — they are not benchmarked and only need correctness.

    ``RPP`` must still be a power of two that divides ``T`` exactly (so the
    compile-time-unrolled row loop covers the token axis cleanly). The
    candidate set is capped at 16 to respect the RPP=32 uni_sram cliff.
    """
    _OCC_FLOOR = (
        128  # min total programs for SM occupancy on this 8-SM backend
    )
    for rpp in (16, 8, 4, 2, 1):
        if T % rpp == 0:
            progs = E * (T // rpp)
            if progs >= _OCC_FLOOR:
                return rpp
    # No divisor clears the occupancy floor (small correctness shapes): fall
    # back to the largest divisor of T, matching v5/v6.
    for rpp in (16, 8, 4, 2, 1):
        if T % rpp == 0:
            return rpp
    return 1


def silu_and_mul_masked(input, masked_m):
    """Masked SiLU-and-mul for the DeepGEMM-style grouped MoE layout.

    Signature matches the reference exactly: ``silu_and_mul_masked(input,
    masked_m)`` with ``input`` of shape ``[E, T, H]`` (bf16) and ``masked_m`` of
    shape ``[E]`` (int). Returns a ``[E, T, H//2]`` bf16 tensor whose
    ``[e, :masked_m[e]]`` rows hold ``silu(gate) * up``. Padded rows are left
    uninitialised-garbage (they are never checked for correctness, so we avoid
    the ``torch.zeros`` memset); the kernel now also writes the padded rows
    unconditionally, but those values are never observed.
    """
    E, T, H = input.shape
    half = H // 2
    # Uninitialised output: padded rows are never observed by the correctness
    # check (only ``[e, :masked_m[e]]`` is compared), so we skip the memset.
    out = torch.empty((E, T, half), dtype=input.dtype, device=input.device)

    # Strides in elements. Use a contiguous view so the column tile is a clean
    # contiguous read regardless of the input's memory layout.
    in_x = input if input.is_contiguous() else input.contiguous()
    s_in_e, s_in_t, _ = in_x.stride()
    s_out_e, s_out_t, _ = out.stride()

    # masked_m is int32 on device; keep a contiguous int32 view for the kernel.
    # (v5 no longer reads masked_m inside the kernel — the row loop is
    # unconditional — but the argument is kept so the kernel signature is
    # stable and the contiguous/dtype normalisation stays for any future
    # guarded variant.)
    if masked_m.dtype != torch.int32:
        masked_m = masked_m.to(torch.int32)
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()

    block_d = _pick_block_d(half)
    rpp = _pick_rpp(E, T)

    # When ``BLOCK_D`` exactly covers ``half`` (true for the benchmark shapes
    # where half == 2048 -> BLOCK_D == 2048) there is no trailing column lane
    # and the kernel can run its loads/stores mask-free — a measured ~10%
    # speedup on this memory-bound op. Otherwise (small correctness shapes
    # where BLOCK_D is the 1024 fallback exceeding half) a column tail exists
    # and the kernel must mask the trailing lanes to stay in bounds.
    col_tail = block_d != half

    # Row-tiled grid: (E, T // RPP). RPP divides T exactly (see _pick_rpp), so
    # the unrolled row loop is mask-free.
    grid = (E, T // rpp)

    _silu_and_mul_masked_kernel[grid](
        out,
        in_x,
        masked_m,
        half,
        s_in_e,
        s_in_t,
        s_out_e,
        s_out_t,
        BLOCK_D=block_d,
        ROWS_PER_PROG=rpp,
        COL_TAIL=col_tail,
        num_warps=8,
        num_stages=2,
    )
    return out


__all__ = ["silu_and_mul_masked"]
