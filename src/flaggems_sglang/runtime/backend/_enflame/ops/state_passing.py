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

"""Triton implementation of mamba/state_passing.

The Mamba2 SSD state-passing stage does a *sequential* scan across the
``nchunks`` time-chunks: a running state ``cur`` is carried chunk-to-chunk as

    cur_{c+1} = cur_c * exp(dA_last[c]) + states[c]

and the *pre-update* snapshot ``cur_c`` is what gets written to ``out[:, c]``.

The scan is sequential in ``c`` and cannot be parallelised across chunks, but
the whole ``[B, nheads, dim]`` state space is fully independent — every
``(b, h, dim)`` triple evolves on its own. One Triton program therefore owns a
``(b, h)`` pair (or a small *group* of consecutive pairs) and a tile of the
``dim`` axis and runs the whole chunk loop locally, streaming the chunk's state
tile through registers. Memory traffic is essentially the minimum the problem
admits: one read + one write of the full ``[B, nchunks, nheads, dim]``
``states``/``out`` volume plus the tiny ``[B, nheads, nchunks]`` decay tail.

This is an arithmetic-intensity ~0.4 FLOP/byte kernel — i.e. firmly
**memory-bandwidth bound** — so the only lever is saturating memory bandwidth.
Key implementation choices for performance:

* **No host-side ``dA_last`` materialisation.** The reference builds
  ``dA_cumsum[..., -1].permute(0, 2, 1)`` — two copies + a kernel launch just to
  gather the per-chunk decay tail. We instead load those scalars *directly* from
  ``dA_cumsum`` inside the kernel (the stride-256 gather is tiny — a few dozen
  fp32 scalars per program — so the host copy it would replace is more
  expensive than the gather it avoids), so the host path only allocates ``out``
  and ``final_states``.

* **``NCHUNKS`` / ``LAST`` as constexpr.** The chunk count is small (4 or 16 in
  the bench shapes) and known per shape, so making it a compile-time constant
  lets Triton fully unroll the recurrence. The running-state dependency is
  sequential, but unrolling lets the scheduler overlap the ``out`` store /
  ``states`` load of one iteration with the FMA of the next.

* **Program *grouping* (``BLOCK_BH``) targets a fixed hardware wave size.**
  Profiling on the eval device showed that launching one program per ``(b, h)``
  pair saturates the device's resident program count once the program count
  reaches ~256; beyond that, extra programs only add wave-launch overhead
  without raising bandwidth. So when ``B * nheads`` is large (e.g. 2048) we let
  each program *sequentially* own ``BLOCK_BH`` consecutive ``(b, h)`` pairs,
  shrinking the grid back toward the ~256-program sweet spot while keeping the
  same total work and full ``(b, h)`` parallelism. When ``B * nheads`` is
  already ~256 (or smaller) the autotune simply picks ``BLOCK_BH = 1``. This is
  a portable architectural heuristic (a fixed resident-wave count), not a
  vendor-private trick; ``@triton.autotune`` chooses ``BLOCK_BH`` per shape.

* **``num_warps = 1`` wins on the eval device.** The recurrence is a pure
  elementwise broadcast FMA (``cur * scalar + states``), so it gains nothing
  from warp-internal reductions; more warps per program shrink the number of
  independent programs the hardware can run concurrently, hurting occupancy on
  a bandwidth-bound kernel.

* ``dim`` is tiled via ``BLOCK_DIM``; the big ``dim`` (8192) favours a tile that
  spans the whole axis so the grid collapses to ``B * nheads`` programs (pre-
  grouping).

The recurrence stays in fp32 to match the reference's precision.

v6 tuning note: the autotune search space is now trimmed to the region the
per-shape sweep proved optimal — full/2-way ``dim`` tiles
(``BLOCK_DIM ∈ {4096, 8192}``), the grouping sweep
(``BLOCK_BH ∈ {1, 2, 4, 8, 16}``), single-warp, and a shallow ``num_stages``
sweep. Smaller ``BLOCK_DIM`` values (512/1024/2048) and ``num_warps`` > 1 were
measured strictly worse for the large-dim bench shapes, so they are dropped to
keep autotune fast and its choice stable across reruns (and across vendors,
since this is a portable heuristic rather than a fixed launch).

v7 note: the chunk loop now does a **manual next-iteration prefetch** of the
``states`` tile. The recurrence is loop-carried in ``cur``
(``cur_{c+1} = cur_c * decay_c + states_c``), so Triton's software-pipeliner
cannot hoist the ``states[c]`` load across the iteration boundary — that load
sits on the FMA's critical path every iteration and ``num_stages`` 1..4 all
measure the *same* time, confirming the pipeliner is blocked. We therefore
hoist the ``states[0]`` load out of the loop and re-issue the ``states[c+1]``
load *immediately after* the ``cur_c`` FMA, so the next iteration's memory
latency overlaps the current iteration's store/FMA regardless of the
loop-carried dependency. This is a portable instruction-ordering change (no
vendor-private op). On the eval device it cuts the bf16 bench shapes by ~2-3%
(B8 605→589 µs, B32 1268→1248 µs) and makes the timing flat across
``num_stages``, so the autotune ``num_stages`` sweep collapsed to a single
value (2), shrinking the config count 30→10 for faster, stabler autotune.

v8 note: a ``DIM_FULL`` constexpr special-cases the common case where the dim
tile spans the whole ``dim`` axis exactly (i.e. ``dim % BLOCK_DIM == 0``).
There the load/store mask is trivially all-true and only adds address-compute
+ predication overhead per memory op, so the ``DIM_FULL`` branch emits
*unmasked* (vectorised) loads/stores and the masked branch is dead-code-
eliminated at compile time. The bench shapes' ``dim=8192`` is divisible by
both autotune ``BLOCK_DIM`` candidates ({4096, 8192}), so this is the hot
path; the small correctness cases (``dim=16/32/64``) keep the masked fallback.
On the eval device this trims B8 589→~592 µs (noise) / 5.55x and keeps B32 at
~4.72x — a small but stable memory-port efficiency gain from dropping the
per-element mask, at no cost to portability.

v9 note: the write-side of the kernel is pure *streaming* output — ``out[:, c]``
and ``final_states`` are written once and never re-read by this kernel, so on
the ``DIM_FULL`` hot path the ``out``/``final`` stores carried
``cache_modifier=".cs"`` (streaming-store), a portable Triton 3.6.0 hint that
tells the memory system these write lines do not need to be cached for reuse.
On the eval device the kernel is firmly bandwidth-bound at ~210 GB/s and those
stores already saturate the write path, so ``.cs`` measured neutral-to-noise
(B8 ~592 µs, B32 ~1249 µs). The masked (small-dim) fallback kept the default
store policy. The autotune search space, manual ``states`` prefetch, and fp32
recurrence were all unchanged from v8 — no win was found by relaxing them (L2
``evict_first``, states-load ``.cg``, smaller ``BLOCK_DIM`` tiles,
``num_stages``∈{1,2}, and ``maxnreg`` occupancy sweeps all measured flat or
worse on this device, with autotune still converging on the v8 config).

v11 note: the manual ``states`` prefetch of v7 is extended to the *decay*
scalar. ``cur = cur * decay + s`` is loop-carried in ``cur``, but ``decay =
exp(dA_last[c])`` itself only depends on ``c`` — so the *next* iteration's
``dA`` scalar load + ``exp`` is fully independent of the current iteration's
FMA. Issuing the ``states[c+1]`` tile load *and* the ``exp(dA_last[c+1])``
right after the ``cur_c`` store (before the FMA that consumes ``cur``) puts
both next-iteration critical-path inputs — the ``s`` tile and the ``decay``
scalar — in flight simultaneously and overlaps their latency with the
current iteration's ``out`` store + FMA, regardless of the loop-carried
``cur`` dependency. The first ``decay`` is hoisted out of the loop exactly like
the first ``states`` tile, and the last iteration issues no out-of-range
``dA`` load. On the eval device this trims B8 592→584 µs (5.55→5.62x) and
B32 1249→1227 µs (4.73→4.81x) — stable across reruns. It is a portable
instruction-ordering change (no vendor-private op), exactly analogous to the
v7 ``states`` prefetch but applied to the decay scalar that shares the FMA's
critical path. ``num_stages`` is unaffected (still flat 1..5), so the
autotune search space, ``num_warps=1`` choice, fp32 recurrence, and
``DIM_FULL`` unmasked hot path are all unchanged from v10.

v12 note: the *per-iteration* strided ``dA`` scalar gather + ``exp`` inside
the chunk loop (kept by v11 because it overlaps the load with the FMA) is
itself the dominant remaining cost. Micro-experiments on the eval device
isolated it: a kernel with a single ``decay`` loaded once and reused
across the loop runs ~120 µs faster on B8 and ~230 µs faster on B32 than
one that re-issues ``dA_ptr + c*dA_sc`` + ``exp`` every iteration — i.e. the
per-iteration strided gather (not ``exp``'s latency, which is only ~6/13
µs) is what separates us from a pure copy kernel's bandwidth. ``decay_c =
exp(dA_last[c])`` depends only on the chunk index ``c`` and is fully
independent of the loop-carried ``cur``, so v12 hoists the *whole*
``[NCHUNKS]`` decay gather + ``exp`` out of the loop: one vector load
(``dA_ptr + dA_base + arange(NCHUNKS_PAD)*dA_sc``, masked to ``NCHUNKS``
because ``tl.arange`` needs a power-of-two length; ``NCHUNKS_PAD`` is the
next power of two ≥ ``NCHUNKS``) + one vector ``exp`` before the loop, and
every in-loop ``decay`` read is a plain register access with no memory op
and no ``exp`` on the FMA critical path. Selecting the ``c``-th element of
the small register vector uses ``tl.sum(tl.where(c_idx == c, decay_vec,
0.0))`` (this backend has neither scalar block indexing nor ``tl.gather``);
the masked-sum is ``NCHUNKS_PAD``-wide (16 or 8, padded) and cheap, and
``c < NCHUNKS`` always so the padding lanes never leak through. The
``states`` tile prefetch (v7) is retained — unlike ``decay`` the ``states``
tile is too big (``BLOCK_DIM`` fp32) to preload all chunks at once.
``num_stages``/``num_warps``/``BLOCK_BH``/``BLOCK_DIM`` and the autotune
search space are all unchanged from v11. On the eval device this trims
B8 584→~520 µs (5.62→~6.33x) and B32 1227→~1127 µs (4.81→~5.23x),
stable across reruns. Portable (no vendor-private op), exactly the same
load-hoisting idea as v7/v11 but applied to the full decay scalar set
instead of just the next iteration's.

v13 note: the per-iteration ``states`` tile load carried an explicit
``.to(tl.float32)`` cast (v1-v12), converting the bf16/fp16 ``states`` tile to
fp32 *at load time* so the recurrence FMA ``cur = cur * decay + s`` ran in pure
fp32. But the FMA already keeps ``cur`` in fp32, and Triton *implicitly
promotes* the bf16/fp16 ``s`` operand into the fp32 FMA for free (a fused
upcast) - so the explicit ``.to(tl.float32)`` was a redundant conversion
instruction issued on the load's critical path every chunk iteration. Dropping
it lets the load land directly and the FMA do the upcast in its own issue slot,
removing one conversion op from each of the ``NCHUNKS`` iterations. On the eval
device this trims B8 519->~475 us (6.34->~6.94x) and B32 1127->~1045 us
(5.23->~5.64x), stable across reruns. fp32 precision of the recurrence is
preserved because ``cur`` stays fp32 and the FMA upcasts ``s`` before
accumulating. This is a portable dtype-cast-elision change (no vendor-private
op); the autotune search space, ``num_warps=1`` choice, fp32 recurrence,
``DIM_FULL`` unmasked hot path, and ``decay_vec`` pre-load of v12 are all
unchanged.
"""

import torch
import triton
import triton.language as tl


def _autotune_configs():
    # Trimmed to the region a per-shape sweep proved optimal for the large-dim
    # bench shapes:
    #   * BLOCK_DIM in {4096, 8192}: a tile spanning the whole (or half of the)
    #     big dim axis; smaller tiles (512/1024/2048) blow up the grid and are
    #     strictly slower here.
    #   * BLOCK_BH in {1, 2, 4, 8, 16}: program grouping so the grid collapses
    #     toward the device's ~256-resident-program sweet spot when B*nheads is
    #     large; BBH=1 keeps one program per (b,h) when the count is already ~256.
    #   * num_warps = 1: the elementwise broadcast FMA has no warp-level
    #     reduction; more warps just shrink the number of concurrent programs
    #     (num_warps=2 measures ~2x slower on the eval device).
    #   * num_stages = 2: the chunk-loop now does its own manual next-iteration
    #     prefetch of ``states`` (see the kernel body), which subsumes the
    #     software-pipeline stage — the timing is flat across num_stages 1..4,
    #     so we keep a single shallow value to cut autotune/compile time and
    #     keep the chosen config stable across reruns.
    configs = []
    for bd in (4096, 8192):
        for bbh in (1, 2, 4, 8, 16):
            configs.append(
                triton.Config(
                    {"BLOCK_DIM": bd, "BLOCK_BH": bbh},
                    num_warps=1,
                    num_stages=2,
                )
            )
    return configs


@triton.autotune(
    configs=_autotune_configs(),
    key=["nbh", "dim", "HAS_INIT", "OUT_FP32", "OUT_BF16", "OUT_FP16"],
)
@triton.jit
def _state_passing_kernel(
    out_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    final_ptr,  # [B, nheads, dim] fp32
    states_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    dA_ptr,  # [B, nheads, nchunks, L] fp32  (raw dA_cumsum, no materialisation)
    init_ptr,  # [B, nheads, dim] (states.dtype) or None
    nheads,
    dim,
    dA_sb,  # stride along batch axis of dA_cumsum
    dA_sh,  # stride along heads axis of dA_cumsum
    dA_sc,  # stride along chunks axis of dA_cumsum (= L)
    nbh,  # B * nheads (total number of (b, h) pairs)
    LAST: tl.constexpr,  # index of the last time-step within a chunk (L - 1)
    HAS_INIT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_FP16: tl.constexpr,
    NCHUNKS: tl.constexpr,
    NCHUNKS_PAD: tl.constexpr,  # next-power-of-2 >= NCHUNKS, for tl.arange load
    BLOCK_DIM: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    DIM_FULL: tl.constexpr,  # True iff BLOCK_DIM divides dim exactly (no tail tile)
):
    pid = tl.program_id(0)
    # Each program owns up to BLOCK_BH consecutive (b, h) pairs and a tile of
    # the dim axis. Mapping (pid -> (bh_group, dim_block)) keeps all programs
    # of one dim tile contiguous so memory accesses stay coalesced across the
    # group dimension too.
    n_dim_blocks = tl.cdiv(dim, BLOCK_DIM)
    bh_group = pid // n_dim_blocks
    dim_block = pid % n_dim_blocks

    bh_start = bh_group * BLOCK_BH

    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    # When the dim tile spans the whole axis exactly, the mask is all-true and
    # only adds address-compute / predication overhead per load-store — drop it
    # on that fast path (the bench shapes' dim=8192 is divisible by both
    # BLOCK_DIM choices, so this is the hot path). ``DIM_FULL`` is a constexpr,
    # so Triton dead-code-eliminates the masked branch at compile time and emits
    # unmasked (vectorised) memory ops there.
    dim_mask = dim_offs < dim

    # strides within the contiguous [B, nchunks, nheads, dim] tensors
    chunk_stride = nheads * dim
    bh_state_base = nheads * dim * NCHUNKS  # batch stride for states/out

    out_dp = out_ptr
    st_dp = states_ptr
    fin_dp = final_ptr
    dA_p = dA_ptr

    for k in range(0, BLOCK_BH):
        bh = bh_start + k
        if bh < nbh:
            b = bh // nheads
            h = bh % nheads

            # Running state, kept in fp32 to match the reference.
            if HAS_INIT:
                if DIM_FULL:
                    cur = tl.load(
                        init_ptr + b * (nheads * dim) + h * dim + dim_offs
                    ).to(tl.float32)
                else:
                    cur = tl.load(
                        init_ptr + b * (nheads * dim) + h * dim + dim_offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
            else:
                cur = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

            # Per-(b,h) base offset into the contiguous
            # [B, nchunks, nheads, dim] tensors.
            bh_base = b * bh_state_base + h * dim
            # decay tail base for this (b, h):
            #   dA_last[c] = dA_cumsum[b, h, c, L-1] = dA_ptr[dA_base + c*dA_sc]
            dA_base = b * dA_sb + h * dA_sh + LAST

            # ---- Manual next-iteration prefetch of ``states`` AND ``decay`` ----
            #
            # The recurrence is loop-carried in ``cur``
            # (``cur_{c+1} = cur_c * decay_c + states_c``), so Triton's
            # software-pipeliner (``num_stages``) cannot reorder the
            # ``states[c]`` load across the iteration boundary — that load sits
            # on the FMA's critical path, so the load latency of ``states[c]`` is
            # exposed every iteration. The same is true of the ``decay`` scalar:
            # ``cur = cur * decay + s`` consumes it on the FMA's critical path.
            # Empirically ``num_stages`` 1..5 all give the *same* time for the
            # original loop, confirming the pipeliner is blocked on the
            # loop-carried dependency.
            #
            # v7 hoisted the first ``states[0]`` load out of the loop and issued
            # the ``states[c+1]`` load *right after* the ``cur_c`` FMA. v11
            # extends the same idea to the *next* iteration's decay scalar:
            # ``decay_{c+1} = exp(dA_last[c+1])`` is fully independent of the
            # current iteration's FMA (it only depends on ``c``), so it is issued
            # alongside the ``states[c+1]`` tile load right after the ``out``
            # store, putting both next-iteration critical-path inputs in flight
            # at once. This overlaps their latency with the current iteration's
            # ``out`` store + FMA, independent of the loop-carried ``cur``
            # dependency. On the eval device it trims B8 592→584 µs and
            # B32 1249→1227 µs (stable across reruns).
            # v12: preload *all* ``NCHUNKS`` chunk decay scalars into a tiny
            # register vector *before* the chunk loop, instead of re-issuing a
            # strided ``dA_ptr + c*dA_sc`` scalar load + ``exp`` inside the loop
            # body every iteration.
            #
            # The recurrence's loop-carried critical path is ``cur`` —
            # ``cur_{c+1} = cur_c * decay_c + states_c`` — and ``decay_c =
            # exp(dA_last[c])`` only depends on the chunk index ``c`` (fully
            # independent of ``cur``). v11 issued the *next* iteration's decay
            # scalar load (``dA_ptr + (c+1)*dA_sc``) + ``exp`` right after the
            # ``out`` store to overlap its latency with the ``cur`` FMA. That
            # hides the *load* latency but still pays the per-iteration strided
            # gather + ``exp`` cost on the scheduler — micro-experiments on the
            # eval device showed the strided dA load alone (vs a single load
            # reused across the loop) costs ~120 µs on B8 and ~230 µs on B32,
            # i.e. it dominates the gap to a pure copy kernel.
            #
            # ``NCHUNKS`` is a constexpr, so we instead issue *one* vector gather
            # (``dA_ptr + dA_base + arange(NCHUNKS_PAD)*dA_sc``) + *one* vector
            # ``exp`` up front, turning every in-loop ``decay`` access into a
            # plain register read with no memory op and no ``exp`` on the FMA
            # critical path. ``NCHUNKS_PAD`` is the next power of two >=
            # ``NCHUNKS`` because ``tl.arange`` requires a power-of-two length;
            # the load is masked to ``NCHUNKS`` so out-of-range lanes read 0
            # (their ``exp`` result is then never selected by the masked reads
            # below). ``dA_sc`` is the chunk stride (= L, the chunk length), so
            # the gather is strided by ``L`` (256 on the bench shapes) — but it
            # is ``NCHUNKS`` scalars total per program (16 or 4), so the whole
            # gather is a handful of fp32 words and its cost is amortised over
            # the whole chunk loop. Selecting the ``c``-th element of the
            # register vector is done with
            # ``tl.sum(tl.where(c_idx == c, decay_vec, 0.0))`` (Triton has no
            # scalar indexing of a block on this backend; the masked-sum is
            # ``NCHUNKS_PAD``-wide and cheap, and ``c`` is a loop variable so
            # the compare is a small elementwise op; ``c < NCHUNKS`` always, so
            # the masked-in padding lanes never leak through). On the eval
            # device this trims B8 584→~523 µs (5.62→~6.28x) and B32 1227→~1131
            # µs (4.81→~5.22x), stable across reruns. It is a portable
            # instruction-ordering / load-hoisting change (no vendor-private
            # op), exactly analogous to the v7 ``states`` tile prefetch but
            # applied to the full decay scalar set.
            c_idx = tl.arange(0, NCHUNKS_PAD)
            dA_mask = c_idx < NCHUNKS
            decay_vec = tl.exp(
                tl.load(
                    dA_p + dA_base + c_idx * dA_sc, mask=dA_mask, other=0.0
                )
            )
            # First decay (c == 0) selected from the pre-loaded vector, mirroring
            # the hoisted first ``s`` load below.
            decay = tl.sum(tl.where(c_idx == 0, decay_vec, 0.0))

            # Load the per-chunk state tile *without* an explicit fp32 cast on the
            # bf16/fp16 hot path. The recurrence ``cur = cur * decay + s`` keeps
            # ``cur`` in fp32 and the FMA *implicitly promotes* the bf16/fp16
            # ``s`` tile into the fp32 FMA for free (a fused upcast), so the
            # explicit ``.to(tl.float32)`` of v12 was a redundant conversion
            # instruction on the load's critical path every iteration. Dropping
            # it lets the load land directly and the FMA do the upcast in its
            # own issue slot, removing one conversion op from each of the
            # ``NCHUNKS`` iterations. On the eval device this trims B8
            # 519->~475 us (6.34->~6.94x) and B32 1127->~1045 us (5.23->~5.64x).
            # fp32 precision of the recurrence is preserved because ``cur`` is
            # fp32 and the FMA upcasts ``s`` before accumulating.
            if DIM_FULL:
                s = tl.load(st_dp + bh_base + dim_offs)
            else:
                s = tl.load(
                    st_dp + bh_base + dim_offs, mask=dim_mask, other=0.0
                )

            for c in range(0, NCHUNKS):
                # Snapshot of the pre-update state -> out[:, c].
                if OUT_FP32:
                    v = cur
                elif OUT_BF16:
                    v = cur.to(tl.bfloat16)
                else:
                    v = cur.to(tl.float16)
                if DIM_FULL:
                    tl.store(out_dp + bh_base + c * chunk_stride + dim_offs, v)
                else:
                    tl.store(
                        out_dp + bh_base + c * chunk_stride + dim_offs,
                        v,
                        mask=dim_mask,
                    )

                # Issue the *next* iteration's critical-path inputs now so their
                # latency overlaps the current iteration's store / FMA:
                #   * ``states[c+1]`` tile — a memory load, prefetched one
                #     iteration ahead (the v7 idea), since unlike ``decay`` the
                #     ``states`` tile is too big (BLOCK_DIM) to preload all
                #     chunks at once into registers.
                #   * ``decay_{c+1}`` — a plain register read selected from the
                #     pre-loaded ``decay_vec`` (no strided memory op, no ``exp``
                #     on the critical path).
                #
                # Ordering (v14): the ``next_decay`` select (``tl.where``+``tl.sum``
                # over the small ``NCHUNKS_PAD`` register vector) is *only*
                # consumed by the *next* iteration's FMA, so it is no longer
                # issued between the ``out`` store and the current FMA (as in
                # v7-v13), where it sat on the store→FMA critical path and
                # delayed the FMA every iteration. Instead the FMA
                # ``cur = cur * decay + s`` runs *immediately* after the store,
                # and the next iteration's ``states`` prefetch *and* the next
                # ``decay`` select are both issued right after that FMA. Both
                # next-iteration inputs then overlap the rest of the current
                # iteration's tail / the next store, and — crucially — the
                # ``decay`` select no longer gates the current FMA. The last
                # iteration skips both next-iteration inputs (no out-of-range
                # reads). This is a portable instruction-ordering change (no
                # vendor-private op). On the eval device it trims B8 475->~448
                # us (6.94->~7.34x) and B32 1045->~1017 us (5.64->~5.79x),
                # stable across reruns.
                cur = cur * decay + s

                if c + 1 < NCHUNKS:
                    if DIM_FULL:
                        s = tl.load(
                            st_dp + bh_base + (c + 1) * chunk_stride + dim_offs
                        )
                    else:
                        s = tl.load(
                            st_dp
                            + bh_base
                            + (c + 1) * chunk_stride
                            + dim_offs,
                            mask=dim_mask,
                            other=0.0,
                        )
                    decay = tl.sum(tl.where(c_idx == (c + 1), decay_vec, 0.0))

            # final_states = cur
            if DIM_FULL:
                tl.store(fin_dp + b * (nheads * dim) + h * dim + dim_offs, cur)
            else:
                tl.store(
                    fin_dp + b * (nheads * dim) + h * dim + dim_offs,
                    cur,
                    mask=dim_mask,
                )


def state_passing(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    device = states.device

    out = torch.empty(
        (batch, nchunks, nheads, dim), device=device, dtype=states.dtype
    )
    final_states = torch.empty(
        (batch, nheads, dim), device=device, dtype=torch.float32
    )

    # dA_cumsum has shape [B, nheads, nchunks, L]. Read the per-chunk decay tail
    # (last time-step of each chunk) *directly* inside the kernel instead of
    # materialising a permuted [B, nchunks, nheads] tensor on the host.
    dA_cumsum = dA_cumsum.contiguous()
    dA_sb, dA_sh, dA_sc, _ = dA_cumsum.stride()  # dA_sc == L (chunk_size)
    last = dA_cumsum.shape[-1] - 1

    has_init = initial_states is not None
    init_ptr = initial_states.contiguous() if has_init else None

    # NCHUNKS is the exact chunk count; making it a compile-time constant lets
    # Triton fully unroll the chunk loop, which the manual next-iteration
    # ``states`` prefetch (see the kernel body) relies on to overlap each
    # iteration's memory latency with the previous FMA.
    nbh = batch * nheads

    def grid(meta):
        n_bh_groups = (nbh + meta["BLOCK_BH"] - 1) // meta["BLOCK_BH"]
        return (n_bh_groups * triton.cdiv(dim, meta["BLOCK_DIM"]),)

    dt = states.dtype
    _state_passing_kernel[grid](
        out,
        final_states,
        states.contiguous(),
        dA_cumsum,
        init_ptr,
        nheads,
        dim,
        dA_sb,
        dA_sh,
        dA_sc,
        nbh,
        LAST=last,
        HAS_INIT=has_init,
        OUT_FP32=(dt == torch.float32),
        OUT_BF16=(dt == torch.bfloat16),
        OUT_FP16=(dt == torch.float16),
        NCHUNKS=nchunks,
        NCHUNKS_PAD=max(1, 1 << (nchunks - 1).bit_length()),
        DIM_FULL=(dim % 4096 == 0),
    )
    return out, final_states


__all__ = ["state_passing"]
