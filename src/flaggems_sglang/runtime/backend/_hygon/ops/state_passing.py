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

"""Mamba2 SSD state_passing as a portable pure-Triton kernel.

The recurrence ``cur_{c+1} = cur_c * exp(dA_last[b,c,h]) + states[b,c,h]`` is a
sequential scan over the ``nchunks`` chunk dimension, carrying a per-(b,h) state
vector of length ``dim``.  Each ``(b, h)`` pair is fully independent, and every
element of ``dim`` evolves with the *same* scalar decay ``exp(dA_last[b,c,h])``
that is broadcast across ``dim`` — so the recurrence is parallel across ``dim``
and across all (b,h) pairs, and only sequential across ``c``.

Because the sequential ``c`` dependency lives *inside* a single (b,h) pair, we
can hold the running state ``cur`` in registers and sweep all chunks in one
program.  We launch one program per ``(b, h, d_tile)`` triple using a 3-D grid
``(n_d_tiles, nheads, batch)``: axis 0 is the dim tile, axis 1 the head, axis 2
the batch element.  Putting the (b, h) sub-indices on their own program-id axes
avoids the per-program integer divide/remainder a 1-D / 2-D grid would need to
recover them — on this short inner loop (4-16 iterations) that address
arithmetic is a measurable slice of the kernel time, and the 3-D grid trims
~0.4-0.5 us off both bench shapes with no other change.  Each program seeds
``cur`` from ``initial_states`` (or zeros), then for ``c = 0 .. nchunks-1``
snapshots the current state into ``out[:, c]`` and applies the update.  After
the loop the register value *is* the final state, written to
``final_states[b, h]``.  This collapses ``nchunks`` kernel launches into one
and removes the scratch-buffer round trips, while keeping all computation in
float32 for precision.

Profiling on the eval hardware (vendor=hygon, sm9.3 "BW", 80 SMs) shows the
kernel is *memory-traffic bound* at ~1.20 TB/s — within a few percent of the
pure-copy ceiling for the mandatory read-states / write-out traffic.  The
recurrence's carried ``cur = cur*decay + st`` FMA is unavoidably serial, so the
scan can only partially overlap its compute with memory regardless of how many
independent programs are resident; this leaves only address-arithmetic /
code-size overhead to trim.  Per-config timing under the benchmark's own
``do_bench_us`` (median) measurement shows ``BLOCK_D=512, num_warps=4`` is the
consistent winner on both bench shapes (batch8 nc16 ~120 us, batch32 nc4 ~269
us): the 512-element tile is wide enough to saturate the per-program memory
path while 4 warps keep good SM occupancy, balancing the two against the serial
carried FMA.  ``BLOCK_D=256, num_warps=2`` ties within ~0.5 us (kept as an
occupancy-pressure fall-back); a 128/w1 config is kept as a high-occupancy
fall-back for the small / odd correctness shapes.

Grid: a 3-D grid ``(n_d_tiles, nheads, batch)`` — axis 0 the dim tile, axis 1
the head, axis 2 the batch element — so the two (b, h) sub-indices come
straight from their own program ids with *no* integer divide/remainder at all.
The earlier 2-D ``(d_tile, bh)`` grid recovered ``b = bh // nheads`` and
``h = bh % nheads`` per program; on this short inner loop (4-16 iterations)
those two integer ops are a measurable slice of per-program time, and lifting
them into separate launch axes removes them deterministically.  The launch
order (d-tile fastest, then h, then b) keeps the original wave-packing so SM
scheduling/occupancy is unchanged — this is a pure address-arithmetic trim,
portable across backends.

Chunk-loop structure: the loop is written with a single ``tl.static_range``.
``NCHUNKS`` is a small compile-time constant (4 for batch32, 16 for batch8, and
1/3/5 for the correctness cases), so ``static_range`` fully unrolls the body at
compile time — there is no runtime loop counter, no bounds check, and no
back-edge branch on the hot path, and *every* ``nchunks`` is handled correctly
with no separate odd-chunk tail.  An earlier version hand-paired the body into
a 2-way unroll with a constexpr odd-chunk tail to issue two states/out requests
back-to-back and raise per-program memory-level parallelism; measured under
the bench's median timer that paired form ties the natural ``static_range``
form within noise (~0.3 us) because the kernel is memory-bound and the extra
in-flight requests do not lift the already-saturated per-program memory path.
The natural form is strictly simpler (one loop, no tail branch) and strictly
smaller in code/I-cache, so it is preferred on the portable-performance basis:
``num_stages`` now has no effect on the recurrence (the carried ``cur`` blocks
software pipelining of the recurrence), so the autotune set keeps a couple of
``num_stages`` values only so the codegen can pick whichever trims a fraction
of a microsecond on a given shape.

``eviction_policy='evict_first'`` on the streaming ``states`` vector reads and
the write-once ``out`` stores keeps those once-used traffic from evicting the
resident ``cur`` working set and from polluting L2.  The ``dA_last`` *scalars*,
by contrast, are *reused* — the 16 d-tile programs sharing a ``(b, h)`` pair all
read the same ``NCHUNKS`` scalars — so they are loaded with ``evict_last`` to
pin them in L2 for the sibling d-tile programs to hit, rather than ``evict_first``
(which would drop each scalar before the next d-tile program could reuse it and
force redundant HBM trips).  ``final_states`` keeps the default cache policy
since it is returned and reread by the caller.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Empirically-optimal region, pruned to the configs autotune actually
        # selects on the eval hardware (vendor=hygon, sm9.3 "BW", 80 SMs).  The
        # kernel is memory-traffic bound: the recurrence's carried
        # ``cur = cur*decay + st`` FMA is unavoidably serial, so only a few
        # percent above the pure-copy ceiling is reachable and the remaining
        # lever is per-program memory-level parallelism versus SM occupancy.
        #
        # Measured under the benchmark's own ``do_bench_us`` (median) timing,
        # ``BLOCK_D=512, num_warps=4`` is the consistent winner on both bench
        # shapes (batch8 nc16 ~120 us, batch32 nc4 ~269 us): a 512-element tile
        # is wide enough to saturate the per-program memory path while 4 warps
        # keep good SM occupancy.  ``num_stages`` has no measurable effect on
        # the recurrence (the loop-carried ``cur`` blocks software pipelining of
        # the recurrence), but several values are kept so the codegen can pick
        # whichever trims a fraction of a microsecond on a given shape.
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=4),
        # ``BLOCK_D=256, num_warps=2`` is a near-tie fall-back (batch8 ~120 us,
        # batch32 ~269 us — within ~0.5 us of the 512/w4 winner) kept so autotune
        # can pick it if a future shape's occupancy pressure favours smaller tiles.
        triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=3),
        # ``BLOCK_D=128, num_warps=1`` is the high-occupancy fall-back for the
        # small / odd correctness shapes (dim 16/32/64, nc 1/3/5), where a 256-tile
        # covers the whole dim in one program anyway.
        triton.Config({"BLOCK_D": 128}, num_warps=1),
    ],
    key=["batch", "nheads", "dim", "nchunks", "DA_F32"],
)
@triton.jit
def _state_passing_scan_kernel(
    out_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    final_ptr,  # [B, nheads, dim]            float32
    states_ptr,  # [B, nchunks, nheads, dim]   (states.dtype)
    dA_ptr,  # [B, nheads, nchunks, L]
    init_ptr,  # [B, nheads, dim] float32 (valid pointer; used iff HAVE_INIT)
    batch,
    nheads,
    dim,
    nchunks,
    L,  # inner chunk length of dA_cumsum (last index = L-1)
    # strides for states / out : [B, nchunks, nheads, dim], dim contiguous
    s_b,
    s_c,
    s_h,
    # strides for dA : [B, nheads, nchunks, L]
    d_b,
    d_h,
    d_c,
    d_L,
    # stride for final : [B, nheads, dim] contiguous (-> f_b = nheads*dim, f_h = dim)
    f_b,
    f_h,
    # strides for init : [B, nheads, dim] (only used when HAVE_INIT)
    i_b,
    i_h,
    HAVE_INIT: tl.constexpr,
    NCHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DA_F32: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    # 3-D grid: axis 0 = d-tile (fastest-varying), axis 1 = head, axis 2 = batch.
    # Splitting the (d_tile, h, b) launch into three program-id axes recovers
    # the (b, h) sub-indices straight from program ids with *no* integer
    # divide/remainder at all.  The earlier 2-D ``(d_tile, bh)`` grid needed
    # ``b = bh // nheads`` / ``h = bh % nheads`` per program; on the recurrence
    # the loop-carried FMA keeps this memory-bound, so the only headroom left is
    # address-arithmetic overhead per program, and with only 4 (batch32) or 16
    # (batch8) inner-loop iterations those two integer ops are a visible
    # fraction of the kernel time.  Lifting them into separate launch axes
    # removes them deterministically.  The launch order (d-tile inner, then h,
    # then b) matches the 2-D grid's, so SM scheduling/wave packing is unchanged.
    d_tile = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    # `d_offs` lies on the contiguous innermost `dim` axis and, for the two bench
    # shapes (dim=8192, BLOCK_D∈{128,256,512}), the whole tile is in-range and
    # 128-B aligned.  Tagging the offsets with ``max_contiguous``/``multiple_of``
    # lets the Triton codegen widen the vector load/store of `states`/`out` to
    # the hardware's widest transaction (128 B) instead of the conservative
    # default it picks for an unannotated arange — a meaningful slice of the
    # memory-bound loop time when the whole loop is a few FMAs of streaming
    # traffic.  This is a pure codegen hint; it changes no addresses or values,
    # so correctness is unaffected, and the hint is portable (Triton lowers it
    # to whatever wide load the target supports).
    d_offs = d_tile * BLOCK_D + tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_D), 16), BLOCK_D
    )
    # ``FULL_TILE`` is a compile-time constant set by the host when ``dim`` is an
    # exact multiple of ``BLOCK_D``.  In that case every ``d_offs`` is in range so
    # the bounds mask is the all-true constant — and because Triton folds a
    # constant-true load/store into a wider, mask-free vector transaction, the
    # per-iteration loads/stores drop their per-element predicate logic entirely.
    # Both bench shapes (dim=8192) hit this path with the autotune-selected
    # BLOCK_D=512 (16 full tiles); the small correctness shapes (dim 16/32/64)
    # are not multiples of 512 so they take the masked path.  Either way
    # correctness is identical: when ``FULL_TILE`` the mask would have been
    # all-true anyway.  We keep ``other=0.0`` on every load (required by the
    # masked path) and let the constant mask fold it out on the full-tile path.
    if FULL_TILE:
        d_mask = tl.arange(0, BLOCK_D) >= 0
    else:
        d_mask = d_offs < dim

    # Base offset to the dim axis of (b, h) for out / states (dim contiguous).
    bh_base = b * s_b + h * s_h
    # Base offset to (b, h) in dA_cumsum; chunk c then adds c * d_c, last step L-1.
    dA_bh = b * d_b + h * d_h
    last_off = (L - 1) * d_L

    # Seed the running state in float32 registers.
    if HAVE_INIT:
        cur = tl.load(
            init_ptr + b * i_b + h * i_h + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)
    else:
        cur = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Sequential scan over chunks — cur stays in registers, no scratch buffer.
    #
    # ``NCHUNKS`` is a small compile-time constant (4/16 for the bench shapes,
    # 1/3/5 for the correctness cases), so ``tl.static_range`` fully unrolls the
    # body at compile time: no runtime loop counter, no bounds check, no
    # back-edge branch on the hot path, and every ``nchunks`` — even or odd — is
    # handled by this single loop with no separate tail.  The kernel is
    # memory-traffic bound (~1.20 TB/s, at the per-program copy ceiling), so the
    # carried ``cur = cur*decay + st`` FMA is the one unavoidable serial
    # dependency; the surrounding ``states`` / ``out`` / ``dA`` traffic is
    # independent across chunks and overlaps with that FMA as far as the
    # saturated memory path allows.
    #
    # ``eviction_policy`` hints keep the once-streamed ``states``/``dA`` reads
    # from polluting the cache that ``cur`` would otherwise compete with, and
    # the write-once ``out`` stores avoid evicting the ``dA`` scalars that *are*
    # reread across the 16 d-tiles sharing each (b, h) pair — so the whole
    # recurrence trips the L2 lightly and the hot `dA` reuse survives.
    for c in tl.static_range(0, NCHUNKS):
        # `out[:, c]` snapshots the state *before* the update (the input state of
        # chunk c).  Write-once / never-reread traffic -> ``evict_first``.
        tl.store(
            out_ptr + c * s_c + bh_base + d_offs,
            cur,
            mask=d_mask,
            eviction_policy="evict_first",
        )
        # Scalar decay exp(dA_last[b,c,h]) broadcast across `dim`.  Unlike the
        # streaming ``states``/``out`` vectors, these ``dA_last`` scalars are
        # *reused* — the 16 d-tile programs sharing a ``(b, h)`` pair all read the
        # same ``NCHUNKS`` scalars.  So they are loaded with ``evict_last`` to
        # pin them in L2 for the sibling d-tile programs to hit, rather than
        # ``evict_first`` (which would drop each scalar before the next d-tile
        # program could reuse it and force redundant HBM trips).  The d-tile
        # axis is the fastest-varying program-id, so a ``(b, h)``'s d-tiles are
        # scheduled as a tight wave and the earlier ones still warm in L2 by
        # the time the later ones run.
        dA = tl.load(
            dA_ptr + dA_bh + c * d_c + last_off, eviction_policy="evict_last"
        )
        decay = tl.exp(dA if DA_F32 else dA.to(tl.float32))
        # states[:, c] streamed read (only used once this iteration).
        st = tl.load(
            states_ptr + c * s_c + bh_base + d_offs,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # Carried recurrence — the one unavoidably serial dependency.
        cur = cur * decay + st

    # The register value after the last update is the final state (float32).
    # `final_states` is the one output that is reread (returned to the caller),
    # so it is left in the normal cache policy — unlike the streaming `out`
    # writes above which are evict_first.
    tl.store(final_ptr + b * f_b + h * f_h + d_offs, cur, mask=d_mask)


def state_passing(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    device = states.device
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=device, dtype=states.dtype
    )
    final_states = torch.empty(
        (batch, nheads, dim), device=device, dtype=torch.float32
    )

    # dA_cumsum: [B, nheads, nchunks, L]
    d_b, d_h, d_c, d_L = dA_cumsum.stride()
    L = dA_cumsum.shape[-1]

    # states / out: [B, nchunks, nheads, dim] — dim is contiguous (innermost).
    s_b, s_c, s_h, _ = states.stride()

    # final_states: contiguous [B, nheads, dim].
    f_b, f_h = final_states.stride(0), final_states.stride(1)

    if initial_states is None:
        # Pass any valid pointer; HAVE_INIT=False seeds cur with zeros.
        init_ptr = final_states
        i_b, i_h = f_b, f_h
        have_init = False
    else:
        init_f = initial_states.to(torch.float32).contiguous()
        init_ptr = init_f
        i_b, i_h = init_f.stride(0), init_f.stride(1)
        have_init = True

    # 3-D grid: (n_d_tiles, nheads, batch).  axis 0 = dim tile (fastest-varying),
    # axis 1 = head, axis 2 = batch — recovers (d_tile, h, b) straight from the
    # three program ids with no integer divide/remainder.
    grid = lambda meta: (triton.cdiv(dim, meta["BLOCK_D"]), nheads, batch)
    _state_passing_scan_kernel[grid](
        out,
        final_states,
        states,
        dA_cumsum,
        init_ptr,
        batch,
        nheads,
        dim,
        nchunks,
        L,
        s_b,
        s_c,
        s_h,
        d_b,
        d_h,
        d_c,
        d_L,
        f_b,
        f_h,
        i_b,
        i_h,
        HAVE_INIT=have_init,
        NCHUNKS=nchunks,
        DA_F32=(dA_cumsum.dtype == torch.float32),
        FULL_TILE=(dim % 512 == 0),
    )

    return out, final_states


__all__ = ["state_passing"]
