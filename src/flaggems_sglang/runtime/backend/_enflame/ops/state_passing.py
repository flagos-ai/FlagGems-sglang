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

"""Triton implementation of mamba/state_passing.

The Mamba2 SSD state-passing stage does a *sequential* scan across the
``nchunks`` time-chunks: a running state ``cur`` is carried chunk-to-chunk as

    cur_{c+1} = cur_c * exp(dA_last[c]) + states[c]

and the *pre-update* snapshot ``cur_c`` is what gets written to ``out[:, c]``.

The scan is sequential in ``c`` and cannot be parallelised across chunks, but
the whole ``[B, nheads, dim]`` state space is fully independent — every
``(b, h, dim)`` triple evolves on its own. One Triton program therefore owns a
``(b, h)`` pair (or a small *group* of consecutive pairs) and a tile of the
``dim`` axis and runs the whole chunk loop locally, streaming the chunk's
state tile through registers. Memory traffic is the minimum the problem
admits: one read + one write of the full ``[B, nchunks, nheads, dim]``
``states``/``out`` volume plus the tiny ``[B, nheads, nchunks]`` decay tail.

This is an arithmetic-intensity ~0.4 FLOP/byte kernel — firmly
**memory-bandwidth bound** — so the only lever is saturating memory bandwidth.
GCU tuning (gcu400 / 24-SIP, triton 3.6, kernel-mode do_bench median):

* **No host-side ``dA_last`` materialisation.** The per-chunk decay tail is
  loaded *directly* from ``dA_cumsum`` inside the kernel — one fp32 scalar
  per chunk per (b, h); the host path only allocates ``out`` and
  ``final_states``.

* **``num_warps = 4``.** The recurrence is an elementwise broadcast FMA, but
  the streamed tiles need warp-level issue parallelism to saturate the memory
  system: num_warps=1 measured 1.2-2.6x *slower* than 4 on the bench shapes,
  2 is consistently ~3-8% slower than 4, and 8 is indistinguishable from 4.

* **Program *grouping* (``BLOCK_BH``) targets a ~128-program wave.** The
  per-shape sweep minimises latency when the grid collapses to ~128 programs
  (B*H=256 -> BLOCK_BH=2, B*H=2048 -> BLOCK_BH=16), so the autotune keeps the
  grouping sweep {1, 2, 4, 8, 16}.

* **``tl.static_range`` chunk loop.** ``NCHUNKS`` is a compile-time constant,
  so the loop fully unrolls and the next-iteration prefetch guard
  ``if c + 1 < NCHUNKS`` is pruned at trace time — no runtime branches and no
  loop structure for the software pipeliner to fight (``num_stages`` measures
  flat 1..3 and is pinned to 2).

* **Per-chunk scalar ``dA`` load + ``exp``.** An earlier variant preloaded the
  whole ``[NCHUNKS]`` decay vector and selected the c-th element with
  ``tl.sum(tl.where(c_idx == c, decay_vec, 0.0))``; on GCU that masked-sum
  select is a cross-lane reduction issued every chunk and measured **+46-48%
  end-to-end** over the plain per-chunk scalar load + exp, so it was removed.

* **Manual next-iteration ``states`` prefetch, issued *after* the FMA.** The
  recurrence is loop-carried in ``cur``, so the ``states[c+1]`` tile load (and
  the next ``decay``) is hoisted right after ``cur = cur * decay + s`` — off
  the FMA's critical path. Measured -6-9% vs no prefetch; issuing the prefetch
  *between* the store and the FMA measured ~2-4% slower.

* ``dim`` is a constexpr (``DIM``), so ``DIM_FULL = (DIM % BLOCK_DIM) == 0``
  is exact for the *autotuned* block size and the unmasked (vectorised) hot
  path is emitted only when the tile truly spans the axis.

Round-2 (E-1) additions — the ``k`` loop over the pairs of a group:

* **Two-arm ``k`` loop, guard-free for full groups.** Only the *last* group
  can be partial, so the kernel branches once per program: full groups run
  the body without the per-iteration ``if bh < nbh`` guard (measured -1.3%
  B32/C4, -2.4% B8/C16), the tail group keeps the guarded loop.

* **Cross-pair ``dA`` prefetch (``PREFETCH_DV``), constexpr-dispatched.** At
  each pair switch the first ``decay`` load sits on the FMA chain's critical
  path; carrying the *scalar* ``dA`` value one ``k`` iteration ahead (loop
  iter_args, clamped address so the tail load stays in bounds) hides it —
  -2.1% on B32/C4 (16 pairs x 4 chunks). With *long* chunk sequences
  (B8/C16: 2 pairs x 16 chunks) the scalar loop-carry itself costs more than
  the latency it hides (+4.7%), so the arm is only emitted when
  ``BLOCK_BH >= 8 and NCHUNKS <= 8`` — both shapes keep their better arm and
  only one arm is ever codegen'd.

* Rejected with numbers (same-day A/B, bitwise-equal outputs):
  ``tl.static_range`` k-unroll (+4..+9%, code size), ``tl.minimum`` clamp
  replacing the guard (+2..+4%), carrying the ``states`` *tile* across k
  iterations (+4..+7% — tile loop-carried via iter_args is a scheduling
  poison on this backend, consistent with the selective_state_update case),
  and hoisting the s/dv loads of the non-prefetch arms out of the per-pair
  body into the k-loop caller (+4% on B8/C16 bbh=2 — the body must stay one
  straight-line load+compute sequence; only the PREFETCH_DV arm, which
  *must* carry dv, tolerates the split and pays for it with the constexpr
  dispatch above).

Measured on the bench shapes (bf16): B8/C16/H32/D8192 189 -> 77 us and
B32/C4/H64/D8192 398 -> 149 us; same-grid-structure copy kernels floor at
77 / 143 us, so the scan runs at the memory-bound roofline (~1.85 / 2.24
TB/s).
"""

import torch
import triton
import triton.language as tl


def _autotune_configs():
    # Swept on gcu400 (24-SIP): both bench shapes' latency minimises when the
    # grid collapses to ~128 programs, which BLOCK_BH selects per shape
    # (B*H=256 -> BLOCK_BH=2; B*H=2048 -> BLOCK_BH=16). BLOCK_DIM=8192 wins
    # for dim=8192 (4096 is ~10% slower, smaller tiles much worse); 4096 is
    # kept for dims that divide it. num_warps=4 is flat vs 8 and ~3-8% faster
    # than 2; num_stages is irrelevant under the fully-unrolled chunk loop and
    # pinned to 2.
    configs = []
    for bd in (4096, 8192):
        for bbh in (1, 2, 4, 8, 16):
            configs.append(
                triton.Config(
                    {"BLOCK_DIM": bd, "BLOCK_BH": bbh},
                    num_warps=4,
                    num_stages=2,
                )
            )
    return configs


@triton.jit
def _state_passing_pair(
    out_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    final_ptr,  # [B, nheads, dim] fp32
    states_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    dA_ptr,  # [B, nheads, nchunks, L] fp32
    init_ptr,  # [B, nheads, dim] (states.dtype) or None
    nheads,
    bh,  # flat (b * nheads + h) index of this pair
    s,  # chunk-0 states tile for this pair (already loaded)
    dv,  # chunk-0 raw dA_last scalar for this pair (already loaded)
    dim_offs,
    dim_mask,
    dA_sb,  # stride along batch axis of dA_cumsum
    dA_sh,  # stride along heads axis of dA_cumsum
    dA_sc,  # stride along chunks axis of dA_cumsum (= L)
    LAST: tl.constexpr,  # index of the last time-step within a chunk (L - 1)
    HAS_INIT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_FP16: tl.constexpr,
    NCHUNKS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    DIM_FULL: tl.constexpr,
):
    b = bh // nheads
    h = bh % nheads
    bh_base = b * (nheads * DIM * NCHUNKS) + h * DIM
    dA_base = b * dA_sb + h * dA_sh + LAST
    chunk_stride = nheads * DIM

    # Running state, kept in fp32 to match the reference.
    if HAS_INIT:
        if DIM_FULL:
            cur = tl.load(
                init_ptr + b * (nheads * DIM) + h * DIM + dim_offs
            ).to(tl.float32)
        else:
            cur = tl.load(
                init_ptr + b * (nheads * DIM) + h * DIM + dim_offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
    else:
        cur = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    decay = tl.exp(dv.to(tl.float32))
    for c in tl.static_range(0, NCHUNKS):
        # Snapshot of the pre-update state -> out[:, c].
        if OUT_FP32:
            v = cur
        elif OUT_BF16:
            v = cur.to(tl.bfloat16)
        else:
            v = cur.to(tl.float16)
        if DIM_FULL:
            tl.store(out_ptr + bh_base + c * chunk_stride + dim_offs, v)
        else:
            tl.store(
                out_ptr + bh_base + c * chunk_stride + dim_offs,
                v,
                mask=dim_mask,
            )

        # FMA first: the loop-carried dependency is ``cur`` itself, so the
        # next-iteration loads are issued after it.
        cur = cur * decay + s

        # Manual next-iteration prefetch of ``states[c+1]`` and ``decay[c+1]``
        # (see module docstring). ``c`` is a Python int under static_range, so
        # the guard is pruned at trace time and the last iteration issues no
        # out-of-range loads.
        if c + 1 < NCHUNKS:
            if DIM_FULL:
                s = tl.load(
                    states_ptr + bh_base + (c + 1) * chunk_stride + dim_offs
                )
            else:
                s = tl.load(
                    states_ptr + bh_base + (c + 1) * chunk_stride + dim_offs,
                    mask=dim_mask,
                    other=0.0,
                )
            dv = tl.load(dA_ptr + dA_base + (c + 1) * dA_sc)
            decay = tl.exp(dv.to(tl.float32))

    # final_states = cur
    if DIM_FULL:
        tl.store(final_ptr + b * (nheads * DIM) + h * DIM + dim_offs, cur)
    else:
        tl.store(
            final_ptr + b * (nheads * DIM) + h * DIM + dim_offs,
            cur,
            mask=dim_mask,
        )


@triton.jit
def _state_passing_pair_selfload(
    out_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    final_ptr,  # [B, nheads, dim] fp32
    states_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    dA_ptr,  # [B, nheads, nchunks, L] fp32
    init_ptr,  # [B, nheads, dim] (states.dtype) or None
    nheads,
    bh,  # flat (b * nheads + h) index of this pair
    dim_offs,
    dim_mask,
    dA_sb,  # stride along batch axis of dA_cumsum
    dA_sh,  # stride along heads axis of dA_cumsum
    dA_sc,  # stride along chunks axis of dA_cumsum (= L)
    LAST: tl.constexpr,  # index of the last time-step within a chunk (L - 1)
    HAS_INIT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_FP16: tl.constexpr,
    NCHUNKS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    DIM_FULL: tl.constexpr,
):
    # Self-loading variant used by the non-prefetch arms: everything for one
    # (b, h) pair — decay loads included — is issued from inside this body.
    # Hoisting the s/dv loads out to the k-loop caller measured +4% on
    # B8/C16 (bbh=2) on GCU400 (scheduler fragility), so this arm keeps the
    # load+compute sequence in one straight-line body.
    b = bh // nheads
    h = bh % nheads
    bh_base = b * (nheads * DIM * NCHUNKS) + h * DIM
    dA_base = b * dA_sb + h * dA_sh + LAST
    chunk_stride = nheads * DIM

    # Running state, kept in fp32 to match the reference.
    if HAS_INIT:
        if DIM_FULL:
            cur = tl.load(
                init_ptr + b * (nheads * DIM) + h * DIM + dim_offs
            ).to(tl.float32)
        else:
            cur = tl.load(
                init_ptr + b * (nheads * DIM) + h * DIM + dim_offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
    else:
        cur = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    # First chunk's inputs, hoisted out of the unrolled loop.
    if DIM_FULL:
        s = tl.load(states_ptr + bh_base + dim_offs)
    else:
        s = tl.load(states_ptr + bh_base + dim_offs, mask=dim_mask, other=0.0)
    decay = tl.exp(tl.load(dA_ptr + dA_base).to(tl.float32))

    for c in tl.static_range(0, NCHUNKS):
        # Snapshot of the pre-update state -> out[:, c].
        if OUT_FP32:
            v = cur
        elif OUT_BF16:
            v = cur.to(tl.bfloat16)
        else:
            v = cur.to(tl.float16)
        if DIM_FULL:
            tl.store(out_ptr + bh_base + c * chunk_stride + dim_offs, v)
        else:
            tl.store(
                out_ptr + bh_base + c * chunk_stride + dim_offs,
                v,
                mask=dim_mask,
            )

        # FMA first: the loop-carried dependency is ``cur`` itself, so the
        # next-iteration loads are issued after it.
        cur = cur * decay + s

        # Manual next-iteration prefetch of ``states[c+1]`` and ``decay[c+1]``.
        # ``c`` is a Python int under static_range, so the guard is pruned at
        # trace time and the last iteration issues no out-of-range loads.
        if c + 1 < NCHUNKS:
            if DIM_FULL:
                s = tl.load(
                    states_ptr + bh_base + (c + 1) * chunk_stride + dim_offs
                )
            else:
                s = tl.load(
                    states_ptr + bh_base + (c + 1) * chunk_stride + dim_offs,
                    mask=dim_mask,
                    other=0.0,
                )
            decay = tl.exp(
                tl.load(dA_ptr + dA_base + (c + 1) * dA_sc).to(tl.float32)
            )

    # final_states = cur
    if DIM_FULL:
        tl.store(final_ptr + b * (nheads * DIM) + h * DIM + dim_offs, cur)
    else:
        tl.store(
            final_ptr + b * (nheads * DIM) + h * DIM + dim_offs,
            cur,
            mask=dim_mask,
        )


@triton.autotune(
    configs=_autotune_configs(),
    key=["nbh", "DIM", "HAS_INIT", "OUT_FP32", "OUT_BF16", "OUT_FP16"],
)
@triton.jit
def _state_passing_kernel(
    out_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    final_ptr,  # [B, nheads, dim] fp32
    states_ptr,  # [B, nchunks, nheads, dim] (states.dtype)
    dA_ptr,  # [B, nheads, nchunks, L] fp32 (raw dA_cumsum, no materialisation)
    init_ptr,  # [B, nheads, dim] (states.dtype) or None
    nheads,
    nbh,  # B * nheads (total number of (b, h) pairs)
    dA_sb,  # stride along batch axis of dA_cumsum
    dA_sh,  # stride along heads axis of dA_cumsum
    dA_sc,  # stride along chunks axis of dA_cumsum (= L)
    LAST: tl.constexpr,  # index of the last time-step within a chunk (L - 1)
    HAS_INIT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_FP16: tl.constexpr,
    NCHUNKS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_BH: tl.constexpr,
):
    # Exact per-block fullness: DIM is a constexpr, so this folds for the
    # *autotuned* BLOCK_DIM and the unmasked path is only emitted when the
    # tile truly spans the dim axis.
    DIM_FULL: tl.constexpr = (DIM % BLOCK_DIM) == 0

    # Cross-pair dA prefetch only pays when the k loop is long and the
    # per-pair chunk sequence short (B32/C4: -2.1%); with long chunk
    # sequences the scalar loop-carry costs more than it hides (B8/C16:
    # +4.7%). Constexpr switch — only one arm is ever codegen'd.
    PREFETCH_DV: tl.constexpr = (BLOCK_BH >= 8) and (NCHUNKS <= 8)

    pid = tl.program_id(0)
    # Each program owns up to BLOCK_BH consecutive (b, h) pairs and a tile of
    # the dim axis. Mapping (pid -> (bh_group, dim_block)) keeps all programs
    # of one dim tile contiguous so memory accesses stay coalesced across the
    # group dimension too.
    n_dim_blocks = tl.cdiv(DIM, BLOCK_DIM)
    bh_group = pid // n_dim_blocks
    dim_block = pid % n_dim_blocks

    bh_start = bh_group * BLOCK_BH

    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offs < DIM

    # batch stride for states/out within the contiguous
    # [B, nchunks, nheads, dim] tensors
    bh_state_base = nheads * DIM * NCHUNKS

    if bh_start + BLOCK_BH <= nbh:
        # Full group: every pair in the group is valid, so the k loop runs
        # without the per-iteration ``bh < nbh`` guard. Only the tail group
        # ever takes the else branch below.
        if PREFETCH_DV:
            b0 = bh_start // nheads
            h0 = bh_start % nheads
            dv = tl.load(dA_ptr + b0 * dA_sb + h0 * dA_sh + LAST)
            for k in range(0, BLOCK_BH):
                bh = bh_start + k
                b = bh // nheads
                h = bh % nheads
                if DIM_FULL:
                    s = tl.load(
                        states_ptr + b * bh_state_base + h * DIM + dim_offs
                    )
                else:
                    s = tl.load(
                        states_ptr + b * bh_state_base + h * DIM + dim_offs,
                        mask=dim_mask,
                        other=0.0,
                    )
                _state_passing_pair(
                    out_ptr, final_ptr, states_ptr, dA_ptr, init_ptr,
                    nheads, bh, s, dv, dim_offs, dim_mask,
                    dA_sb, dA_sh, dA_sc,
                    LAST, HAS_INIT, OUT_FP32, OUT_BF16, OUT_FP16,
                    NCHUNKS, DIM, BLOCK_DIM, DIM_FULL,
                )
                # Next pair's chunk-0 decay, one k iteration ahead. The
                # address is clamped so the last iteration's (dead) load
                # stays in bounds; the value is only consumed when a further
                # iteration follows.
                bhn = tl.minimum(bh + 1, nbh - 1)
                dv = tl.load(
                    dA_ptr
                    + (bhn // nheads) * dA_sb
                    + (bhn % nheads) * dA_sh
                    + LAST
                )
        else:
            for k in range(0, BLOCK_BH):
                _state_passing_pair_selfload(
                    out_ptr, final_ptr, states_ptr, dA_ptr, init_ptr,
                    nheads, bh_start + k, dim_offs, dim_mask,
                    dA_sb, dA_sh, dA_sc,
                    LAST, HAS_INIT, OUT_FP32, OUT_BF16, OUT_FP16,
                    NCHUNKS, DIM, BLOCK_DIM, DIM_FULL,
                )
    else:
        # Partial tail group: keep the per-pair guard.
        for k in range(0, BLOCK_BH):
            bh = bh_start + k
            if bh < nbh:
                _state_passing_pair_selfload(
                    out_ptr, final_ptr, states_ptr, dA_ptr, init_ptr,
                    nheads, bh, dim_offs, dim_mask,
                    dA_sb, dA_sh, dA_sc,
                    LAST, HAS_INIT, OUT_FP32, OUT_BF16, OUT_FP16,
                    NCHUNKS, DIM, BLOCK_DIM, DIM_FULL,
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
        nbh,
        dA_sb,
        dA_sh,
        dA_sc,
        LAST=last,
        HAS_INIT=has_init,
        OUT_FP32=(dt == torch.float32),
        OUT_BF16=(dt == torch.bfloat16),
        OUT_FP16=(dt == torch.float16),
        NCHUNKS=nchunks,
        DIM=dim,
    )
    return out, final_states


__all__ = ["state_passing"]
