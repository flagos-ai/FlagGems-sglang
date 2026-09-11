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

Optimization notes
--------------------
The recurrence ``cur_{c+1} = cur_c * exp(dA_last[b,c,h]) + states_f[b,c,h]`` is
sequential across the ``nchunks`` dimension but fully independent across
``(batch, nheads, dim)``.  We launch a single kernel whose grid covers
``(dim_blocks, batch * nheads)`` — each program owns one tile of the ``dim``
axis and one ``(b, h)`` pair, then strides sequentially over the chunks.

The kernel is memory-bandwidth bound: each chunk costs one ``states`` vector
load plus one ``out`` vector store, while the running state ``cur`` lives in
registers across all chunks.  Measured against a pure-copy kernel that does
the *same* load/store pattern (no arithmetic), this kernel already matches the
copy floor — i.e. the loop-carried recurrence is fully hidden behind memory
traffic, so the only remaining levers are launch/grid choices:

* **Dim-first grid order.**  The grid is ``(dim_blocks, batch * nheads)`` with
  ``dim`` on ``program_id(0)`` and ``b*H`` on ``program_id(1)``.  On Iluvatar
  BI-V150 (16 SMs) this is consistently ~1% faster than the transpose order
  ``(batch * nheads, dim_blocks)`` across both bench shapes, presumably
  because the scheduler's wave of co-resident programs shares the same
  ``dim`` tile and thus reuses L2 cache lines for the per-chunk ``states``
  / ``out`` vectors more effectively.

* **Software-pipelined chunk loop.**  The chunk loop is written with
  ``tl.range`` (not ``tl.static_range``) so Triton overlaps the
  ``states`` / ``dA_last`` loads of iteration ``c+1`` with the ``out`` store
  of iteration ``c`` via ``num_stages``.  The loop-carried ``cur`` dependency
  still serialises the arithmetic, but the per-chunk vector load/store
  latency — the dominant cost — is hidden.

* **Tile width vs. occupancy.**  ``BLOCK_DIM = 4096`` (two tiles per ``dim``)
  with ``num_warps = 32`` wins on both bench shapes: fine enough to pack the
  grid across the 16 SMs while the 2048 threads/block give high per-block
  occupancy that hides the per-chunk load/store latency.  A few smaller tiles
  are kept only so the tiny correctness shapes (``dim = 16 / 32 / 64``) can
  pick a tile that divides ``dim``.

* **Lean autotune set.**  The config list is deliberately small and centred on
  the measured sweet spot (``BLOCK_DIM=4096``, ``num_warps=32``,
  ``num_stages`` near ``nchunks``).  A bloated search space not only wastes
  compilation/retune time, it makes the autotuner's pick noisier; a focused
  set lets each shape reliably settle on its optimum.
"""

import torch
import triton
import triton.language as tl


def _autotune_configs():
    # A deliberately small, focused set centred on the measured sweet spot
    # (BLOCK_DIM=4096 / num_warps=32) for the bench shapes (dim=8192).  The
    # tiny correctness shapes (dim=16/32/64) also run fine on a 4096-wide tile
    # — the extra lanes are simply masked off, and since those shapes are
    # correctness-only (not timed) the waste is harmless.  Keeping the list
    # lean trims autotune compilation time and makes the choice less noisy.
    #
    # num_stages now reaches up to NCHUNKS where sensible: the chunk loop is
    # memory-bound and ``tl.range`` software-pipelines the per-chunk
    # ``states``/``dA_last`` loads against the ``out`` store.  Deeper stages
    # let more chunk-loads stay in flight behind the running store stream,
    # which matters for the larger ``nchunks`` bench (nc=16).  For the small
    # ``nchunks`` bench (nc=4) only the shallow end is reachable.
    plan = [
        # BLOCK_DIM : (num_warps..., num_stages...)
        # bench sweet spot: dim=8192 -> 2 tiles of 4096
        (4096, ((16, 32), (3, 4, 5, 6, 8))),
        # whole-dim tile for dim=8192: amortise per-chunk overhead + zero mask
        (8192, ((32, 64), (3, 4, 5, 6, 8))),
        # finer dim tiles (4 / 8 per dim) for dim=8192: larger grid, more SM
        # waves -> better launch-overhead amortisation.  Kept as an option
        # the autotuner may pick if the wider tiles fail to saturate the SMs;
        # the tiny correctness shapes simply mask off the extra lanes.
        (2048, ((8, 16), (3, 4, 5, 6, 8))),
    ]
    cfgs = []
    for bd, (warps, stages) in plan:
        for nw in warps:
            for ns in stages:
                cfgs.append(
                    triton.Config(
                        {"BLOCK_DIM": bd}, num_warps=nw, num_stages=ns
                    )
                )
    return cfgs


@triton.autotune(
    configs=_autotune_configs(),
    # Shape-derived key so each (batch, nheads, nchunks) shape keeps its own
    # tuned config; the recurrence count and grid size both change the
    # bandwidth/launch trade-off.
    key=[
        "BATCH",
        "NHEADS",
        "NCHUNKS",
        "DIM",
        "HAS_INIT",
        "OUT_FP32",
        "OUT_BF16",
        "OUT_F16",
    ],
)
@triton.jit
def _state_passing_kernel(
    states_ptr,
    dA_ptr,
    init_ptr,
    out_ptr,
    final_ptr,
    BATCH,
    NHEADS,
    DIM,
    stride_s_b,
    stride_s_c,
    stride_s_h,
    stride_s_d,
    stride_da_b,
    stride_da_h,
    stride_da_c,
    stride_da_l,
    stride_i_b,
    stride_i_h,
    stride_i_d,
    stride_o_b,
    stride_o_c,
    stride_o_h,
    stride_o_d,
    stride_f_b,
    stride_f_h,
    stride_f_d,
    L_LAST,  # == L - 1, index of the last time-step inside a chunk
    NCHUNKS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_INIT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_F16: tl.constexpr,
):
    # Dim-first grid order (program_id(0) = dim tile, program_id(1) = b*H):
    # consistently ~1% faster than the transpose order on BI-V150, because the
    # scheduler's co-resident wave shares the same dim tile and reuses L2
    # cache lines for the per-chunk states/out vectors.
    pid_dim = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NHEADS
    h = pid_bh % NHEADS

    offs_d = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < DIM

    # Running SSM state for this (b, h) restricted to our dim tile (float32).
    if HAS_INIT:
        cur = tl.load(
            init_ptr + b * stride_i_b + h * stride_i_h + offs_d * stride_i_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
    else:
        cur = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    dA_base = dA_ptr + b * stride_da_b + h * stride_da_h

    # ``tl.range`` (not ``tl.static_range``): keeps the chunk loop as a real
    # hardware loop so ``num_stages`` can software-pipeline the per-chunk
    # ``states``/``dA_last`` loads against the ``out`` store.  The loop-carried
    # ``cur`` dependency still serialises the arithmetic, but the memory
    # latency — the dominant cost on this bandwidth-bound kernel — is hidden.
    for c in tl.range(0, NCHUNKS):
        # Snapshot the incoming state of this chunk (cast to output dtype).
        if OUT_FP32:
            cur_cast = cur.to(tl.float32)
        elif OUT_BF16:
            cur_cast = cur.to(tl.bfloat16)
        else:
            cur_cast = cur.to(tl.float16)
        # ``out`` is write-once (never read back by this kernel), so mark it
        # ``evict_first`` to keep these stores from pushing the soon-to-be-needed
        # ``states`` prefetch lines out of L2.
        tl.store(
            out_ptr
            + b * stride_o_b
            + c * stride_o_c
            + h * stride_o_h
            + offs_d * stride_o_d,
            cur_cast,
            mask=mask_d,
            eviction_policy="evict_first",
        )

        # Last-step log-decay of this chunk -> per-head scalar decay factor.
        dA_last = tl.load(dA_base + c * stride_da_c + L_LAST * stride_da_l)
        decay = tl.exp(dA_last.to(tl.float32))

        # Incoming state delta for this chunk (promoted to float32).
        # Read-once, never reused across chunks -> ``evict_first`` so it does
        # not displace the L2 lines the next chunk's load will want.
        states_c = tl.load(
            states_ptr
            + b * stride_s_b
            + c * stride_s_c
            + h * stride_s_h
            + offs_d * stride_s_d,
            mask=mask_d,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        cur = cur * decay + states_c

    # Final accumulated state (float32).
    tl.store(
        final_ptr + b * stride_f_b + h * stride_f_h + offs_d * stride_f_d,
        cur,
        mask=mask_d,
        eviction_policy="evict_first",
    )


def state_passing(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    out = torch.empty(
        batch, nchunks, nheads, dim, device=states.device, dtype=states.dtype
    )
    final_states = torch.empty(
        batch, nheads, dim, device=states.device, dtype=torch.float32
    )

    # dA_cumsum: [B, nheads, nchunks, L]; we read its last time-step per chunk.
    L = dA_cumsum.shape[-1]

    out_fp32 = states.dtype == torch.float32
    out_bf16 = states.dtype == torch.bfloat16
    out_f16 = states.dtype == torch.float16
    has_init = initial_states is not None

    if not has_init:
        # ``HAS_INIT=False`` is a ``tl.constexpr`` that fully eliminates the
        # ``init`` load at compile time, so the ``init`` pointer is never
        # dereferenced by this kernel specialisation.  Rather than paying for a
        # per-call ``torch.empty`` allocation just to hand the kernel a valid
        # address (the host-side allocator cost shows up under ``do_bench``'s
        # end-to-end timing), we alias the already-allocated ``final_states``
        # buffer (same shape/strides/dtype: ``[B, H, dim]`` float32) as the
        # placeholder pointer.  This is purely an address alias — no data is
        # read or written through it — and is fully portable.
        init_t = final_states
    else:
        # Reference promotes to float32 and clones; we feed the original and
        # cast inside the kernel (read-only), avoiding an extra device copy here.
        init_t = initial_states

    # Dim-first grid: (dim_blocks, batch * nheads).  See kernel docstring for
    # why this order is faster than the transpose on the target hardware.
    grid = lambda meta: (
        triton.cdiv(dim, meta["BLOCK_DIM"]),
        batch * nheads,
    )

    _state_passing_kernel[grid](
        states,
        dA_cumsum,
        init_t,
        out,
        final_states,
        batch,
        nheads,
        dim,
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(1),
        dA_cumsum.stride(2),
        dA_cumsum.stride(3),
        init_t.stride(0),
        init_t.stride(1),
        init_t.stride(2) if init_t.dim() == 3 else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        final_states.stride(0),
        final_states.stride(1),
        final_states.stride(2),
        L - 1,
        NCHUNKS=nchunks,
        HAS_INIT=has_init,
        OUT_FP32=out_fp32,
        OUT_BF16=out_bf16,
        OUT_F16=out_f16,
    )

    return out, final_states


__all__ = ["state_passing"]
