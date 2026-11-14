# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""hash_topk — hash-routed MoE gate with sqrt-softplus scoring (Triton).

Contract (task statement, with the reference implementation as the source of
truth for the shared-slot ids):

    expert_ids = tid2eid[input_ids]                     # [T, topk_routed] int32
    logits     = router_logits.gather(1, expert_ids)    # [T, topk_routed] fp32
    w          = sqrt(softplus(logits))
    w          = w / w.sum(-1, keepdim=True)
    shared_w   = full((T, num_fused_shared), 1/routed_scaling_factor)
    shared_ids = num_routed_experts + arange(num_fused_shared)
    weights    = cat([w, shared_w], -1).float()
    ids        = cat([expert_ids, shared_ids], -1).int32

There is no top-k search: the expert set is looked up in a precomputed table,
so the whole operator is a gather, a small elementwise chain and a row reduction.

Why fusing wins
---------------
The reference materialises several `[T, topk_routed]` intermediates in fp32 plus
two `cat`s and their outputs.  One fused program per token reads the token's
`topk_routed` table entries, gathers that many logits, normalises and writes both
outputs once.

Four launch arms (GCU tuning)
-----------------------------
The per-token kernel (one program per row, grid=(T,)) is optimal for tiny
batches: its three-level dependent load chain (tid -> table row -> logit row)
hides behind the launch floor, and every value lives in registers, so the
row normalisation needs no second pass.  For larger T, however, T
single-token programs form a micro-CTA wave train whose per-program serial
chain dominates the wall clock (measured ~134 ns/token marginal on the
framework bench).

The row-batch kernel batches BLOCK_T tokens per program with the tokens as
the lane dimension:

  * `tid` becomes one contiguous [BLOCK_T] vector load,
  * the table and logit gathers become per-k unrolled [BLOCK_T] gather ops
    (`TOPK` is constexpr, so `tl.static_range` fully unrolls both passes and
    the per-k loads issue independently),
  * the row sum becomes a [BLOCK_T] accumulator built over the unrolled k
    loop, and
  * every store is a 1-D [BLOCK_T] op with an affine row base.

Its row sum is an elementwise accumulation (cheap), but it only completes
after the last expert, so the normalised weights need a second pass that
re-issues both gathers (L2-hot) and re-runs the scoring math.  Ablation at
T=512/BLOCK_T=32 splits that pass into ~3.9 us re-gather chain + ~0.5 us
math + ~2.0 us store drain (of 12.1 us total).  Every in-kernel alternative
to the second pass was measured and lost at these shapes: re-reading the
just-stored weights pays a serial store->load round trip per column
(+1.2 us), and holding the weights in a register tile costs more in layout
ops than the re-gather it saves while BLOCK_T <= 64 (+3.2 us at 32), so the
two-pass row-batch kernel is a measured local optimum for 256 <= T <= 1024.

Between the two extremes the row-loop kernel wins: its lanes are one
token's TOPK experts (exactly the per-token kernel's lane layout) and
`tl.static_range` unrolls a handful of tokens per program (BLOCK_T 2 or 4).
The row sum is an in-lane `tl.sum(axis=0)` -> scalar, so each token's
normalisation completes in registers: the second pass does not exist.  The
cost is that the per-token chain (scalar token load, in-lane cross reduce)
serialises at ~1.2-1.4 us inside a program, which is why BLOCK_T stays
tiny; the grid then remains a fat-CTA form (<= 64 programs) that is immune
to the cross-process device-state drift that hits micro-CTA grids (see
knowledge/bench_protocol.md).  Measured in a registry-importing process
(same run): T=64 12.1 -> 6.3 us, T=128 11.2 -> 8.9 us; at T=256 the
row-batch arm is faster again (11.3 vs 14.2 us), which sets the boundary.

For T > 1024 the row-batch BLOCK_T clamps to 128 and the register-tile form
flips from liability to win: the tile's layout ops amortise over four times
more tokens while the second-pass re-gather chain grows with the lane
count.  `_hash_topk_regtile_kernel` keeps every w_k in a register-only
[BLOCK_K, BLOCK_T] tile (one `tl.where` per expert with a constant row
selector -- no divmod, no memory) and stores the normalised values after a
masked row-extraction sum, replacing the second pass entirely.  Measured:
T=2048 14.0 -> 12.9 us, T=4096 15.4 -> 13.8 us, T=8192 28.4 -> 25.3 us.

All memory ops stay 1-D on purpose in every arm: 2-D gather tiles lower
through ``memref_ext`` staging ops that a global ``gcu_compiler_opt`` dialect shim
(installed at import time by ``extend_attention.py`` to work around an external
tool mismatch for ``memref_ext.slice_start``) corrupts for unrelated kernels --
``memref_ext.slice_deslice_start`` with a trailing-zero operand segment gets
its segment list truncated and the external compiler rejects the op.  The 1-D
formulation used here never emits that op family, so the op compiles in any
process that has imported the full operator registry (see
compiler_issues/sglang_hash_topk_2d_gather_shim.md; the row-loop and
register-tile arms were additionally compile- and accuracy-verified in such
a process).

Arm boundaries: per-token below T=64 (launch-floor regime, unchanged -- its
bench shapes double as an in-run canary for the drift-sensitive process
state); row-loop for 64 <= T < 256 (BLOCK_T=2 below 128, 4 above -- measured
crossover); row-batch two-pass for 256 <= T <= 1024 with
`BLOCK_T = clamp(next_pow2(cdiv(T, 16)), 16, 128)` keeping the program
count inside one ~48-program wave; register-tile with BLOCK_T=128 above
1024.  First-round harness numbers (do_bench median, L2 flushed): T=512
39 -> 12 us, T=4096 289 -> 15 us (per-token arm vs row-batch arm); the
framework-bench numbers are recorded in the baseline report.

Numerical form of softplus
--------------------------
``log(1 + exp(x))`` overflows fp32 above ``x ~ 88``.  The stable identity

    softplus(x) = max(x, 0) + log(1 + exp(-|x|))

is used instead, with ``|x|`` written as ``maximum(x, -x)`` so that no ``tl.abs``
(or any other unlisted intrinsic) is needed.  This is the same expression the
reference's ``F.softplus`` evaluates.

Masked lanes
------------
The gather for lanes ``j >= topk_routed`` is loaded with ``other=-inf``.  Then
``softplus(-inf) = log(1 + 0) = 0`` and ``sqrt(0) = 0``, so those lanes
contribute nothing to the row sum and write ``0`` -- no ``tl.where`` is needed,
which matters because ``where``-based accumulation miscompiled on Kunlunxin in an
earlier task.  In the tiled arm, out-of-range rows (``rows >= T``) are masked on
every load/store the same way; their row sum is 0 and the division yields NaN,
but those lanes are never stored.

Index arithmetic
----------------
`topk_routed` and the two widths are `constexpr`, so every address is
`row * WIDTH + lane` with no runtime integer division or remainder -- the two
operations the strict backends cannot handle.  Rows map to `program_id(axis=0)`,
so no `grid.y` (whose >255 limit breaks the Enflame launch).  All loads and
stores are 1-D in both arms: no 2-D broadcast, no `tl.full`, no scatter store,
no bitcast.  All index arithmetic stays in int32 (same domain as the original
kernel: max offset `T * num_routed` must fit in int32, which holds for any
realistic router width).

No fallback of any kind: no `try`/`except`, no device inspection, and no PyTorch
operator on the executed path."""
import torch
import triton
import triton.language as tl

__all__ = ["hash_topk"]

_ROWLOOP_MIN_TOKENS = 64
_ROWLOOP_MAX_TOKENS = 256
_REGTILE_MIN_TOKENS = 1024


@triton.jit
def _hash_topk_kernel(
    rl_ptr,
    tid_ptr,
    table_ptr,
    w_ptr,
    id_ptr,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
    NR: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    token = tl.load(tid_ptr + row)

    # ---- routed slots -------------------------------------------------
    j = tl.arange(0, BLOCK_K)
    m = j < TOPK
    eid = tl.load(table_ptr + token * TOPK + j, mask=m, other=0)

    # gather the logits of this token's experts; masked lanes -> -inf so that
    # softplus(-inf) == 0 and they drop out of the reduction.
    logit = tl.load(rl_ptr + row * NR + eid, mask=m, other=float("-inf")).to(
        tl.float32
    )

    ax = tl.maximum(logit, -logit)  # |logit| without tl.abs
    sp = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-ax))
    w = tl.sqrt(sp)

    total = tl.sum(w, axis=0)
    wn = w / total

    tl.store(w_ptr + row * W_OUT + j, wn, mask=m)
    tl.store(id_ptr + row * W_OUT + j, eid, mask=m)

    # ---- fused shared slots -------------------------------------------
    s = tl.arange(0, BLOCK_S)
    # `NSHARED` is the true count, *not* clamped to 1: when it is 0 the mask is
    # empty and nothing is written.  Clamping it to 1 while the host computes
    # ``W_OUT = TOPK + 0`` would store one float into ``row * TOPK + TOPK``,
    # i.e. column 0 of the *next* row -- an out-of-bounds overwrite that
    # corrupts the routed ids of the following token.
    ms = s < NSHARED
    shared_w = tl.zeros([BLOCK_S], dtype=tl.float32) + INV_SCALE
    tl.store(w_ptr + row * W_OUT + TOPK + s, shared_w, mask=ms)
    tl.store(id_ptr + row * W_OUT + TOPK + s, (NR + s).to(tl.int32), mask=ms)


@triton.jit
def _hash_topk_tiled_kernel(
    rl_ptr,
    tid_ptr,
    table_ptr,
    w_ptr,
    id_ptr,
    T,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
    NR: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    # One program covers BLOCK_T consecutive tokens.  All memory ops are 1-D
    # [BLOCK_T] vectors -- the same lowering family as the per-token kernel
    # below -- because 2-D gather tiles lower through memref_ext staging ops
    # that a global gcu_compiler_opt dialect shim installed by another op
    # module corrupts (see compiler_issues/sglang_hash_topk_2d_gather_shim.md).
    # TOPK/NSHARED are constexpr, so both static_range loops fully unroll and
    # their per-k loads issue independently.  The expert ids do not depend on
    # the row sum, so pass 1 stores them while accumulating the total; pass 2
    # re-reads the two gather sources (L2-hot by then) and rewrites the
    # normalised weights.
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rmask = rows < T

    tokens = tl.load(tid_ptr + rows, mask=rmask, other=0)

    total = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        eid = tl.load(table_ptr + tokens * TOPK + k, mask=rmask, other=0)
        logit = tl.load(rl_ptr + rows * NR + eid, mask=rmask, other=float("-inf")).to(
            tl.float32
        )
        ax = tl.maximum(logit, -logit)
        sp = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-ax))
        w = tl.sqrt(sp)
        total += w
        tl.store(id_ptr + rows * W_OUT + k, eid, mask=rmask)
    for k in tl.static_range(TOPK):
        eid = tl.load(table_ptr + tokens * TOPK + k, mask=rmask, other=0)
        logit = tl.load(rl_ptr + rows * NR + eid, mask=rmask, other=float("-inf")).to(
            tl.float32
        )
        ax = tl.maximum(logit, -logit)
        sp = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-ax))
        w = tl.sqrt(sp)
        tl.store(w_ptr + rows * W_OUT + k, w / total, mask=rmask)
    for s in tl.static_range(NSHARED):
        shared_w = tl.zeros((BLOCK_T,), dtype=tl.float32) + INV_SCALE
        tl.store(w_ptr + rows * W_OUT + TOPK + s, shared_w, mask=rmask)
        shared_id = tl.zeros((BLOCK_T,), dtype=tl.int32) + (NR + s)
        tl.store(id_ptr + rows * W_OUT + TOPK + s, shared_id, mask=rmask)


@triton.jit
def _hash_topk_rowloop_kernel(
    rl_ptr,
    tid_ptr,
    table_ptr,
    w_ptr,
    id_ptr,
    T,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
    NR: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    # Lanes are one token's TOPK experts (the per-token kernel's lane layout);
    # the token loop is a static_range over BLOCK_T=2/4 tokens per program.
    # The row sum is an in-lane tl.sum(axis=0) -> scalar, so each token's
    # normalisation completes in registers: no second pass, no re-gather.
    # Out-of-range rows (row >= T) clamp their loads to the last row --
    # always in-bounds, finite -- and mask every store with (row < T).
    pid = tl.program_id(axis=0)
    j = tl.arange(0, BLOCK_K)
    jm = j < TOPK
    row0 = pid * BLOCK_T
    for t in tl.static_range(BLOCK_T):
        row = row0 + t
        rowc = tl.minimum(row, T - 1)
        smask = row < T
        token = tl.load(tid_ptr + rowc)
        eid = tl.load(table_ptr + token * TOPK + j, mask=jm, other=0)
        logit = tl.load(rl_ptr + rowc * NR + eid, mask=jm, other=float("-inf")).to(
            tl.float32
        )
        ax = tl.maximum(logit, -logit)
        sp = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-ax))
        w = tl.sqrt(sp)
        total = tl.sum(w, axis=0)
        tl.store(w_ptr + row * W_OUT + j, w / total, mask=jm & smask)
        tl.store(id_ptr + row * W_OUT + j, eid, mask=jm & smask)
    rows = row0 + tl.arange(0, BLOCK_T)
    rmask = rows < T
    for s in tl.static_range(NSHARED):
        shared_w = tl.zeros((BLOCK_T,), dtype=tl.float32) + INV_SCALE
        tl.store(w_ptr + rows * W_OUT + TOPK + s, shared_w, mask=rmask)
        shared_id = tl.zeros((BLOCK_T,), dtype=tl.int32) + (NR + s)
        tl.store(id_ptr + rows * W_OUT + TOPK + s, shared_id, mask=rmask)


@triton.jit
def _hash_topk_regtile_kernel(
    rl_ptr,
    tid_ptr,
    table_ptr,
    w_ptr,
    id_ptr,
    T,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
    NR: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    # Row-batch form (lanes = tokens) that keeps every w_k in a register-only
    # [BLOCK_K, BLOCK_T] tile -- one tl.where per expert with a constant row
    # selector, no divmod, no memory -- instead of re-gathering in a second
    # pass.  Tile rows >= TOPK stay zero and are never selected (k < TOPK) nor
    # extracted, so a non-power-of-two TOPK only costs dead registers.  At
    # BLOCK_T=128 the tile layout ops amortise enough to beat the two-pass
    # re-gather chain; every load/store stays 1-D, so the kernel stays in the
    # shim-immune lowering family (see the module docstring).
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rmask = rows < T
    tokens = tl.load(tid_ptr + rows, mask=rmask, other=0)
    kar = tl.arange(0, BLOCK_K)
    total = tl.zeros((BLOCK_T,), dtype=tl.float32)
    w2d = tl.zeros((BLOCK_K, BLOCK_T), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        eid = tl.load(table_ptr + tokens * TOPK + k, mask=rmask, other=0)
        logit = tl.load(rl_ptr + rows * NR + eid, mask=rmask, other=float("-inf")).to(
            tl.float32
        )
        ax = tl.maximum(logit, -logit)
        sp = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-ax))
        w = tl.sqrt(sp)
        total += w
        tl.store(id_ptr + rows * W_OUT + k, eid, mask=rmask)
        w2d = tl.where(kar[:, None] == k, w[None, :], w2d)
    inv_total = 1.0 / total
    wn2d = w2d * inv_total[None, :]
    for k in tl.static_range(TOPK):
        wn_k = tl.sum(tl.where(kar[:, None] == k, wn2d, 0.0), axis=0)
        tl.store(w_ptr + rows * W_OUT + k, wn_k, mask=rmask)
    for s in tl.static_range(NSHARED):
        shared_w = tl.zeros((BLOCK_T,), dtype=tl.float32) + INV_SCALE
        tl.store(w_ptr + rows * W_OUT + TOPK + s, shared_w, mask=rmask)
        shared_id = tl.zeros((BLOCK_T,), dtype=tl.int32) + (NR + s)
        tl.store(id_ptr + rows * W_OUT + TOPK + s, shared_id, mask=rmask)


def _tiled_block_t(num_tokens: int) -> int:
    # Keep the program count inside one ~48-program wave while amortising the
    # per-program dependent-load chain over as many tokens as possible.
    # Measured sweet spots: T=128 -> BT=16, T=256 -> BT=16, T=512 -> BT=32,
    # T=4096 -> BT=128 (do_bench kernel-mode medians).
    block = triton.next_power_of_2(-(-num_tokens // 16))
    return min(max(block, 16), 128)


def hash_topk(
    router_logits,
    input_ids,
    tid2eid,
    num_fused_shared_experts,
    routed_scaling_factor,
    scoring_func,
):
    """Hash-routed MoE gate: gather, sqrt-softplus scoring, row renormalisation."""
    assert scoring_func == "sqrtsoftplus", scoring_func

    router_logits = router_logits.contiguous()
    input_ids = input_ids.contiguous()
    tid2eid = tid2eid.contiguous()

    num_tokens, num_routed = router_logits.shape
    topk_routed = tid2eid.shape[1]
    n_shared = int(num_fused_shared_experts)
    w_out = topk_routed + n_shared

    weights = torch.empty(
        (num_tokens, w_out), dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty(
        (num_tokens, w_out), dtype=torch.int32, device=router_logits.device
    )
    if num_tokens == 0:
        return weights, ids

    if num_tokens < _ROWLOOP_MIN_TOKENS:
        _hash_topk_kernel[(num_tokens,)](
            router_logits,
            input_ids,
            tid2eid,
            weights,
            ids,
            TOPK=topk_routed,
            NSHARED=n_shared,
            NR=num_routed,
            W_OUT=w_out,
            BLOCK_K=triton.next_power_of_2(max(topk_routed, 1)),
            BLOCK_S=triton.next_power_of_2(max(n_shared, 1)),
            INV_SCALE=1.0 / float(routed_scaling_factor),
            num_warps=1,
        )
        return weights, ids

    if num_tokens < _ROWLOOP_MAX_TOKENS:
        # Measured crossover: BLOCK_T=2 wins below 128 tokens (T=64 6.3 vs
        # 8.0 us), BLOCK_T=4 above (T=128 8.9 vs 10.1 us).
        block_t = 2 if num_tokens < 128 else 4
        _hash_topk_rowloop_kernel[(-(-num_tokens // block_t),)](
            router_logits,
            input_ids,
            tid2eid,
            weights,
            ids,
            num_tokens,
            TOPK=topk_routed,
            NSHARED=n_shared,
            NR=num_routed,
            W_OUT=w_out,
            BLOCK_T=block_t,
            BLOCK_K=triton.next_power_of_2(max(topk_routed, 1)),
            INV_SCALE=1.0 / float(routed_scaling_factor),
            num_warps=1,
        )
        return weights, ids

    if num_tokens <= _REGTILE_MIN_TOKENS:
        block_t = _tiled_block_t(num_tokens)
        _hash_topk_tiled_kernel[(-(-num_tokens // block_t),)](
            router_logits,
            input_ids,
            tid2eid,
            weights,
            ids,
            num_tokens,
            TOPK=topk_routed,
            NSHARED=n_shared,
            NR=num_routed,
            W_OUT=w_out,
            BLOCK_T=block_t,
            INV_SCALE=1.0 / float(routed_scaling_factor),
            num_warps=1,
        )
        return weights, ids

    _hash_topk_regtile_kernel[(-(-num_tokens // 128),)](
        router_logits,
        input_ids,
        tid2eid,
        weights,
        ids,
        num_tokens,
        TOPK=topk_routed,
        NSHARED=n_shared,
        NR=num_routed,
        W_OUT=w_out,
        BLOCK_T=128,
        BLOCK_K=triton.next_power_of_2(max(topk_routed, 1)),
        INV_SCALE=1.0 / float(routed_scaling_factor),
        num_warps=1,
    )
    return weights, ids
