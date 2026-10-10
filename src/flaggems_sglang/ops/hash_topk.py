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
earlier task.

Index arithmetic
----------------
`topk_routed` and the two widths are `constexpr`, so every address is
`row * WIDTH + lane` with no runtime integer division or remainder -- the two
operations the strict backends cannot handle.  Rows map to `program_id(axis=0)`,
so no `grid.y` (whose >255 limit breaks the Enflame launch).  All loads are 1-D:
no 2-D broadcast, no `tl.full`, no scatter store, no bitcast.

No fallback of any kind: no `try`/`except`, no device inspection, and no PyTorch
operator on the executed path."""
import torch
import triton
import triton.language as tl

__all__ = ["hash_topk"]


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
    logit = tl.load(rl_ptr + row * NR + eid, mask=m, other=float("-inf")).to(tl.float32)

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
