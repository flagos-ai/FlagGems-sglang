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

import triton
import triton.language as tl

_KEY_MAX_TOKENS = 512
_MANY_ROWS = 4
# Expert-chunk width of the scan path.
_SCAN_CHUNK = 64
_SCAN_ROWS = 4


@triton.jit
def _topk_sigmoid_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    num_tokens,
    num_experts,
    routed_scaling_factor,
    E: tl.constexpr,  # runtime row length, kept constexpr for address math
    E_PAD: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EVICT: tl.constexpr,
):
    """Small-``num_tokens`` path: int-key selection sweep."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rmask = rows < num_tokens

    cols = tl.arange(0, E_PAD)
    cmask = rmask[:, None] & (cols < E)[None, :]

    if EVICT:
        logits = tl.load(
            gating_ptr + rows[:, None] * E + cols[None, :],
            mask=cmask,
            other=float("-inf"),
            eviction_policy="evict_first",
        )
    else:
        logits = tl.load(
            gating_ptr + rows[:, None] * E + cols[None, :],
            mask=cmask,
            other=float("-inf"),
        )
    b = logits.to(tl.float32).to(tl.int32, bitcast=True)
    key = tl.where(b >= 0, b, (~b) | (-2147483648))

    # Inverted column: a tie on `key` resolves to the smallest expert index.
    col_inv = (E_PAD - 1 - cols).to(tl.int32)

    tcols = tl.arange(0, K_PAD)
    out_mask = rmask[:, None] & (tcols < K)[None, :]

    ids = tl.zeros((BLOCK_T, K_PAD), tl.int32)
    for i in tl.static_range(K_PAD):
        m = tl.max(key, axis=1)
        eq = key == m[:, None]
        idx = tl.min(tl.where(eq, col_inv[None, :], 0x7FFFFFFF), axis=1)
        ids = tl.where(
            ((i < K) & (tcols == i))[None, :], (E_PAD - 1 - idx)[:, None], ids
        )
        # Poison the winner's lane so the next sweep picks the runner-up.
        key = tl.where(eq & (i < K), -2147483648, key)

    sel = tl.load(
        gating_ptr + rows[:, None] * E + ids, mask=out_mask, other=0.0
    )
    vals = tl.sigmoid(sel.to(tl.float32))
    vals = tl.where(out_mask, vals, 0.0)

    if RENORM:
        denom = tl.sum(vals, axis=1) + 1e-20
        vals = vals * (routed_scaling_factor / denom)[:, None]

    offs = rows[:, None] * K + tcols[None, :]
    tl.store(weights_ptr + offs, vals, mask=out_mask)
    tl.store(ids_ptr + offs, ids, mask=out_mask)


@triton.jit
def _topk_sigmoid_many_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    RENORM: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
):
    """Large-``num_tokens`` path: ``R`` rows per program, flat."""
    neg = -3.4028234663852886e38
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    slots = tl.arange(0, A)
    live = (rows[:, None] < M) & (cols[None, :] < N)
    # Clamp the row/col indices into range and mask, so the load stays in
    # bounds for the ragged last block.
    raw = tl.load(
        gating_ptr
        + tl.minimum(rows, M - 1)[:, None] * N
        + tl.minimum(cols, N - 1)[None, :]
    )
    current = tl.where(live, raw, neg)
    picked = tl.full((R, A), neg, tl.float32)
    indices = tl.full((R, A), 0, tl.int32)
    best = tl.max(current, axis=1)
    for slot in tl.static_range(K):
        chosen = current == best[:, None]
        winner = tl.min(tl.where(chosen, cols[None, :], B), axis=1)
        picked = tl.where(slots[None, :] == slot, best[:, None], picked)
        indices = tl.where(slots[None, :] == slot, winner[:, None], indices)
        current = tl.where(chosen, neg, current)
        best = tl.max(current, axis=1)
    weights = tl.where(slots[None, :] < K, tl.sigmoid(picked), 0.0)
    if RENORM:
        weights = (
            weights * (SCALE / (tl.sum(weights, axis=1) + 1.0e-20))[:, None]
        )
    keep = (rows[:, None] < M) & (slots[None, :] < K)
    tl.store(weights_ptr + rows[:, None] * K + slots[None, :], weights, keep)
    tl.store(
        ids_ptr + rows[:, None] * K + slots[None, :],
        indices.to(tl.int32),
        keep,
    )


@triton.jit
def _topk_sigmoid_scan_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    RENORM: tl.constexpr,
    SCALE: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
    T: tl.constexpr,
):
    """Small-``num_experts`` path: deterministic block scan."""
    neg = -3.4028234663852886e38
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, T)
    slots = tl.arange(0, A)
    picked = tl.full((R, A), neg, tl.float32)
    indices = tl.full((R, A), 0, tl.int32)
    bound = tl.full((R,), 3.4028234663852886e38, tl.float32)
    for slot in tl.static_range(K):
        top = tl.full((R,), neg, tl.float32)
        pos = tl.full((R,), N, tl.int32)
        for start in tl.static_range(0, N, T):
            cols = start + lanes
            seg = tl.load(gating_ptr + rows[:, None] * N + cols[None, :])
            cand = tl.where(seg < bound[:, None], seg, neg)
            local = tl.max(cand, axis=1)
            index = tl.min(
                tl.where(cand == local[:, None], cols[None, :], N), axis=1
            )
            better = local > top
            pos = tl.where(better, index, pos)
            top = tl.where(better, local, top)
        bound = top
        picked = tl.where(slots[None, :] == slot, top[:, None], picked)
        indices = tl.where(slots[None, :] == slot, pos[:, None], indices)
    weights = tl.where(slots[None, :] < K, tl.sigmoid(picked), 0.0)
    if RENORM:
        weights = (
            weights * (SCALE / (tl.sum(weights, axis=1) + 1.0e-20))[:, None]
        )
    keep = slots[None, :] < K
    tl.store(weights_ptr + rows[:, None] * K + slots[None, :], weights, keep)
    tl.store(ids_ptr + rows[:, None] * K + slots[None, :], indices, keep)


def topk_sigmoid(
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    num_tokens, num_experts = gating_output.shape
    k = topk_weights.shape[1]
    if num_tokens == 0 or k == 0:
        return topk_weights, topk_ids

    e_pad = triton.next_power_of_2(num_experts)
    k_pad = triton.next_power_of_2(k)
    scale = float(routed_scaling_factor)

    if num_tokens <= _KEY_MAX_TOKENS:
        # Launch-latency regime: int-key sweep, launch config by shape.
        if num_tokens <= 8:
            block_t, num_warps, evict = 4, 4, False
        elif num_tokens <= 64:
            block_t, num_warps, evict = 4, 2, False
        elif num_tokens <= 256:
            block_t, num_warps, evict = 1, 1, False
        else:
            block_t, num_warps, evict = 8, 4, False
        _topk_sigmoid_kernel[(triton.cdiv(num_tokens, block_t),)](
            gating_output,
            topk_weights,
            topk_ids,
            num_tokens,
            num_experts,
            scale,
            E=num_experts,
            E_PAD=e_pad,
            K=k,
            K_PAD=k_pad,
            RENORM=renormalize,
            BLOCK_T=block_t,
            EVICT=evict,
            num_warps=num_warps,
            num_stages=1,
        )
    elif e_pad <= _SCAN_CHUNK and num_tokens % _SCAN_ROWS == 0:
        # Narrow expert axis: deterministic block scan.
        _topk_sigmoid_scan_kernel[(num_tokens // _SCAN_ROWS,)](
            gating_output,
            topk_weights,
            topk_ids,
            num_experts,
            k,
            renormalize,
            scale,
            max(2, k_pad),
            _SCAN_ROWS,
            e_pad,
            num_warps=1,
            num_stages=1,
        )
    else:
        # Wide regime: R rows per program, flat selection.
        _topk_sigmoid_many_kernel[(triton.cdiv(num_tokens, _MANY_ROWS),)](
            gating_output,
            topk_weights,
            topk_ids,
            num_tokens,
            num_experts,
            k,
            renormalize,
            scale,
            e_pad,
            max(2, k_pad),
            _MANY_ROWS,
            num_warps=1,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]
