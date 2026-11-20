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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _route(
    X,
    S,
    W,
    I,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    SS: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    slots = tl.arange(0, A)
    full_tile = (
        T >= 512
        and E <= 128
        and E == B
        and (T % R == 0)
        and (K == A)
        and (X0 == E)
        and (X1 == 1)
    )
    if full_tile:
        vals = tl.load(
            X + rows[:, None] * X0 + cols[None, :] * X1,
            offset_state_policy="continuous",
        ).to(tl.float32)
        active = tl.full((R, B), 1, tl.int1)
    elif T >= 64 and E == B and (T % R == 0) and (X0 == E) and (X1 == 1):
        vals = tl.load(
            X
            + tl.minimum(rows, T - 1)[:, None] * X0
            + tl.minimum(cols, E - 1)[None, :] * X1,
            offset_state_policy="continuous",
        ).to(tl.float32)
        active = tl.broadcast_to(cols[None, :] < E, (R, B))
        vals = tl.where(active, vals, -float("inf"))
    else:
        vals = tl.load(
            X
            + tl.minimum(rows, T - 1)[:, None] * X0
            + tl.minimum(cols, E - 1)[None, :] * X1
        ).to(tl.float32)
        active = tl.broadcast_to(cols[None, :] < E, (R, B))
        vals = tl.where(active, vals, -float("inf"))
    picked = tl.full((R, A), 0, tl.float32)
    indices = tl.full((R, A), 0, tl.float32)
    first = tl.max(vals, 1)
    denom = tl.full((R,), 0, tl.float32)
    for slot in tl.static_range(K):
        best = tl.max(tl.where(active, vals, -float("inf")), 1)
        winner_float = tl.min(
            tl.where(
                active & (vals == best[:, None]),
                cols[None, :].to(tl.float32),
                float(B),
            ),
            1,
        )
        winner = winner_float.to(tl.int32)
        numer = tl.exp(best - first)
        denom += numer
        if T >= 512 and (E <= 128 or T > 512):
            selected_scale = tl.load(S + winner * SS).to(tl.float32)
            weighted_numer = numer * selected_scale
        else:
            weighted_numer = numer
        picked = tl.where(
            slots[None, :] == slot, weighted_numer[:, None], picked
        )
        indices = tl.where(
            slots[None, :] == slot, winner_float[:, None], indices
        )
        active = active & (
            cols[None, :].to(tl.float32) != winner_float[:, None]
        )
    indices = indices.to(tl.int32)
    if T >= 512 and (E <= 128 or T > 512):
        weights = picked / denom[:, None]
    else:
        scale = tl.load(S + indices * SS).to(tl.float32)
        weights = picked / denom[:, None] * scale
    if full_tile:
        tl.store(W + rows[:, None] * K + slots[None, :], weights)
        tl.store(I + rows[:, None] * K + slots[None, :], indices)
    else:
        live = (rows[:, None] < T) & (slots[None, :] < K)
        tl.store(W + rows[:, None] * K + slots[None, :], weights, live)
        tl.store(I + rows[:, None] * K + slots[None, :], indices, live)


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t > 0 and topk > 0:
        b = triton.next_power_of_2(e)
        r = 1 if t <= 8 else min(64, max(1, 16384 // b), max(1, t // 4))
        if t > 512:
            r = min(r, max(1, 8192 // b))
        r = 1 << r.bit_length() - 1
        options = {}
        if (
            t >= 512
            and e == 256
            and topk == 8
            and t % 64 == 0
            and gating_output.stride(0) == e
            and gating_output.stride(1) == 1
        ):
            r = 64
            options["groups_per_cluster"] = 64
        _route[triton.cdiv(t, r),](
            gating_output,
            per_expert_scale,
            weights,
            ids,
            t,
            e,
            topk,
            gating_output.stride(0),
            gating_output.stride(1),
            per_expert_scale.stride(0),
            b,
            max(2, triton.next_power_of_2(topk)),
            r,
            num_warps=1,
            num_stages=1,
            **options,
        )
    return (weights, ids)


__all__ = ["gemma4_fused_routing"]
