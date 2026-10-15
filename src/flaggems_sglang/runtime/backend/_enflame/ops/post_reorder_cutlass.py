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

import triton
import triton.language as tl


@triton.jit
def _pick(v, lanes, k):
    return tl.sum(tl.where(lanes == k, v, 0))


@triton.jit
def _reorder(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    S: tl.constexpr,
    NLE: tl.constexpr,
    ROWS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    B: tl.constexpr,
    NT: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
):
    t0 = tl.program_id(0) // NT * T
    h = tl.program_id(0) % NT * B + tl.arange(0, B)
    keep = h < H
    lanes = tl.arange(0, KP)
    for j in tl.static_range(T):
        t = t0 + j
        alive = t < N
        live = alive & (lanes < K)
        dv = tl.load(I + t * I0 + lanes * I1, live, other=0).to(tl.int32)
        ev = tl.load(E + t * E0 + lanes * E1, live, other=NLE).to(tl.int32)
        wv = tl.load(W + t * W0 + lanes * W1, live, other=0).to(tl.float32)
        wv = tl.where(ev != NLE, wv, 0.0)
        rv = tl.where(dv < 0, dv + ROWS, dv)
        d0 = _pick(rv, lanes, 0)
        d1 = _pick(rv, lanes, 1)
        w0 = _pick(wv, lanes, 0).to(tl.float32)
        w1 = _pick(wv, lanes, 1).to(tl.float32)
        if KP > 2:
            d2 = _pick(rv, lanes, 2)
            d3 = _pick(rv, lanes, 3)
            w2 = _pick(wv, lanes, 2).to(tl.float32)
            w3 = _pick(wv, lanes, 3).to(tl.float32)
        if KP > 4:
            d4 = _pick(rv, lanes, 4)
            d5 = _pick(rv, lanes, 5)
            d6 = _pick(rv, lanes, 6)
            d7 = _pick(rv, lanes, 7)
            w4 = _pick(wv, lanes, 4).to(tl.float32)
            w5 = _pick(wv, lanes, 5).to(tl.float32)
            w6 = _pick(wv, lanes, 6).to(tl.float32)
            w7 = _pick(wv, lanes, 7).to(tl.float32)
        v0 = tl.load(D + d0 * D0 + h * D1, keep, other=0).to(tl.float32)
        v1 = tl.load(D + d1 * D0 + h * D1, keep, other=0).to(tl.float32)
        if KP > 2:
            v2 = tl.load(D + d2 * D0 + h * D1, keep, other=0).to(tl.float32)
            v3 = tl.load(D + d3 * D0 + h * D1, keep, other=0).to(tl.float32)
        if KP > 4:
            v4 = tl.load(D + d4 * D0 + h * D1, keep, other=0).to(tl.float32)
            v5 = tl.load(D + d5 * D0 + h * D1, keep, other=0).to(tl.float32)
            v6 = tl.load(D + d6 * D0 + h * D1, keep, other=0).to(tl.float32)
            v7 = tl.load(D + d7 * D0 + h * D1, keep, other=0).to(tl.float32)
        a = v0 * w0 + v1 * w1
        if KP > 2:
            a = a + (v2 * w2 + v3 * w3)
        if KP > 4:
            a = a + (v4 * w4 + v5 * w5) + (v6 * w6 + v7 * w7)
        tl.store(
            O + t * O0 + h * O1, (a * S).to(O.dtype.element_ty), keep & alive
        )


def post_reorder_cutlass(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    if not 1 <= topk <= 8:
        return _fallback(
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            num_tokens,
            hidden_size,
            routed_scaling_factor,
        )
    out = output.new_empty((num_tokens, hidden_size))
    if num_tokens == 0 or hidden_size == 0:
        return out
    b = min(8192, triton.next_power_of_2(hidden_size))
    nt = triton.cdiv(hidden_size, b)
    t_pack = max(1, 8 // topk)
    grid = -(-num_tokens // t_pack) * nt
    _reorder[(grid,)](
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        hidden_size,
        topk,
        triton.next_power_of_2(topk),
        float(routed_scaling_factor),
        num_local_experts,
        down_output.shape[0],
        *down_output.stride(),
        *out.stride(),
        *src2dst.stride(),
        *topk_ids.stride(),
        *topk_weights.stride(),
        b,
        nt,
        t_pack,
        num_tokens,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _serial(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    NLE: tl.constexpr,
    ROWS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    B: tl.constexpr,
):
    t = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    for slot in tl.static_range(K):
        row = tl.load(I + t * I0 + slot * I1).to(tl.int32)
        row = tl.where(row < 0, row + ROWS, row)
        eid = tl.load(E + t * E0 + slot * E1)
        w = tl.load(W + t * W0 + slot * W1).to(tl.float32)
        w = tl.where(eid != NLE, w, 0.0)
        v = tl.load(D + row * D0 + col * D1, col < H, other=0).to(tl.float32)
        acc += v * w
    tl.store(O + t * O0 + col * O1, acc * S, col < H)


def _fallback(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    out = output.new_empty((num_tokens, hidden_size))
    if num_tokens == 0 or hidden_size == 0:
        return out
    b = 2048
    while b > 128 and num_tokens * ((hidden_size + b - 1) // b) < 1024:
        b //= 2
    _serial[(num_tokens, (hidden_size + b - 1) // b)](
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        hidden_size,
        topk,
        float(routed_scaling_factor),
        num_local_experts,
        down_output.shape[0],
        *down_output.stride(),
        *out.stride(),
        *src2dst.stride(),
        *topk_ids.stride(),
        *topk_weights.stride(),
        b,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["post_reorder_cutlass"]
