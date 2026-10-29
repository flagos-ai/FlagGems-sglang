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
def _prepare(
    C,
    V,
    S,
    M,
    E: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    BE: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    es = tl.arange(0, BE)
    counts = tl.load(C + es * CS, es < E, other=0)
    start = tl.sum(tl.where(es < e, counts, 0), 0)
    count = tl.load(C + e * CS)
    valid = tl.load(V + e * VS)
    tl.store(S + e * SS, start)
    offs = tl.arange(0, B)
    for base in range(0, count, B):
        r = base + offs
        tl.store(M + (start + r) * MS, tl.where(r < valid, e, -1), r < count)


@triton.jit
def _dispatch(
    X,
    XS,
    K,
    S,
    Y,
    YS,
    O,
    T: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    B: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
):
    cols = tl.arange(0, B)
    slots = tl.arange(0, BK)
    scales = tl.arange(0, BS)
    for token in range(tl.program_id(0), T, tl.num_programs(0)):
        e = tl.load(K + token * K0 + slots * K1, slots < NK, other=-1)
        valid = (slots < NK) & (e >= 0)
        dest = tl.atomic_add(S + e * S0, 1, valid, sem="relaxed")
        tl.store(O + token * O0 + slots * O1, dest, valid)
        for tile in range((H * Q + B - 1) // B):
            c = tile * B + cols
            value = tl.load(
                X + token * X0 + (c // Q) * X1 + c % Q, c < H * Q, other=0
            )
            tl.store(
                Y
                + dest[:, None] * Y0
                + (c[None, :] // Q) * Y1
                + c[None, :] % Q,
                value[None, :],
                valid[:, None] & (c[None, :] < H * Q),
            )
        if HAS_SCALE:
            scale = tl.load(
                XS + token * XS0 + scales * XS1, scales < D, other=0
            )
            tl.store(
                YS + dest[:, None] * YS0 + scales[None, :] * YS1,
                scale[None, :],
                valid[:, None] & (scales[None, :] < D),
            )


@triton.jit
def _dispatch_grouped(
    X,
    XS,
    K,
    S,
    Y,
    YS,
    O,
    T: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    B: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
):
    cols = tl.arange(0, B)
    slots = tl.arange(0, BK)
    scales = tl.arange(0, BS)
    lanes = tl.arange(0, 4 * BK)
    experts = tl.arange(0, BE)
    for group in range(tl.program_id(0), (T + 3) // 4, tl.num_programs(0)):
        tokens = group * 4 + lanes // BK
        ks = lanes % BK
        ids = tl.load(
            K + tokens * K0 + ks * K1, (tokens < T) & (ks < NK), other=-1
        )
        hits = (experts[:, None] < E) & (ids[None, :] == experts[:, None])
        counts = tl.sum(hits.to(tl.int32), 1)
        ranks = tl.cumsum(hits.to(tl.int32), 1) - 1
        bases = tl.atomic_add(
            S + experts * S0,
            counts,
            (experts < E) & (counts > 0),
            sem="relaxed",
        )
        destinations = tl.sum(tl.where(hits, bases[:, None] + ranks, 0), 0)
        for i in tl.static_range(4):
            token = group * 4 + i
            if token < T:
                indices = i * BK + slots
                e = tl.gather(ids, indices, 0)
                dest = tl.gather(destinations, indices, 0)
                valid = (slots < NK) & (e >= 0)
                tl.store(O + token * O0 + slots * O1, dest, valid)
                for tile in range((H * Q + B - 1) // B):
                    c = tile * B + cols
                    value = tl.load(
                        X + token * X0 + (c // Q) * X1 + c % Q,
                        c < H * Q,
                        other=0,
                    )
                    tl.store(
                        Y
                        + dest[:, None] * Y0
                        + (c[None, :] // Q) * Y1
                        + c[None, :] % Q,
                        value[None, :],
                        valid[:, None] & (c[None, :] < H * Q),
                    )
                if HAS_SCALE:
                    scale = tl.load(
                        XS + token * XS0 + scales * XS1, scales < D, other=0
                    )
                    tl.store(
                        YS + dest[:, None] * YS0 + scales[None, :] * YS1,
                        scale[None, :],
                        valid[:, None] & (scales[None, :] < D),
                    )


def ep_scatter(
    recv_x,
    recv_x_scale,
    recv_topk,
    num_recv_tokens_per_expert,
    num_valid_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_scale,
    m_indices,
    output_index,
):
    e = num_recv_tokens_per_expert.numel()
    t, k = recv_topk.shape
    h = recv_x.shape[1]
    q = recv_x.element_size()
    has_scale = recv_x_scale is not None
    d = recv_x_scale.shape[1] if has_scale else 0
    if e:
        _prepare[(e,)](
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            e,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
            triton.next_power_of_2(e),
            128,
        )
    if t and k:
        if (
            0 < e <= 16
            and 1 <= k <= 8
            and (t + 3) // 4 >= 128
            and 4 * k >= 2 * e
        ):
            _dispatch_grouped[(min((t + 3) // 4, 1024),)](
                triton.reinterpret(recv_x, tl.uint8),
                recv_x_scale,
                recv_topk,
                expert_start_loc,
                triton.reinterpret(output_tensor, tl.uint8),
                output_tensor_scale,
                output_index,
                t,
                k,
                h,
                d,
                recv_x.stride(0) * q,
                recv_x.stride(1) * q,
                q,
                recv_x_scale.stride(0) if has_scale else 0,
                recv_x_scale.stride(1) if has_scale else 0,
                recv_topk.stride(0),
                recv_topk.stride(1),
                expert_start_loc.stride(0),
                output_tensor.stride(0) * q,
                output_tensor.stride(1) * q,
                output_tensor_scale.stride(0) if has_scale else 0,
                output_tensor_scale.stride(1) if has_scale else 0,
                output_index.stride(0),
                output_index.stride(1),
                has_scale,
                min(1024, triton.next_power_of_2(h * q)),
                triton.next_power_of_2(max(d, 1)),
                triton.next_power_of_2(k),
                e,
                triton.next_power_of_2(e),
            )
        else:
            _dispatch[(min(t, 1024),)](
                triton.reinterpret(recv_x, tl.uint8),
                recv_x_scale,
                recv_topk,
                expert_start_loc,
                triton.reinterpret(output_tensor, tl.uint8),
                output_tensor_scale,
                output_index,
                t,
                k,
                h,
                d,
                recv_x.stride(0) * q,
                recv_x.stride(1) * q,
                q,
                recv_x_scale.stride(0) if has_scale else 0,
                recv_x_scale.stride(1) if has_scale else 0,
                recv_topk.stride(0),
                recv_topk.stride(1),
                expert_start_loc.stride(0),
                output_tensor.stride(0) * q,
                output_tensor.stride(1) * q,
                output_tensor_scale.stride(0) if has_scale else 0,
                output_tensor_scale.stride(1) if has_scale else 0,
                output_index.stride(0),
                output_index.stride(1),
                has_scale,
                min(1024, triton.next_power_of_2(h * q)),
                triton.next_power_of_2(max(d, 1)),
                triton.next_power_of_2(k),
            )
    return (
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )


__all__ = ["ep_scatter"]
