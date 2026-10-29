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
def _small(
    C,
    V,
    S,
    M,
    X,
    XS,
    K,
    Y,
    YS,
    O,
    E: tl.constexpr,
    T: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BE: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    es = tl.arange(0, BE)
    counts = tl.load(C + es * CS, es < E, 0)
    p = tl.arange(0, BN)
    ids = tl.load(K + (p // NK) * K0 + (p % NK) * K1, p < T * NK, -1)
    if pid < E:
        start = tl.sum(tl.where(es < pid, counts, 0), 0)
        count = tl.load(C + pid * CS)
        valid = tl.load(V + pid * VS)
        actual = tl.sum(((ids == pid) & (p < T * NK)).to(tl.int32), 0)
        tl.store(S + pid * SS, start + actual)
        m = tl.arange(0, 128)
        for base in range(0, count, 128):
            r = base + m
            tl.store(
                M + (start + r) * MS, tl.where(r < valid, pid, -1), r < count
            )
    if pid < T:
        token = pid.to(tl.int64)
        c = tl.arange(0, BH)
        scales = tl.arange(0, BS)
        row = tl.load(X + token * X0 + (c // Q) * X1 + c % Q, c < H * Q, 0)
        if HAS_SCALE:
            srow = tl.load(XS + token * XS0 + scales * XS1, scales < D, 0)
        slots = tl.arange(0, BK)
        experts = tl.load(K + token * K0 + slots * K1, slots < NK, -1)
        keep = (slots < NK) & (experts >= 0)
        starts = tl.sum(
            tl.where(es[None, :] < experts[:, None], counts[None, :], 0), 1
        )
        ranks = tl.sum(
            (
                (ids[None, :] == experts[:, None])
                & (p[None, :] < token * NK + slots[:, None])
            ).to(tl.int32),
            1,
        )
        dests = (starts + ranks).to(tl.int64)
        tl.store(
            Y
            + dests[:, None] * Y0
            + (c // Q)[None, :] * Y1
            + (c % Q)[None, :],
            row[None, :],
            keep[:, None] & (c < H * Q)[None, :],
        )
        if HAS_SCALE:
            tl.store(
                YS + dests[:, None] * YS0 + scales[None, :] * YS1,
                srow[None, :],
                keep[:, None] & (scales < D)[None, :],
            )
        tl.store(O + token * O0 + slots * O1, dests, keep)


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
    NKP: tl.constexpr,
    HB: tl.constexpr,
    BS: tl.constexpr,
):
    slots = tl.arange(0, NKP)
    cols = tl.arange(0, HB)
    scales = tl.arange(0, BS)
    for ti in range(tl.program_id(0), T, tl.num_programs(0)):
        token = ti.to(tl.int64)
        es = tl.load(K + token * K0 + slots * K1, slots < NK, other=-1)
        row = tl.load(
            X + token * X0 + (cols // Q) * X1 + cols % Q, cols < H * Q, other=0
        )
        if HAS_SCALE:
            srow = tl.load(
                XS + token * XS0 + scales * XS1, scales < D, other=0
            )
        keep = es >= 0
        dests = tl.atomic_add(S + es * S0, 1, mask=keep, sem="relaxed").to(
            tl.int64
        )
        tl.store(O + token * O0 + slots * O1, dests, keep)
        tl.store(
            Y
            + dests[:, None] * Y0
            + (cols // Q)[None, :] * Y1
            + (cols % Q)[None, :],
            row[None, :],
            keep[:, None] & (cols < H * Q)[None, :],
        )
        if HAS_SCALE:
            tl.store(
                YS + dests[:, None] * YS0 + scales[None, :] * YS1,
                srow[None, :],
                keep[:, None] & (scales < D)[None, :],
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
    if e and t and k and t * k * k <= 8192 and t * k <= 1024:
        _small[(max(t, e),)](
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            triton.reinterpret(recv_x, tl.uint8),
            recv_x_scale,
            recv_topk,
            triton.reinterpret(output_tensor, tl.uint8),
            output_tensor_scale,
            output_index,
            e,
            t,
            k,
            h,
            d,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
            recv_x.stride(0) * q,
            recv_x.stride(1) * q,
            q,
            recv_x_scale.stride(0) if has_scale else 0,
            recv_x_scale.stride(1) if has_scale else 0,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor.stride(0) * q,
            output_tensor.stride(1) * q,
            output_tensor_scale.stride(0) if has_scale else 0,
            output_tensor_scale.stride(1) if has_scale else 0,
            output_index.stride(0),
            output_index.stride(1),
            has_scale,
            triton.next_power_of_2(e),
            triton.next_power_of_2(t * k),
            triton.next_power_of_2(h * q),
            triton.next_power_of_2(max(d, 1)),
            triton.next_power_of_2(k),
            num_warps=4,
        )
        return (
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
        )
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
            1024,
        )
    if t and k:
        _dispatch[(min(t, 8192),)](
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
            triton.next_power_of_2(k),
            triton.next_power_of_2(h * q),
            triton.next_power_of_2(max(d, 1)),
            num_warps=4,
        )
    return (
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )


__all__ = ["ep_scatter"]
