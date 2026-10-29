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
        for slot in range(NK):
            e = tl.load(K + token * K0 + slot * K1)
            if e >= 0:
                start = tl.sum(tl.where(es < e, counts, 0), 0)
                rank = tl.sum(
                    ((ids == e) & (p < token * NK + slot)).to(tl.int32), 0
                )
                dest = (start + rank).to(tl.int64)
                tl.store(Y + dest * Y0 + (c // Q) * Y1 + c % Q, row, c < H * Q)
                if HAS_SCALE:
                    tl.store(YS + dest * YS0 + scales * YS1, srow, scales < D)
                tl.store(O + token * O0 + slot * O1, dest)


@triton.jit
def _local(
    K,
    R,
    P,
    T: tl.constexpr,
    NK: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    NP: tl.constexpr,
    BP: tl.constexpr,
    B: tl.constexpr,
    NB: tl.constexpr,
    E: tl.constexpr,
):
    for task in range(tl.program_id(0), NB * E, tl.num_programs(0)):
        block = task % NB
        e = task // NB
        p = block * B + tl.arange(0, B)
        safe = tl.minimum(p, T * NK - 1)
        ids = tl.load(K + (safe // NK) * K0 + (safe % NK) * K1)
        hit = ((ids == e) & (p < T * NK)).to(tl.int32)
        rank = tl.cumsum(hit, 0)
        tl.store(R + e.to(tl.int64) * NP + p, rank - 1)
        tl.store(P + e.to(tl.int64) * BP + block, tl.sum(hit, 0))


@triton.jit
def _prefix(
    C,
    V,
    S,
    M,
    P,
    E: tl.constexpr,
    NB: tl.constexpr,
    BP: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
):
    e = tl.program_id(0)
    offsets = tl.arange(0, BP)
    counts = tl.load(C + tl.minimum(offsets, E - 1) * CS)
    start = tl.sum(tl.where(offsets < e, counts, 0), 0)
    if NB > 0:
        partial = tl.load(P + e * BP + tl.minimum(offsets, NB - 1))
        partial = tl.where(offsets < NB, partial, 0)
    else:
        partial = tl.full((BP,), 0, tl.int32)
    prefix = tl.cumsum(partial, 0) - partial + start
    tl.store(P + e * BP + offsets, prefix)
    tl.store(S + e * SS, start + tl.sum(partial, 0))
    count = tl.load(C + e * CS)
    valid = tl.load(V + e * VS)
    rows = tl.arange(0, 1024)
    for base in range(0, count, 1024):
        r = base + rows
        tl.store(M + (start + r) * MS, tl.where(r < valid, e, -1), r < count)


@triton.jit
def _copy(
    X,
    XS,
    K,
    R,
    P,
    Y,
    YS,
    O,
    T: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NP: tl.constexpr,
    BP: tl.constexpr,
    BR: tl.constexpr,
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
    B: tl.constexpr,
    BS: tl.constexpr,
):
    cols = tl.arange(0, B)
    scales = tl.arange(0, BS)
    for ti in range(tl.program_id(0), T, tl.num_programs(0)):
        token = ti.to(tl.int64)
        row = tl.load(
            X + token * X0 + (cols // Q) * X1 + cols % Q, cols < H * Q, 0
        )
        if HAS_SCALE:
            srow = tl.load(XS + token * XS0 + scales * XS1, scales < D, 0)
        for slot in range(NK):
            e = tl.load(K + token * K0 + slot * K1)
            if e >= 0:
                dest = (
                    tl.load(R + e.to(tl.int64) * NP + token * NK + slot)
                    + tl.load(P + e * BP + (token * NK + slot) // BR)
                ).to(tl.int64)
                tl.store(
                    Y + dest * Y0 + (cols // Q) * Y1 + cols % Q,
                    row,
                    cols < H * Q,
                )
                if HAS_SCALE:
                    tl.store(YS + dest * YS0 + scales * YS1, srow, scales < D)
                tl.store(O + token * O0 + slot * O1, dest)


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
    scaled = recv_x_scale is not None
    d = recv_x_scale.shape[1] if scaled else 0
    if e and t and k and t * k <= 1024:
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
            recv_x_scale.stride(0) if scaled else 0,
            recv_x_scale.stride(1) if scaled else 0,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor.stride(0) * q,
            output_tensor.stride(1) * q,
            output_tensor_scale.stride(0) if scaled else 0,
            output_tensor_scale.stride(1) if scaled else 0,
            output_index.stride(0),
            output_index.stride(1),
            scaled,
            triton.next_power_of_2(e),
            triton.next_power_of_2(t * k),
            triton.next_power_of_2(h * q),
            triton.next_power_of_2(max(d, 1)),
        )
        return (
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
        )
    nb = triton.cdiv(t * k, 256)
    np = nb * 256
    bp = max(128, triton.next_power_of_2(max(e, nb, 1)))
    if e:
        ranks = torch.empty(
            (e, np), dtype=torch.int32, device=recv_topk.device
        )
        partial = torch.empty(
            (e, bp), dtype=torch.int32, device=recv_topk.device
        )
        if t and k:
            _local[(min(nb * e, 65535),)](
                recv_topk,
                ranks,
                partial,
                t,
                k,
                recv_topk.stride(0),
                recv_topk.stride(1),
                np,
                bp,
                256,
                nb,
                e,
            )
        _prefix[(e,)](
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            partial,
            e,
            nb,
            bp,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
        )
        if t and k:
            _copy[(min(t, 40),)](
                triton.reinterpret(recv_x, tl.uint8),
                recv_x_scale,
                recv_topk,
                ranks,
                partial,
                triton.reinterpret(output_tensor, tl.uint8),
                output_tensor_scale,
                output_index,
                t,
                k,
                h,
                d,
                np,
                bp,
                256,
                recv_x.stride(0) * q,
                recv_x.stride(1) * q,
                q,
                recv_x_scale.stride(0) if scaled else 0,
                recv_x_scale.stride(1) if scaled else 0,
                recv_topk.stride(0),
                recv_topk.stride(1),
                output_tensor.stride(0) * q,
                output_tensor.stride(1) * q,
                output_tensor_scale.stride(0) if scaled else 0,
                output_tensor_scale.stride(1) if scaled else 0,
                output_index.stride(0),
                output_index.stride(1),
                scaled,
                triton.next_power_of_2(h * q),
                triton.next_power_of_2(max(d, 1)),
            )
    return (
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )


__all__ = ["ep_scatter"]
