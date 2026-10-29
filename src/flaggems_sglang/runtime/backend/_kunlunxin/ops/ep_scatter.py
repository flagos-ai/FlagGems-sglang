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
    tl.store(S + e * SS, start)


@triton.jit
def _mapping(
    C,
    V,
    S,
    M,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    start = tl.load(S + e * SS)
    count = tl.load(C + e * CS)
    valid = tl.load(V + e * VS)
    offs = tl.arange(0, B)
    for base in range(0, count, B):
        r = base + offs
        tl.store(M + (start + r) * MS, tl.where(r < valid, e, -1))


@triton.jit
def _rank(
    K,
    S,
    R,
    T: tl.constexpr,
    NK: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    NP: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    n = T * NK
    start = tl.load(S + e * S0)
    carry = 0
    offsets = tl.arange(0, B)
    for base in range(0, n, B):
        p = base + offsets
        if n % B == 0:
            ids = tl.load(K + (p // NK) * K0 + (p % NK) * K1)
        else:
            safe = tl.minimum(p, n - 1)
            ids = tl.load(K + (safe // NK) * K0 + (safe % NK) * K1)
        hit = ((ids == e) & (p < n)).to(tl.int32)
        rank = tl.cumsum(hit, 0)
        tl.store(R + e * NP + p, start + carry + rank - 1)
        carry += tl.sum(hit, 0)
    tl.store(S + e * S0, start + carry)


@triton.jit
def _dispatch(
    X,
    XS,
    K,
    S,
    Y,
    YS,
    O,
    R,
    NP: tl.constexpr,
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
):
    cols = tl.arange(0, B)
    scales = tl.arange(0, BS)
    for token in range(tl.program_id(0), T, tl.num_programs(0)):
        for slot in range(NK):
            e = tl.load(K + token * K0 + slot * K1)
            if e >= 0:
                dest = tl.load(R + e * NP + token * NK + slot)
                for base in range(0, H * Q, B):
                    b = base + cols
                    value = tl.load(
                        X + token * X0 + (b // Q) * X1 + b % Q,
                        b < H * Q,
                        other=0,
                    )
                    tl.store(
                        Y + dest * Y0 + (b // Q) * Y1 + b % Q, value, b < H * Q
                    )
                if HAS_SCALE:
                    for base in range(0, D, BS):
                        d = base + scales
                        value = tl.load(
                            XS + token * XS0 + d * XS1, d < D, other=0
                        )
                        tl.store(YS + dest * YS0 + d * YS1, value, d < D)
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
    has_scale = recv_x_scale is not None
    d = recv_x_scale.shape[1] if has_scale else 0
    packed = (
        h * q % 4 == 0
        and recv_x.stride(1) == 1
        and output_tensor.stride(1) == 1
        and recv_x.stride(0) * q % 4 == 0
        and output_tensor.stride(0) * q % 4 == 0
        and recv_x.storage_offset() * q % 4 == 0
        and output_tensor.storage_offset() * q % 4 == 0
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
            128,
        )
        _mapping[(e,)](
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
            128,
        )
    if t and k:
        np = triton.cdiv(t * k, 256) * 256
        ranks = torch.empty(
            (e, np), dtype=torch.int32, device=recv_topk.device
        )
        _rank[(e,)](
            recv_topk,
            expert_start_loc,
            ranks,
            t,
            k,
            recv_topk.stride(0),
            recv_topk.stride(1),
            expert_start_loc.stride(0),
            np,
            256,
        )
        if packed:
            _dispatch[(min(t, 1024),)](
                triton.reinterpret(recv_x, tl.uint32),
                recv_x_scale,
                recv_topk,
                expert_start_loc,
                triton.reinterpret(output_tensor, tl.uint32),
                output_tensor_scale,
                output_index,
                ranks,
                np,
                t,
                k,
                h * q // 4,
                d,
                recv_x.stride(0) * q // 4,
                1,
                1,
                recv_x_scale.stride(0) if has_scale else 0,
                recv_x_scale.stride(1) if has_scale else 0,
                recv_topk.stride(0),
                recv_topk.stride(1),
                expert_start_loc.stride(0),
                output_tensor.stride(0) * q // 4,
                1,
                output_tensor_scale.stride(0) if has_scale else 0,
                output_tensor_scale.stride(1) if has_scale else 0,
                output_index.stride(0),
                output_index.stride(1),
                has_scale,
                1024,
                128,
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
                ranks,
                np,
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
                1024,
                128,
            )
    return (
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )


__all__ = ["ep_scatter"]
