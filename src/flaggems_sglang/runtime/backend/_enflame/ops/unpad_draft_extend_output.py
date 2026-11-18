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

_MAX_PROGRAMS = 48


@triton.jit
def _copy_flat(
    X,
    C,
    L,
    Y,
    T: tl.constexpr,
    HD: tl.constexpr,
    CS,
    LS,
    BS,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, B)
    for b in range(pid, BS, nprog):
        s = tl.load(L + b * LS)
        start = tl.load(C + b * CS)
        n = s * HD
        v = tl.load(X + b * T * HD + i, i < n)
        tl.store(Y + start * HD + i, v, i < n)


@triton.jit
def _copy_chunks(
    X,
    C,
    L,
    Y,
    T: tl.constexpr,
    HD: tl.constexpr,
    CS,
    LS,
    BS,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, B)
    for b in range(pid, BS, nprog):
        s = tl.load(L + b * LS)
        start = tl.load(C + b * CS)
        n = s * HD
        for off in range(0, n, B):
            j = off + i
            v = tl.load(X + b * T * HD + j, j < n)
            tl.store(Y + start * HD + j, v, j < n)


@triton.jit
def _copy_flat_u32(
    X,
    C,
    L,
    Y,
    T: tl.constexpr,
    HD2: tl.constexpr,
    CS,
    LS,
    BS,
    B2: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, B2)
    for b in range(pid, BS, nprog):
        s = tl.load(L + b * LS)
        start = tl.load(C + b * CS)
        n = s * HD2
        v = tl.load(X + b * T * HD2 + i, i < n)
        tl.store(Y + start * HD2 + i, v, i < n)


@triton.jit
def _copy(
    X,
    C,
    L,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    CS: tl.constexpr,
    LS: tl.constexpr,
    CHUNKS: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    CLAMP: tl.constexpr,
):
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        chunk = pid % CHUNKS
        row = pid // CHUNKS
        b = row // T
        t = row % T
        length = tl.load(L + b * LS)
        if t < length:
            start = tl.load(C + b * CS)
            i = chunk * B + tl.arange(0, B)
            if CLAMP:
                j = tl.minimum(i, H * D - 1)
                v = tl.load(X + b * S0 + t * S1 + (j // D) * S2 + (j % D) * S3)
            else:
                v = tl.load(
                    X + b * S0 + t * S1 + (i // D) * S2 + (i % D) * S3,
                    i < H * D,
                    other=0,
                )
            tl.store(Y + (start + t) * H * D + i, v, i < H * D)


def _to_i32(tensor):
    if tensor.dtype != torch.int32:
        return tensor.to(torch.int32)
    return tensor


def unpad_draft_extend_output(
    raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q
):
    bs = seq_lens_q.shape[0]
    t, h, d = raw_out.shape[1:]
    out = raw_out.new_empty((sum_seq_lens_q, h, d))
    if bs and t and h and d and sum_seq_lens_q:
        cu_seqlens_q = _to_i32(cu_seqlens_q)
        seq_lens_q = _to_i32(seq_lens_q)
        cs = cu_seqlens_q.stride(0)
        ls = seq_lens_q.stride(0)
        if raw_out.is_contiguous():
            span = t * h * d
            grid = (min(bs, _MAX_PROGRAMS),)
            if (
                raw_out.element_size() == 2
                and d % 2 == 0
                and span <= 16384
                and raw_out.data_ptr() % 4 == 0
                and out.data_ptr() % 4 == 0
            ):
                _copy_flat_u32[grid](
                    raw_out.view(torch.int32),
                    cu_seqlens_q,
                    seq_lens_q,
                    out.view(torch.int32),
                    t,
                    h * d // 2,
                    cs,
                    ls,
                    bs,
                    triton.next_power_of_2(span // 2),
                    num_warps=4,
                )
            elif span <= 16384:
                _copy_flat[grid](
                    raw_out,
                    cu_seqlens_q,
                    seq_lens_q,
                    out,
                    t,
                    h * d,
                    cs,
                    ls,
                    bs,
                    triton.next_power_of_2(span),
                    num_warps=4,
                )
            else:
                _copy_chunks[grid](
                    raw_out,
                    cu_seqlens_q,
                    seq_lens_q,
                    out,
                    t,
                    h * d,
                    cs,
                    ls,
                    bs,
                    16384,
                    num_warps=4,
                )
        else:
            block = min(triton.next_power_of_2(h * d), 4096)
            chunks = triton.cdiv(h * d, block)
            total = bs * t * chunks
            _copy[(min(total, 65535),)](
                raw_out,
                cu_seqlens_q,
                seq_lens_q,
                out,
                t,
                h,
                d,
                *raw_out.stride(),
                cs,
                ls,
                chunks,
                total,
                block,
                True,
                num_warps=2,
            )
    return out


__all__ = ["unpad_draft_extend_output"]
