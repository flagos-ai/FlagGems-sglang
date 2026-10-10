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
def _copy_req(X, C, Y, TH: tl.constexpr, HD: tl.constexpr, B: tl.constexpr):
    b = tl.program_id(0)
    c0 = tl.load(C + b)
    n = (tl.load(C + b + 1) - c0) * HD
    dst = c0 * HD
    for off in range(0, n, B):
        i = off + tl.arange(0, B)
        v = tl.load(X + b * TH + i, i < n, other=0)
        tl.store(Y + dst + i, v, i < n)


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


def unpad_draft_extend_output(
    raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q
):
    bs = seq_lens_q.shape[0]
    t, h, d = raw_out.shape[1:]
    out = raw_out.new_empty((sum_seq_lens_q, h, d))
    if bs and t and h and d and sum_seq_lens_q:
        if (
            raw_out.is_contiguous()
            and seq_lens_q.is_contiguous()
            and cu_seqlens_q.is_contiguous()
        ):
            _copy_req[(bs,)](
                raw_out,
                cu_seqlens_q,
                out,
                t * h * d,
                h * d,
                16384,
                num_warps=2,
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
                cu_seqlens_q.stride(0),
                seq_lens_q.stride(0),
                chunks,
                total,
                block,
                True,
                num_warps=2,
            )
    return out


__all__ = ["unpad_draft_extend_output"]
