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
def _gather(src, lens, out, S: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    length = tl.load(lens + row).to(tl.int64)
    values = tl.load(
        src + row.to(tl.int64) * S + col, col < S, 0, cache_modifier=".cg"
    )
    value = tl.sum(tl.where(col == length - 1, values, 0), 0)
    tl.store(out + row, value)


@triton.jit
def _selected(
    src,
    lens,
    out,
    N: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
):
    r = tl.arange(0, B)
    c = tl.arange(0, K)
    row = tl.program_id(0) * B + r
    length = tl.load(lens + row, row < N, 1).to(tl.int64)
    values = tl.load(
        src + row[:, None].to(tl.int64) * S + c[None, :],
        (row[:, None] < N) & (c[None, :] < S),
        0,
        cache_modifier=C,
    )
    pointers = out + row[:, None] + tl.zeros((B, K), tl.int32)
    tl.store(
        pointers,
        values,
        (row[:, None] < N) & (c[None, :].to(tl.int64) == length[:, None] - 1),
    )


@triton.jit
def _vector_register(
    src,
    lens,
    out,
    N: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    safe = tl.minimum(row, N - 1)
    length = tl.load(lens + safe).to(tl.int64)
    pointer = src + safe.to(tl.int64) * S
    result = tl.full((B,), 0, tl.int64)
    for col in tl.static_range(0, S, 2):
        a, b = tl.inline_asm_elementwise(
            "ld.global" + C + ".v2.b64 {$0, $1}, [$2];",
            constraints="=l,=l,l",
            args=[pointer + col],
            dtype=(tl.int64, tl.int64),
            is_pure=False,
            pack=1,
        )
        result = tl.where(length == col + 1, a, result)
        result = tl.where(length == col + 2, b, result)
    tl.store(out + row, result, row < N)


def fill_bonus_tokens(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    n = accept_lens.shape[0]
    out = torch.empty(n, dtype=bonus_tokens.dtype, device=accept_tokens.device)
    if n:
        src = accept_tokens.contiguous()
        lens = accept_lens.contiguous()
        s = int(accept_stride)
        if (
            n >= 2048
            and s <= 16
            and s % 2 == 0
            and src.dtype == torch.int64
            and src.data_ptr() % 16 == 0
            and torch.version.hip is None
        ):
            _vector_register[(triton.cdiv(n, 32),)](
                src, lens, out, n, s, 32, ".cg", num_warps=1
            )
        elif n >= 2048 and s <= 16:
            _selected[(triton.cdiv(n, 4),)](
                src,
                lens,
                out,
                n,
                s,
                4,
                triton.next_power_of_2(s),
                ".cv",
                num_warps=1,
            )
        else:
            _gather[(n,)](
                src, lens, out, s, triton.next_power_of_2(s), num_warps=1
            )
    return out


__all__ = ["fill_bonus_tokens"]
