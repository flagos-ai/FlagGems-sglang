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

import torch
import triton
import triton.language as tl


@triton.jit
def _set_token_filter_wordwise_kernel(
    mask_ptr,
    token_ids_ptr,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        for i in range(num_tokens):
            token = tl.load(token_ids_ptr + i)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        tl.store(mask_ptr + off, acc, w < row_words)


@triton.jit
def _set_token_filter_wordargs_0_kernel(
    mask_ptr,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        tl.store(mask_ptr + off, acc, w < row_words)


@triton.jit(do_not_specialize=["TOKEN0", "TOKEN1", "TOKEN2", "TOKEN3"])
def _set_token_filter_wordargs_4_kernel(
    mask_ptr,
    TOKEN0,
    TOKEN1,
    TOKEN2,
    TOKEN3,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        if num_tokens >= 1:
            token = TOKEN0.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 2:
            token = TOKEN1.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 3:
            token = TOKEN2.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 4:
            token = TOKEN3.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        tl.store(mask_ptr + off, acc, w < row_words)


@triton.jit(
    do_not_specialize=[
        "TOKEN0",
        "TOKEN1",
        "TOKEN2",
        "TOKEN3",
        "TOKEN4",
        "TOKEN5",
        "TOKEN6",
        "TOKEN7",
    ]
)
def _set_token_filter_wordargs_8_kernel(
    mask_ptr,
    TOKEN0,
    TOKEN1,
    TOKEN2,
    TOKEN3,
    TOKEN4,
    TOKEN5,
    TOKEN6,
    TOKEN7,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        if num_tokens >= 1:
            token = TOKEN0.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 2:
            token = TOKEN1.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 3:
            token = TOKEN2.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 4:
            token = TOKEN3.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 5:
            token = TOKEN4.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 6:
            token = TOKEN5.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 7:
            token = TOKEN6.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 8:
            token = TOKEN7.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        tl.store(mask_ptr + off, acc, w < row_words)


@triton.jit(
    do_not_specialize=[
        "TOKEN0",
        "TOKEN1",
        "TOKEN2",
        "TOKEN3",
        "TOKEN4",
        "TOKEN5",
        "TOKEN6",
        "TOKEN7",
        "TOKEN8",
        "TOKEN9",
        "TOKEN10",
        "TOKEN11",
        "TOKEN12",
        "TOKEN13",
        "TOKEN14",
        "TOKEN15",
    ]
)
def _set_token_filter_wordargs_16_kernel(
    mask_ptr,
    TOKEN0,
    TOKEN1,
    TOKEN2,
    TOKEN3,
    TOKEN4,
    TOKEN5,
    TOKEN6,
    TOKEN7,
    TOKEN8,
    TOKEN9,
    TOKEN10,
    TOKEN11,
    TOKEN12,
    TOKEN13,
    TOKEN14,
    TOKEN15,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        if num_tokens >= 1:
            token = TOKEN0.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 2:
            token = TOKEN1.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 3:
            token = TOKEN2.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 4:
            token = TOKEN3.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 5:
            token = TOKEN4.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 6:
            token = TOKEN5.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 7:
            token = TOKEN6.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 8:
            token = TOKEN7.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 9:
            token = TOKEN8.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 10:
            token = TOKEN9.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 11:
            token = TOKEN10.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 12:
            token = TOKEN11.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 13:
            token = TOKEN12.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 14:
            token = TOKEN13.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 15:
            token = TOKEN14.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 16:
            token = TOKEN15.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        tl.store(mask_ptr + off, acc, w < row_words)


@triton.jit(
    do_not_specialize=[
        "TOKEN0",
        "TOKEN1",
        "TOKEN2",
        "TOKEN3",
        "TOKEN4",
        "TOKEN5",
        "TOKEN6",
        "TOKEN7",
        "TOKEN8",
        "TOKEN9",
        "TOKEN10",
        "TOKEN11",
        "TOKEN12",
        "TOKEN13",
        "TOKEN14",
        "TOKEN15",
        "TOKEN16",
        "TOKEN17",
        "TOKEN18",
        "TOKEN19",
        "TOKEN20",
        "TOKEN21",
        "TOKEN22",
        "TOKEN23",
        "TOKEN24",
        "TOKEN25",
        "TOKEN26",
        "TOKEN27",
        "TOKEN28",
        "TOKEN29",
        "TOKEN30",
        "TOKEN31",
    ]
)
def _set_token_filter_wordargs_32_kernel(
    mask_ptr,
    TOKEN0,
    TOKEN1,
    TOKEN2,
    TOKEN3,
    TOKEN4,
    TOKEN5,
    TOKEN6,
    TOKEN7,
    TOKEN8,
    TOKEN9,
    TOKEN10,
    TOKEN11,
    TOKEN12,
    TOKEN13,
    TOKEN14,
    TOKEN15,
    TOKEN16,
    TOKEN17,
    TOKEN18,
    TOKEN19,
    TOKEN20,
    TOKEN21,
    TOKEN22,
    TOKEN23,
    TOKEN24,
    TOKEN25,
    TOKEN26,
    TOKEN27,
    TOKEN28,
    TOKEN29,
    TOKEN30,
    TOKEN31,
    row_words: tl.constexpr,
    num_tokens: tl.constexpr,
    batch_row: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    RESET_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for block in range(
        tl.program_id(0), triton.cdiv(row_words, BLOCK), tl.num_programs(0)
    ):
        w = block * BLOCK + lane
        off = batch_row * STRIDE0 + w.to(tl.int64) * STRIDE1
        if RESET_MASK:
            acc = tl.full((BLOCK,), 0 if IS_ALLOWED else -1, tl.int32)
        else:
            acc = tl.load(mask_ptr + off, w < row_words, other=0)
        if num_tokens >= 1:
            token = TOKEN0.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 2:
            token = TOKEN1.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 3:
            token = TOKEN2.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 4:
            token = TOKEN3.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 5:
            token = TOKEN4.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 6:
            token = TOKEN5.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 7:
            token = TOKEN6.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 8:
            token = TOKEN7.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 9:
            token = TOKEN8.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 10:
            token = TOKEN9.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 11:
            token = TOKEN10.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 12:
            token = TOKEN11.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 13:
            token = TOKEN12.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 14:
            token = TOKEN13.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 15:
            token = TOKEN14.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 16:
            token = TOKEN15.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 17:
            token = TOKEN16.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 18:
            token = TOKEN17.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 19:
            token = TOKEN18.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 20:
            token = TOKEN19.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 21:
            token = TOKEN20.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 22:
            token = TOKEN21.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 23:
            token = TOKEN22.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 24:
            token = TOKEN23.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 25:
            token = TOKEN24.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 26:
            token = TOKEN25.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 27:
            token = TOKEN26.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 28:
            token = TOKEN27.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 29:
            token = TOKEN28.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 30:
            token = TOKEN29.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 31:
            token = TOKEN30.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        if num_tokens >= 32:
            token = TOKEN31.to(tl.int64)
            tw = token >> 5
            tw = tl.where(tw < 0, tw + row_words, tw)
            bit = (1 << (token.to(tl.int32) & 31)).to(tl.int32)
            update = tl.where(w == tw, bit, 0)
            if IS_ALLOWED:
                acc = acc | update
            else:
                acc = acc & ~update
        tl.store(mask_ptr + off, acc, w < row_words)


def set_token_filter(
    vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
):
    words = vocab_mask.shape[1]
    n = len(token_ids)
    row = batch_idx % vocab_mask.shape[0]
    aligned = (
        vocab_mask.data_ptr() + row * vocab_mask.stride(0) * 4
    ) % 128 == 0 and vocab_mask.stride(1) == 1
    if n <= 0:
        grid = triton.cdiv(words, 256) if aligned else 1
        _set_token_filter_wordargs_0_kernel[(grid,)](
            vocab_mask,
            words,
            n,
            row,
            vocab_mask.stride(0),
            vocab_mask.stride(1),
            is_allowed,
            reset_vocab_mask,
            256,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        return vocab_mask
    if n <= 4:
        values = tuple(token_ids) + (0,) * (4 - n)
        grid = triton.cdiv(words, 256) if aligned else 1
        _set_token_filter_wordargs_4_kernel[(grid,)](
            vocab_mask,
            values[0],
            values[1],
            values[2],
            values[3],
            words,
            n,
            row,
            vocab_mask.stride(0),
            vocab_mask.stride(1),
            is_allowed,
            reset_vocab_mask,
            256,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        return vocab_mask
    if n <= 8:
        values = tuple(token_ids) + (0,) * (8 - n)
        grid = triton.cdiv(words, 256) if aligned else 1
        _set_token_filter_wordargs_8_kernel[(grid,)](
            vocab_mask,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            words,
            n,
            row,
            vocab_mask.stride(0),
            vocab_mask.stride(1),
            is_allowed,
            reset_vocab_mask,
            256,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        return vocab_mask
    if n <= 16:
        values = tuple(token_ids) + (0,) * (16 - n)
        grid = triton.cdiv(words, 256) if aligned else 1
        _set_token_filter_wordargs_16_kernel[(grid,)](
            vocab_mask,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
            values[13],
            values[14],
            values[15],
            words,
            n,
            row,
            vocab_mask.stride(0),
            vocab_mask.stride(1),
            is_allowed,
            reset_vocab_mask,
            256,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        return vocab_mask
    if n <= 32:
        values = tuple(token_ids) + (0,) * (32 - n)
        grid = triton.cdiv(words, 256) if aligned else 1
        _set_token_filter_wordargs_32_kernel[(grid,)](
            vocab_mask,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
            values[13],
            values[14],
            values[15],
            values[16],
            values[17],
            values[18],
            values[19],
            values[20],
            values[21],
            values[22],
            values[23],
            values[24],
            values[25],
            values[26],
            values[27],
            values[28],
            values[29],
            values[30],
            values[31],
            words,
            n,
            row,
            vocab_mask.stride(0),
            vocab_mask.stride(1),
            is_allowed,
            reset_vocab_mask,
            256,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        return vocab_mask
    ids = vocab_mask.new_tensor(token_ids, dtype=torch.int64)
    grid = triton.cdiv(words, 256) if aligned else 1
    _set_token_filter_wordwise_kernel[(grid,)](
        vocab_mask,
        ids,
        words,
        len(token_ids),
        row,
        vocab_mask.stride(0),
        vocab_mask.stride(1),
        is_allowed,
        reset_vocab_mask,
        256,
        num_warps=4,
        num_stages=1,
        isCloseUnrollControl=True,
    )
    return vocab_mask


__all__ = ["set_token_filter"]
