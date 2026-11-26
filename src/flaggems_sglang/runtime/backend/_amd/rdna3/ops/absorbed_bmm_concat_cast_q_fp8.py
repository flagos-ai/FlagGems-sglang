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

"""absorbed_bmm_concat_cast_q_fp8 for RDNA3 (gfx1100).

One program per (token block, head, column block) of the absorbed bmm, with
the heads of a token block next to each other: q_nope is (token, head, k),
and the generic op walks one head over all tokens at a time below
2^24 outputs, which reads q_nope in 256-byte pieces from cold memory. The
fp32 sum is rounded to bf16 and encoded to fp8 with the generic op's
encoder, four codes packed into each 32-bit store; the column-block-0
programs also cast their rows of q_rope. That encoder truncates to bf16
before rounding, so only bf16 inputs come here. Below 12288 (token, head)
rows, from 2^24 outputs up, and for strided, misaligned or overlapping
inputs, the generic op runs.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8 import _encode
from flaggems_sglang.ops.absorbed_bmm_concat_cast_q_fp8 import (
    absorbed_bmm_concat_cast_q_fp8 as _generic,
)


@triton.jit
def _pack4(code):
    # Four adjacent uint8 codes into one little-endian int32.
    shape: tl.constexpr = code.shape
    words = tl.reshape(code.to(tl.int32), (shape[0], shape[1] // 4, 2, 2))
    even, odd = tl.split(words)
    b0, b2 = tl.split(even)
    b1, b3 = tl.split(odd)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@triton.jit
def _absorbed_kernel(X, W, R, OUT, T, META: tl.constexpr):
    # The shape comes in as one constexpr tuple, which keeps the launch
    # cheap; OUT is the fp8 output viewed as int32 when PACK is set.
    O0: tl.constexpr = META[0]
    O1: tl.constexpr = META[1]
    H: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    D: tl.constexpr = META[5]
    BM: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    BK: tl.constexpr = META[8]
    BR: tl.constexpr = META[9]
    PACK: tl.constexpr = META[10]
    TN: tl.constexpr = N // BN
    pid = tl.program_id(0)
    ni = pid % TN
    h = pid // TN % H
    mi = pid // (TN * H)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    rows = m < T
    acc = tl.zeros((BM, BN), tl.float32)
    ones = tl.full((BM, BN), 1.0, tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + k
        x = tl.load(
            X + (m[:, None] * H + h) * K + kk[None, :], rows[:, None], 0
        )
        w = tl.load(W + (h * K + kk[:, None]) * N + n[None, :])
        # A dot accumulating into acc truncates the fp32 addition.
        acc = tl.fma(tl.dot(x, w), ones, acc)
    code = _encode(acc.to(tl.bfloat16).to(tl.float32))
    if PACK:
        nw = ni * (BN // 4) + tl.arange(0, BN // 4)
        tl.store(
            OUT + (m[:, None] * O0 + h * O1) // 4 + nw[None, :],
            _pack4(code),
            rows[:, None],
        )
    else:
        tl.store(
            OUT + m[:, None] * O0 + h * O1 + n[None, :], code, rows[:, None]
        )
    if ni == 0:
        r = tl.arange(0, BR)
        value = tl.load(
            R + (m[:, None] * H + h) * D + r[None, :],
            rows[:, None] & (r < D)[None, :],
            0,
        )
        rope = _encode(value.to(tl.float32))
        if PACK:
            rw = tl.arange(0, BR // 4)
            tl.store(
                OUT + (m[:, None] * O0 + h * O1 + N) // 4 + rw[None, :],
                _pack4(rope),
                rows[:, None] & (rw < D // 4)[None, :],
            )
        else:
            tl.store(
                OUT + m[:, None] * O0 + h * O1 + N + r[None, :],
                rope,
                rows[:, None] & (r < D)[None, :],
            )


def _config(t, h, n):
    """BM, BN, num_warps, pack; None keeps the generic op.

    Below 12288 (token, head) rows the kernel takes less time than the CPU
    side of the launch, where the generic op's launch is cheaper. From 2^24
    outputs up the generic op puts the heads of a token block next to each
    other itself and is as fast.
    """
    if t * h < 12288 or t * h * n >= 1 << 24:
        return None
    return 64, 128, 8, True


def _overlaps(a, b):
    a0 = a.data_ptr()
    b0 = b.data_ptr()
    return (
        a0 < b0 + b.numel() * b.element_size()
        and b0 < a0 + a.numel() * a.element_size()
    )


def absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    t, h, k = q_nope.shape
    n = w_kc.shape[-1]
    d = q_rope.shape[-1]
    out = q_fp8_pad
    cfg = _config(t, h, n)
    if (
        cfg is None
        or h != num_heads
        or h > out.shape[1]
        or t > out.shape[0]
        or out.shape[2] != n + d
        or out.stride(2) != 1
        or out.dtype != torch.float8_e4m3fn
        or q_nope.dtype != torch.bfloat16
        or w_kc.dtype != torch.bfloat16
        or q_rope.dtype != torch.bfloat16
        or not (
            q_nope.is_contiguous()
            and w_kc.is_contiguous()
            and q_rope.is_contiguous()
        )
        or k % 16
        or n % cfg[1]
        or d % 4
        or d > 256
        or (out.stride(0) | out.stride(1) | n | out.data_ptr()) % 4
        or (q_nope.data_ptr() | w_kc.data_ptr() | q_rope.data_ptr()) % 16
        or max(out.numel(), q_nope.numel(), w_kc.numel()) >= 1 << 31
        or _overlaps(out, q_nope)
        or _overlaps(out, w_kc)
        or _overlaps(out, q_rope)
    ):
        return _generic(q_fp8_pad, q_nope, w_kc, q_rope, num_heads)
    bm, bn, warps, pack = cfg
    meta = (
        out.stride(0),
        out.stride(1),
        h,
        k,
        n,
        d,
        bm,
        bn,
        64 if k % 64 == 0 else 32 if k % 32 == 0 else 16,
        max(16, triton.next_power_of_2(d)),
        pack,
    )
    _absorbed_kernel.run(
        q_nope,
        w_kc,
        q_rope,
        out.view(torch.int32) if pack else out.view(torch.uint8),
        t,
        meta,
        grid=(triton.cdiv(t, bm) * h * (n // bn),),
        warmup=False,
        num_warps=warps,
    )
    return out


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
