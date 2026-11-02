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
# implied. See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
import torch  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _copy(
    D,
    S,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    SH: tl.constexpr,
    DS: tl.constexpr,
    SS: tl.constexpr,
    FROM: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    PERSIST: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * 32768
    offset = tl.load(OFF).to(tl.int64)
    size = tl.load(SZ).to(tl.int64)
    for tile in range(pid, T, tl.num_programs(0) * tl.num_programs(1)):
        i = (tile * B + tl.arange(0, B)).to(tl.int64)
        row = i // W
        tail = i % W
        sr = row + offset if FROM else row
        dr = row if FROM else row + offset
        sa = sr * SS[0]
        da = dr * DS[0]
        for j in tl.static_range(len(SH) - 1, 0, -1):
            c = tail % SH[j]
            tail = tail // SH[j]
            sa += c * SS[j]
            da += c * DS[j]
        valid = (i < N) & (row < size)
        value = tl.load(S + sa, valid, other=0)
        tl.store(D + da, value, valid)
        if not PERSIST:
            break


@triton.jit
def _linear(
    D,
    S,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    FROM: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    I32: tl.constexpr,
):
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    offset = tl.load(OFF).to(dt) * W
    size = tl.load(SZ).to(dt) * W
    for tile in range(tl.program_id(0), T, tl.num_programs(0)):
        i = (tile * B + tl.arange(0, B)).to(dt)
        source = i + offset if FROM else i
        dest = i if FROM else i + offset
        value = tl.load(S + source, (i < N) & (i < size), other=0)
        tl.store(D + dest, value, (i < N) & (i < size))


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    w = 1
    for size in dst.shape[1:]:
        w *= size
    n = min(dst.shape[0], src.shape[0]) * w
    if n:
        b = 2048
        t = triton.cdiv(n, b)
        if dst.is_contiguous() and src.is_contiguous():
            _linear[(min(t, 4096),)](
                dst,
                src,
                offset,
                sz,
                n,
                w,
                offset_src,
                b,
                t,
                max(dst.numel(), src.numel()) + 2048 < 2**31,
                num_warps=4,
                num_stages=1,
            )
            return dst
        _copy[(min(t, 4096),)](
            dst,
            src,
            offset,
            sz,
            n,
            w,
            tuple(dst.shape),
            tuple(dst.stride()),
            tuple(src.stride()),
            offset_src,
            b,
            t,
            True,
            num_warps=4,
            num_stages=1,
        )
    return dst


__all__ = ["memcpy_triton"]
