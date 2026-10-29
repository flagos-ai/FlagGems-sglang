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
def _late_weight(
    X,
    W,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    offsets = rows[:, None] * D + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + offsets, valid, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    normalized = x * rstd[:, None]
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    tl.store(Y + offsets, normalized * w[None, :], valid)


@triton.jit(do_not_specialize=["X", "W", "Y"])
def _compact_loop1(X, W, Y, M: tl.constexpr, D: tl.constexpr, E: tl.constexpr):
    B: tl.constexpr = triton.next_power_of_2(D)
    CAP: tl.constexpr = 1 if B >= 16384 else 16384 // B if B >= 1024 else 16
    SMALL: tl.constexpr = 4 if B <= 2048 else 2
    R: tl.constexpr = CAP if M >= 512 or CAP < SMALL else SMALL
    cols = tl.arange(0, B)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    for block in range(tl.program_id(0), tl.cdiv(M, R), tl.num_programs(0)):
        rows = block * R + tl.arange(0, R)
        offsets = rows[:, None] * D + cols[None, :]
        valid = (rows[:, None] < M) & (cols[None, :] < D)
        x = tl.load(X + offsets, valid, 0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1) / D + E)[:, None] * w[None, :]
        tl.store(Y + offsets, y, valid)


@triton.jit(do_not_specialize=["X", "W", "Y"])
def _small_no_m(X, W, Y, D: tl.constexpr, E: tl.constexpr):
    B: tl.constexpr = triton.next_power_of_2(D)
    rows = tl.program_id(0) + tl.arange(0, 1)
    cols = tl.arange(0, B)
    offsets = rows[:, None] * D + cols[None, :]
    valid = tl.broadcast_to(cols[None, :] < D, (1, B))
    x = tl.load(X + offsets, valid, 0).to(tl.float32)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    result = x * tl.rsqrt(tl.sum(x * x, 1) / D + E)[:, None] * w[None, :]
    tl.store(Y + offsets, result, valid)


def one_pass_rms_norm(x, w, eps):
    values = x.contiguous()
    weights = w.contiguous()
    out = torch.empty_like(values)
    d = values.shape[-1]
    m = values.numel() // d
    if m < 32 and d <= 4096:
        kernel = _small_no_m
        kernel.run(
            values,
            weights,
            out,
            d,
            eps,
            grid=(m,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return out
    b = triton.next_power_of_2(d)
    if b > 4096:
        _late_weight.run(
            values,
            weights,
            out,
            m,
            d,
            eps,
            1,
            b,
            grid=(m,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    else:
        cap = min(16, max(1, 16384 // b))
        r = min(4 if b <= 2048 else 2, cap) if m < 512 else cap
        _compact_loop1.run(
            values,
            weights,
            out,
            m,
            d,
            eps,
            grid=(min(40, triton.cdiv(m, r)),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["one_pass_rms_norm"]
