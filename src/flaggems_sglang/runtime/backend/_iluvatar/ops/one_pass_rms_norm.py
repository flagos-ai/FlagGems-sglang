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
def _tile(
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
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    y = x * rstd[:, None] * w[None, :]
    tl.store(Y + offsets, y, valid)


@triton.jit(do_not_specialize=["X", "W", "Y"])
def _lean(X, W, Y, M: tl.constexpr, D: tl.constexpr, E: tl.constexpr):
    Y = tl.multiple_of(Y, 16)
    B: tl.constexpr = triton.next_power_of_2(D)
    R: tl.constexpr = 1 if (D > 2048 or M < 1024) or M >= 8192 else 2
    if M >= 4096:
        _late_weight(X, W, Y, M, D, E, R, B)
    else:
        _tile(X, W, Y, M, D, E, R, B)


@triton.jit(do_not_specialize=["X", "W", "Y"])
def _lean_unaligned(
    X, W, Y, M: tl.constexpr, D: tl.constexpr, E: tl.constexpr
):
    B: tl.constexpr = triton.next_power_of_2(D)
    R: tl.constexpr = 1 if (D > 2048 or M < 1024) or M >= 8192 else 2
    if M >= 4096:
        _late_weight(X, W, Y, M, D, E, R, B)
    else:
        _tile(X, W, Y, M, D, E, R, B)


def one_pass_rms_norm(x, w, eps):
    values = x.contiguous()
    weights = w.contiguous()
    out = torch.empty_like(values)
    D = values.shape[-1]
    M = values.numel() // D
    R = 1 if D > 2048 or M < 1024 or M >= 8192 else 2
    warps = 16 if D > 2048 else 8 if M >= 4096 else 4
    kernel = _lean_unaligned if M < 4096 else _lean
    kernel.run(
        values,
        weights,
        out,
        M,
        D,
        eps,
        grid=(triton.cdiv(M, R),),
        warmup=False,
        num_warps=warps,
        num_stages=1,
    )
    return out


@triton.jit(do_not_specialize=["X", "W", "Y"])
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
    Y = tl.multiple_of(Y, 16)
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    offsets = rows[:, None] * D + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + offsets, valid, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    normalized = x * rstd[:, None]
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    tl.store(Y + offsets, normalized * w[None, :], valid)


__all__ = ["one_pass_rms_norm"]
