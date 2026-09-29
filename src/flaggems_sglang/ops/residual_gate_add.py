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

"""Keep low-precision typed operands and explicit product rounding; enable native packed arithmetic.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _residual_gate_add(
    R,
    U,
    G,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    BROADCAST: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    gidx = i % D if BROADCAST else i
    r = tl.load(R + i, valid, 0)
    u = tl.load(U + i, valid, 0)
    g = tl.load(G + gidx, valid, 0)
    product = (u * g).to(O.dtype.element_ty)
    result = r + product
    tl.store(O + i, result.to(O.dtype.element_ty), valid)


@triton.jit
def _broadcast(
    R,
    U,
    G,
    O,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    cols = tl.cdiv(D, B)
    row = (tile // cols) * 4 + tl.arange(0, 4)
    col = (tile % cols) * B + tl.arange(0, B)
    offsets = row[:, None] * D + col[None, :]
    valid = (row[:, None] < ROWS) & (col[None, :] < D)
    r = tl.load(R + offsets, valid, 0)
    u = tl.load(U + offsets, valid, 0)
    g = tl.load(G + col, col < D, 0)
    product = (u * g[None, :]).to(O.dtype.element_ty)
    result = r + product
    tl.store(O + offsets, result.to(O.dtype.element_ty), valid)


def residual_gate_add(residual, update, gate):
    output = torch.empty(
        residual.shape, dtype=residual.dtype, device=residual.device
    )
    n = residual.numel()
    if n and gate.numel() != n:
        d = residual.shape[-1]
        rows = n // d
        b = min(512, triton.next_power_of_2(d))
        _broadcast[(triton.cdiv(rows, 4) * triton.cdiv(d, b),)](
            residual,
            update,
            gate,
            output,
            rows,
            d,
            b,
            n >= 2147483648,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif n:
        _residual_gate_add[(triton.cdiv(n, 1024),)](
            residual,
            update,
            gate,
            output,
            n,
            residual.shape[-1],
            gate.numel() != n,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


__all__ = ["residual_gate_add"]
