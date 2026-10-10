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
def _sigmoid_gate_mul_broadcast_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    N,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pn = tl.program_id(0)
    pd = tl.program_id(1)

    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < N
    offs = rows[:, None] * D + cols[None, :]
    mask = row_mask[:, None] & (cols[None, :] < D)

    g = tl.load(gate_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_config(N, D):
    if N <= 8:
        return 1, 512, 4
    if N <= 32:
        return 2, 512, 4
    if N <= 1024:
        if D <= 1024:
            return 2, 1024, 4
        if D >= 8192:
            return 8, 1024, 8
        return 8, 512, 4
    if D <= 1024:
        return 8, 1024, 8
    return 2, 1024, 4


def sigmoid_gate_mul_broadcast(x, gate):
    N, D = x.shape
    out = torch.empty_like(x)
    if N == 0 or D == 0:
        return out

    if x.stride(1) != 1:
        x = x.contiguous()
    gate_in = gate if gate.is_contiguous() else gate.reshape(-1).contiguous()

    block_n, block_d, num_warps = _pick_config(N, D)
    grid = (triton.cdiv(N, block_n), triton.cdiv(D, block_d))
    _sigmoid_gate_mul_broadcast_kernel[grid](
        x,
        gate_in,
        out,
        N,
        D,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
