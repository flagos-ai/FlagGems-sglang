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
def _sigmoid_gate_mul_row_broadcast_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    D,
    stride_x,
    stride_o,
    TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // TILES
    tile = pid % TILES

    offs = tile * BLOCK_D + tl.arange(0, BLOCK_D)

    g = tl.load(gate_ptr + row).to(tl.float32)
    sig = tl.sigmoid(g)

    x_offs = row * stride_x + offs
    o_offs = row * stride_o + offs
    if EVEN:
        x = tl.load(x_ptr + x_offs, cache_modifier=".cg")
        y = x.to(tl.float32) * sig
        tl.store(
            out_ptr + o_offs,
            y.to(out_ptr.dtype.element_ty),
            cache_modifier=".cg",
        )
    else:
        mask = offs < D
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0, cache_modifier=".cg")
        y = x.to(tl.float32) * sig
        tl.store(
            out_ptr + o_offs,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=".cg",
        )


def sigmoid_gate_mul_broadcast(x, gate):
    N, D = x.shape
    out = torch.empty_like(x)
    if N == 0 or D == 0:
        return out

    if x.stride(-1) != 1:
        x = x.contiguous()
    if gate.dim() > 1 and N > 1 and gate.stride(0) != 1:
        gate = gate.contiguous()

    p2 = triton.next_power_of_2(D)
    if N <= 2:
        block_d = min(4096, max(256, p2 // 2))
        num_warps = max(2, min(32, block_d // 128))
    elif N <= 8:
        block_d = 4096 if p2 > 4096 else p2
        num_warps = min(32, max(1, block_d // 128))
    else:
        block_d = 1024 if p2 > 1024 else p2
        num_warps = max(1, min(8, block_d // 128))
        if block_d == 1024 and N <= 512:
            block_d = 512
            num_warps = 4
    tiles = (D + block_d - 1) // block_d

    _sigmoid_gate_mul_row_broadcast_kernel[(N * tiles,)](
        x,
        gate,
        out,
        D,
        x.stride(0),
        out.stride(0),
        TILES=tiles,
        BLOCK_D=block_d,
        EVEN=(D % block_d) == 0,
        num_warps=num_warps,
    )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
