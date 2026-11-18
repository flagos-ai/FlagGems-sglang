# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
def _gate_flat_kernel(X, G, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(X + i, valid, other=0).to(tl.float32)
    g = tl.load(G + i, valid, other=0).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(Y + i, y, valid)


@triton.jit
def _gate_flat_wide_kernel(X, G, Y, N, BLOCK: tl.constexpr, P: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), P):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        x = tl.load(X + i, valid, other=0).to(tl.float32)
        g = tl.load(G + i, valid, other=0).to(tl.float32)
        y = x * (1.0 / (1.0 + tl.exp(-g)))
        tl.store(Y + i, y, valid)


def sigmoid_gate_mul(x, gate):
    n = x.numel()
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if not n:
        return output
    source, gates = (x.contiguous(), gate.contiguous())
    # The enflame toolchain caps a 1-D load tile at 32 KiB per CTA and 8 KiB
    # per warp (memref_ext.slice_pad_start attribute error beyond that), so
    # blocks are scaled by itemsize. Within that cap the sweep ladder favors
    # more narrow nw=1 CTAs for tiny n and fatter nw=4 CTAs up to a single
    # 16-CTA wave; beyond one wave a 48-program grid-stride loop wins.
    itemsize = source.element_size()
    cap = 32768 // itemsize
    warp_cap = 8192 // itemsize
    if n <= 16 * cap:
        if n <= warp_cap:
            block, warps = triton.next_power_of_2(max(n, 64)), 1
        elif n <= cap:
            block, warps = warp_cap, 1
        elif n <= 8 * cap:
            block, warps = cap // 2, 4
        else:
            block, warps = cap, 4
        _gate_flat_kernel.run(
            source,
            gates,
            output,
            n,
            block,
            grid=(triton.cdiv(n, block),),
            warmup=False,
            num_warps=warps,
            num_stages=1,
        )
    else:
        _gate_flat_wide_kernel.run(
            source,
            gates,
            output,
            n,
            cap // 2,
            48,
            grid=(min(48, triton.cdiv(n, cap // 2)),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["sigmoid_gate_mul"]
