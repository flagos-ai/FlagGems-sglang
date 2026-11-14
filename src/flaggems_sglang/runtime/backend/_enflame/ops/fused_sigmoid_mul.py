# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task 97: FP32 sigmoid and multiplication fused with stride-aware reads."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_kernel(
    A,
    G,
    O,
    N: tl.constexpr,
    HIDDEN: tl.constexpr,
    DIM: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    token = i // HIDDEN
    hidden = i % HIDDEN
    av = tl.load(A + token * A0 + hidden * A1, valid, other=0).to(tl.float32)
    gv = tl.load(
        G + token * G0 + (hidden // DIM) * G1 + (hidden % DIM) * G2,
        valid,
        other=0,
    ).to(tl.float32)
    value = av * (1.0 / (1.0 + tl.exp(-gv)))
    tl.store(O + i, value, valid)


@triton.jit
def _gate_flat_kernel(A, G, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    av = tl.load(A + i, valid, other=0).to(tl.float32)
    gv = tl.load(G + i, valid, other=0).to(tl.float32)
    value = av * (1.0 / (1.0 + tl.exp(-gv)))
    tl.store(O + i, value, valid)


@triton.jit
def _gate_flat_wide_kernel(A, G, O, N, BLOCK: tl.constexpr, P: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), P):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        av = tl.load(A + i, valid, other=0).to(tl.float32)
        gv = tl.load(G + i, valid, other=0).to(tl.float32)
        value = av * (1.0 / (1.0 + tl.exp(-gv)))
        tl.store(O + i, value, valid)


def fused_sigmoid_mul(attn_output, gate):
    tokens, hidden = attn_output.shape
    out = torch.empty(
        (tokens, hidden), dtype=attn_output.dtype, device=attn_output.device
    )
    if out.numel() == 0:
        return out
    if gate.ndim == 3:
        dim = gate.shape[2]
        g0, g1, g2 = gate.stride()
    else:
        dim = hidden
        g0, g2 = gate.stride()
        g1 = 0
    a0, a1 = attn_output.stride()
    n = out.numel()
    # Flat fast path: gate[t, h // dim, h % dim] sits at flat offset
    # t * hidden + h whenever both tensors are contiguous, so both loads
    # become pure 1-D streams with no per-element div/mod addressing.
    # Block/warp ladder is scaled by itemsize: the enflame toolchain caps
    # a 1-D load tile at 32 KiB per CTA and 8 KiB per warp
    # (memref_ext.slice_pad_start attribute error beyond that).
    if (
        a1 == 1
        and a0 == hidden
        and g2 == 1
        and g0 == hidden
        and (gate.ndim != 3 or g1 == dim)
    ):
        itemsize = attn_output.element_size()
        cap = 32768 // itemsize
        warp_cap = 8192 // itemsize
        if n <= 16 * cap:
            # Single-CTA tiles only win for tiny n; from 2K elements on a
            # ~16-CTA grid hides load latency best: block =
            # next_pow2(n/16) floored at 2048 and capped at the compile
            # limit (interleaved do_bench on GCU400: +12% at n=16K/32K vs
            # one fat CTA or cap-wide pairs, +8-9% at 64K/128K, while 32+
            # CTAs regresses). n=16*cap lands exactly on cap@16 CTAs, so
            # the top of this band keeps the previous configuration.
            if n <= 2048:
                block = triton.next_power_of_2(max(n, 64))
            else:
                block = min(cap, max(2048, triton.next_power_of_2(n // 16)))
            warps = max(1, triton.cdiv(block, warp_cap))
            _gate_flat_kernel.run(
                attn_output,
                gate,
                out,
                n,
                block,
                grid=(triton.cdiv(n, block),),
                warmup=False,
                num_warps=warps,
                num_stages=1,
            )
        else:
            _gate_flat_wide_kernel.run(
                attn_output,
                gate,
                out,
                n,
                cap // 2,
                48,
                grid=(min(48, triton.cdiv(n, cap // 2)),),
                warmup=False,
                num_warps=4,
                num_stages=1,
            )
        return out
    block = 1024
    _gate_kernel[(triton.cdiv(n, block),)](
        attn_output,
        gate,
        out,
        n,
        hidden,
        dim,
        a0,
        a1,
        g0,
        g1,
        g2,
        block,
        num_warps=4,
    )
    return out


__all__ = ["fused_sigmoid_mul"]
