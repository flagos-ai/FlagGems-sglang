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

# e4m3fn max; scales are computed in fp32 exactly like the reference.
_FP8_MAX = 448.0

_SMALL_MAX_ELEMS = 6815744  # 6.5M, between the 6.3M win and 8.4M loss points
_G_SMALL = 16
_SCAN = 16384
_TILE_S = 4096

_GP = 512
_BLOCK_L = 8192
_TILE_L = 4096
_GQ_CAP = 512


@triton.jit
def _codes_fp32(v, rcp):
    """e4m3fn code (0..255, sign in bit 7) of v*rcp, round-half-up."""
    b = (v * rcp).to(tl.int32, bitcast=True)
    mag = b & 0x7FFFFFFF
    code = ((mag + 0x7FFFF) >> 20) - 960
    code = tl.minimum(tl.maximum(code, 0), 126)
    return code | ((b >> 24) & 0x80)


@triton.jit
def _codes_u8(v, rcp):
    """e4m3fn code as a uint8 (sign in bit 7) of v*rcp, round-half-up."""
    return _codes_fp32(v, rcp).to(tl.uint8)


@triton.jit
def _quant_small_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    N,
    eps,
    SCAN: tl.constexpr,
    TILE: tl.constexpr,
):
    """Single-launch tier: G programs, each scans the full tensor for the
    global absmax (int16 bit-pattern max over 16384 lanes, software-
    pipelined), then quantizes tiles pid, pid+G, ... writing one uint8
    code per element (no pair-packing, no output views on the host)."""
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    acc = tl.zeros((SCAN,), dtype=tl.int16)
    for start in tl.range(0, N, SCAN, num_stages=2):
        offs = start + tl.arange(0, SCAN)
        v = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        m = v.to(tl.int16, bitcast=True) & 0x7FFF
        acc = tl.maximum(acc, m)
    amax = tl.max(acc, axis=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
    smax = tl.maximum(amax, eps)
    scale = smax / 448.0
    rcp = 448.0 / smax
    if pid == 0:
        tl.store(s_ptr, scale)
    for start in tl.range(pid * TILE, N, G * TILE, num_stages=2):
        offs = start + tl.arange(0, TILE)
        v = tl.load(x_ptr + offs, mask=offs < N, other=0.0).to(tl.float32)
        tl.store(q_ptr + offs, _codes_u8(v, rcp), mask=offs < N)


@triton.jit
def _absmax_partial_kernel(
    x_ptr,
    part_ptr,
    N,
    GP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GP programs over strided BLOCK-element tiles; one int16 partial
    absmax per program. Programs with no tiles store 0 (never the max)."""
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int16)
    for start in tl.range(pid * BLOCK, N, GP * BLOCK, num_stages=2):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        m = v.to(tl.int16, bitcast=True) & 0x7FFF
        acc = tl.maximum(acc, m)
    tl.store(part_ptr + pid, tl.max(acc, axis=0))


@triton.jit
def _quant_reduce_kernel(
    x_ptr,
    part_ptr,
    q_ptr,
    s_ptr,
    N,
    eps,
    GP: tl.constexpr,
    TILE: tl.constexpr,
):
    """Reduce the GP int16 partials, then quantize two strided tile
    streams per program (tiles pid and pid+GQ per step, so the compiler
    sees two independent load/store streams). Program 0 stores the
    scale."""
    parts = tl.load(part_ptr + tl.arange(0, GP))
    amax = tl.max(parts, axis=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
    smax = tl.maximum(amax, eps)
    scale = smax / 448.0
    rcp = 448.0 / smax
    pid = tl.program_id(0)
    GQ = tl.num_programs(0)
    if pid == 0:
        tl.store(s_ptr, scale)
    n16 = (N + 1) // 2
    STEP = GQ * TILE * 2
    for start in tl.range(pid * TILE, N, STEP, num_stages=2):
        offsA = start + tl.arange(0, TILE)
        offsB = start + GQ * TILE + tl.arange(0, TILE)
        vA = tl.load(x_ptr + offsA, mask=offsA < N, other=0.0).to(tl.float32)
        vB = tl.load(x_ptr + offsB, mask=offsB < N, other=0.0).to(tl.float32)
        bA = (vA * rcp).to(tl.int32, bitcast=True)
        bB = (vB * rcp).to(tl.int32, bitcast=True)
        magA = bA & 0x7FFFFFFF
        magB = bB & 0x7FFFFFFF
        codeA = ((magA + 0x7FFFF) >> 20) - 960
        codeB = ((magB + 0x7FFFF) >> 20) - 960
        codeA = tl.minimum(tl.maximum(codeA, 0), 126)
        codeB = tl.minimum(tl.maximum(codeB, 0), 126)
        cA = codeA | ((bA >> 24) & 0x80)
        cB = codeB | ((bB >> 24) & 0x80)
        pA = tl.reshape(cA, (TILE // 2, 2))
        pB = tl.reshape(cB, (TILE // 2, 2))
        loA, hiA = tl.split(pA)
        loB, hiB = tl.split(pB)
        wA = (loA | (hiA << 8)).to(tl.uint16)
        wB = (loB | (hiB << 8)).to(tl.uint16)
        uA = start // 2 + tl.arange(0, TILE // 2)
        uB = (start + GQ * TILE) // 2 + tl.arange(0, TILE // 2)
        tl.store(q_ptr + uA, wA, mask=uA < n16)
        tl.store(q_ptr + uB, wB, mask=uB < n16)


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    xc = x if x.is_contiguous() else x.contiguous()
    N = xc.numel()
    s = torch.empty(1, dtype=torch.float32, device=xc.device)

    if N == 0:
        s.fill_(eps / 448.0)
        return (
            torch.empty(xc.shape, dtype=torch.float8_e4m3fn, device=xc.device),
            s,
        )

    if N <= _SMALL_MAX_ELEMS:
        q = torch.empty(N, dtype=torch.uint8, device=xc.device)
        G = min(_G_SMALL, triton.cdiv(N, _TILE_S))
        _quant_small_kernel[(G,)](
            xc, q, s, N, eps, _SCAN, _TILE_S, num_warps=4
        )
        return q.view(xc.shape).view(torch.float8_e4m3fn), s

    n16 = (N + 1) // 2
    qb = torch.empty(n16 * 2, dtype=torch.uint8, device=xc.device)
    q16 = qb.view(torch.uint16)
    part = torch.empty(_GP, dtype=torch.int16, device=xc.device)
    _absmax_partial_kernel[(_GP,)](xc, part, N, _GP, _BLOCK_L, num_warps=4)
    GQ = min(triton.cdiv(N, _TILE_L), _GQ_CAP)
    _quant_reduce_kernel[(GQ,)](
        xc, part, q16, s, N, eps, _GP, _TILE_L, num_warps=8
    )

    return qb[:N].view(xc.shape).view(torch.float8_e4m3fn), s


__all__ = ["per_tensor_quant_mla_fp8"]
