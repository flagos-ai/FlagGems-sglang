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
def _round(v, DT: tl.constexpr):
    if DT == tl.bfloat16:
        bits = v.to(tl.uint32, bitcast=True)
        rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
            tl.float32, bitcast=True
        )
        return tl.where(bits & 2139095040 == 2139095040, v, rounded)
    else:
        return v.to(DT).to(tl.float32)


@triton.jit
def _kernel(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    L: tl.constexpr,
    D: tl.constexpr,
    RS: tl.constexpr,
    US: tl.constexpr,
    GS: tl.constexpr,
    WS: tl.constexpr,
    SS: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    batch = tl.program_id(0) // PROGRAMS
    start = tl.program_id(0) % PROGRAMS
    c = tl.arange(0, BLOCK)
    w = tl.load(W + c * WS, c < D, other=0).to(tl.float32)
    gate = tl.load(G + batch * GS[0] + c * GS[2], c < D, other=0).to(
        tl.float32
    )
    scale = tl.load(S + batch * SS[0] + c * SS[2], c < D, other=0).to(
        tl.float32
    )
    factor = _round(1.0 + scale, S.dtype.element_ty)
    shift = tl.load(H + batch * HS[0] + c * HS[2], c < D, other=0).to(
        tl.float32
    )
    for seq in range(start, L, PROGRAMS):
        r = tl.load(
            R + batch * RS[0] + seq * RS[1] + c * RS[2], c < D, other=0
        ).to(tl.float32)
        u = tl.load(
            U + batch * US[0] + seq * US[1] + c * US[2], c < D, other=0
        ).to(tl.float32)
        update = _round(gate * u, O.dtype.element_ty)
        res = _round(r + update, O.dtype.element_ty)
        row = batch * L + seq
        tl.store(O + row * D + c, res, c < D)
        variance = tl.sum(res * res, 0) / D + EPS
        inv = tl.rsqrt(variance)
        refined = tl.fma(0.5 * inv, tl.fma(-variance, inv * inv, 1.0), inv)
        inv = tl.where(variance == float("inf"), inv, refined)
        norm = _round(res * inv * w, Y.dtype.element_ty)
        product = _round(norm * factor, Y.dtype.element_ty)
        tl.store(Y + row * D + c, product + shift, c < D)


def scale_residual_rmsnorm_scale_shift(
    residual, update, gate, weight, scale, shift, eps
):
    b, s, d = residual.shape
    y = torch.empty(
        residual.shape, dtype=residual.dtype, device=residual.device
    )
    out = torch.empty(
        residual.shape, dtype=residual.dtype, device=residual.device
    )
    if residual.numel() == 0:
        return (y, out)
    programs = min(s, 12)
    _kernel[b * programs,](
        residual,
        update,
        gate,
        weight,
        scale,
        shift,
        y,
        out,
        s,
        d,
        tuple(residual.stride()),
        tuple(update.stride()),
        tuple(gate.stride()),
        weight.stride(0),
        tuple(scale.stride()),
        tuple(shift.stride()),
        float(eps),
        1 << (d - 1).bit_length(),
        programs,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
        isCloseUnrollControl=True,
    )
    return (y, out)


__all__ = ["scale_residual_rmsnorm_scale_shift"]
