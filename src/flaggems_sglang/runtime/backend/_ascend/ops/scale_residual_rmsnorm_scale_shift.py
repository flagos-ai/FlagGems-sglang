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
def _kernel(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    RS: tl.constexpr,
    US: tl.constexpr,
    GS: tl.constexpr,
    WS: tl.constexpr,
    SS: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    ro = tl.arange(0, ROWS)
    tiles: tl.constexpr = triton.cdiv(L, ROWS)
    w = tl.load(W + c * WS, c < D, other=0).to(tl.float32)
    for tile in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        batch = tile // tiles
        seq = tile % tiles * ROWS + ro
        rows = batch * L + seq
        mask = (seq[:, None] < L) & (c[None, :] < D)
        r = tl.load(
            R + batch * RS[0] + seq[:, None] * RS[1] + c[None, :] * RS[2],
            mask,
            other=0,
        ).to(tl.float32)
        u = tl.load(
            U + batch * US[0] + seq[:, None] * US[1] + c[None, :] * US[2],
            mask,
            other=0,
        ).to(tl.float32)
        g = tl.load(G + batch * GS[0] + c * GS[2], c < D, other=0).to(
            tl.float32
        )
        p = (g[None, :] * u).to(O.dtype.element_ty).to(tl.float32)
        res = (r + p).to(O.dtype.element_ty).to(tl.float32)
        tl.store(O + rows[:, None] * D + c[None, :], res, mask)
        inv = tl.rsqrt(tl.sum(res * res, 1) / D + EPS)
        norm = (
            (res * inv[:, None] * w[None, :])
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        scale = tl.load(S + batch * SS[0] + c * SS[2], c < D, other=0).to(
            tl.float32
        )
        shift = tl.load(H + batch * HS[0] + c * HS[2], c < D, other=0).to(
            tl.float32
        )
        factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
        product = (
            (norm * factor[None, :]).to(Y.dtype.element_ty).to(tl.float32)
        )
        tl.store(
            Y + rows[:, None] * D + c[None, :], product + shift[None, :], mask
        )


@triton.jit
def _fast(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    BLOCK: tl.constexpr = triton.next_power_of_2(D)
    c = tl.arange(0, BLOCK)
    ro = tl.arange(0, ROWS)
    tiles: tl.constexpr = triton.cdiv(L, ROWS)
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    for tile in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        batch = tile // tiles
        seq = tile % tiles * ROWS + ro
        rows = batch * L + seq
        mask = (seq[:, None] < L) & (c[None, :] < D)
        r = tl.load(R + rows[:, None] * D + c[None, :], mask, other=0).to(
            tl.float32
        )
        u = tl.load(U + rows[:, None] * D + c[None, :], mask, other=0).to(
            tl.float32
        )
        g = tl.load(G + batch * D + c, c < D, other=0).to(tl.float32)
        p = (g[None, :] * u).to(O.dtype.element_ty).to(tl.float32)
        res = (r + p).to(O.dtype.element_ty).to(tl.float32)
        tl.store(O + rows[:, None] * D + c[None, :], res, mask)
        inv = tl.rsqrt(tl.sum(res * res, 1) / D + EPS)
        norm = (
            (res * inv[:, None] * w[None, :])
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        scale = tl.load(S + batch * D + c, c < D, other=0).to(tl.float32)
        shift = tl.load(H + batch * D + c, c < D, other=0).to(tl.float32)
        factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
        product = (
            (norm * factor[None, :]).to(Y.dtype.element_ty).to(tl.float32)
        )
        tl.store(
            Y + rows[:, None] * D + c[None, :], product + shift[None, :], mask
        )


def scale_residual_rmsnorm_scale_shift(
    residual, update, gate, weight, scale, shift, eps
):
    b, s, d = residual.shape
    y = torch.empty_like(residual, memory_format=torch.contiguous_format)
    out = torch.empty_like(y)
    if residual.numel() == 0:
        return (y, out)
    rows = 4 if d <= 2048 else 2
    if (
        residual.is_contiguous()
        and update.is_contiguous()
        and gate.is_contiguous()
        and weight.is_contiguous()
        and scale.is_contiguous()
        and shift.is_contiguous()
    ):
        _fast.run(
            residual,
            update,
            gate,
            weight,
            scale,
            shift,
            y,
            out,
            b,
            s,
            d,
            float(eps),
            rows,
            grid=(min(40, b * ((s + rows - 1) // rows)),),
            warmup=False,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    else:
        _kernel[min(40, b * ((s + rows - 1) // rows)),](
            residual,
            update,
            gate,
            weight,
            scale,
            shift,
            y,
            out,
            b,
            s,
            d,
            tuple(residual.stride()),
            tuple(update.stride()),
            tuple(gate.stride()),
            weight.stride(0),
            tuple(scale.stride()),
            tuple(shift.stride()),
            float(eps),
            rows,
            1 << (d - 1).bit_length(),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return (y, out)


__all__ = ["scale_residual_rmsnorm_scale_shift"]
