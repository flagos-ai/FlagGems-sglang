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
    M: tl.constexpr,
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
):
    c = tl.arange(0, BLOCK)
    w = tl.load(W + c * WS, c < D, other=0).to(tl.float32)
    if M >= 2048 and M <= 65535 and (BLOCK <= 2048):
        row = tl.program_id(0)
        batch = row // L
        seq = row % L
        r = tl.load(
            R + batch * RS[0] + seq * RS[1] + c * RS[2], c < D, other=0
        ).to(tl.float32)
        u = tl.load(
            U + batch * US[0] + seq * US[1] + c * US[2], c < D, other=0
        ).to(tl.float32)
        gate = tl.load(G + batch * GS[0] + c * GS[2], c < D, other=0).to(
            tl.float32
        )
        update = (gate * u).to(O.dtype.element_ty).to(tl.float32)
        res = (r + update).to(O.dtype.element_ty).to(tl.float32)
        tl.store(O + row * D + c, res, c < D)
        inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
        norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
        scale = tl.load(S + batch * SS[0] + c * SS[2], c < D, other=0).to(
            tl.float32
        )
        shift = tl.load(H + batch * HS[0] + c * HS[2], c < D, other=0).to(
            tl.float32
        )
        factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
        product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + c, product + shift, c < D)
    else:
        for row in range(tl.program_id(0), M, tl.num_programs(0)):
            batch = row // L
            seq = row % L
            r = tl.load(
                R + batch * RS[0] + seq * RS[1] + c * RS[2], c < D, other=0
            ).to(tl.float32)
            u = tl.load(
                U + batch * US[0] + seq * US[1] + c * US[2], c < D, other=0
            ).to(tl.float32)
            gate = tl.load(G + batch * GS[0] + c * GS[2], c < D, other=0).to(
                tl.float32
            )
            update = (gate * u).to(O.dtype.element_ty).to(tl.float32)
            res = (r + update).to(O.dtype.element_ty).to(tl.float32)
            tl.store(O + row * D + c, res, c < D)
            inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
            norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
            scale = tl.load(S + batch * SS[0] + c * SS[2], c < D, other=0).to(
                tl.float32
            )
            shift = tl.load(H + batch * HS[0] + c * HS[2], c < D, other=0).to(
                tl.float32
            )
            factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
            product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(Y + row * D + c, product + shift, c < D)


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
    M: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    w = tl.load(W + c * 1, c < D, other=0).to(tl.float32)
    if M >= 2048 and M <= 65535 and (BLOCK <= 2048):
        row = tl.program_id(0)
        batch = row // L
        seq = row % L
        r = tl.load(R + batch * (L * D) + seq * D + c * 1, c < D, other=0).to(
            tl.float32
        )
        u = tl.load(U + batch * (L * D) + seq * D + c * 1, c < D, other=0).to(
            tl.float32
        )
        gate = tl.load(G + batch * D + c * 1, c < D, other=0).to(tl.float32)
        update = (gate * u).to(O.dtype.element_ty).to(tl.float32)
        res = (r + update).to(O.dtype.element_ty).to(tl.float32)
        tl.store(O + row * D + c, res, c < D)
        inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
        norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
        scale = tl.load(S + batch * D + c * 1, c < D, other=0).to(tl.float32)
        shift = tl.load(H + batch * D + c * 1, c < D, other=0).to(tl.float32)
        factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
        product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + c, product + shift, c < D)
    else:
        for row in range(tl.program_id(0), M, tl.num_programs(0)):
            batch = row // L
            seq = row % L
            r = tl.load(
                R + batch * (L * D) + seq * D + c * 1, c < D, other=0
            ).to(tl.float32)
            u = tl.load(
                U + batch * (L * D) + seq * D + c * 1, c < D, other=0
            ).to(tl.float32)
            gate = tl.load(G + batch * D + c * 1, c < D, other=0).to(
                tl.float32
            )
            update = (gate * u).to(O.dtype.element_ty).to(tl.float32)
            res = (r + update).to(O.dtype.element_ty).to(tl.float32)
            tl.store(O + row * D + c, res, c < D)
            inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
            norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
            scale = tl.load(S + batch * D + c * 1, c < D, other=0).to(
                tl.float32
            )
            shift = tl.load(H + batch * D + c * 1, c < D, other=0).to(
                tl.float32
            )
            factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
            product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(Y + row * D + c, product + shift, c < D)


def scale_residual_rmsnorm_scale_shift(
    residual, update, gate, weight, scale, shift, eps
):
    b, s, d = residual.shape
    y = torch.empty_like(residual, memory_format=torch.contiguous_format)
    out = torch.empty_like(y)
    if residual.numel() == 0:
        return (y, out)
    block = 1 << (d - 1).bit_length()
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
            b * s,
            s,
            d,
            float(eps),
            block,
            num_warps=4 if block <= 2048 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(min(b * s, 65535),),
            warmup=False,
        )
    else:
        _kernel.run(
            residual,
            update,
            gate,
            weight,
            scale,
            shift,
            y,
            out,
            b * s,
            s,
            d,
            tuple(residual.stride()),
            tuple(update.stride()),
            tuple(gate.stride()),
            weight.stride(0),
            tuple(scale.stride()),
            tuple(shift.stride()),
            float(eps),
            block,
            num_warps=4 if block <= 2048 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(min(b * s, 65535),),
            warmup=False,
        )
    return (y, out)


__all__ = ["scale_residual_rmsnorm_scale_shift"]
