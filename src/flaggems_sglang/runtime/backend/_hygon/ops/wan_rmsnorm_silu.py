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
from triton.language.extra import libdevice


@triton.jit
def _tile0(X, G, B, Out, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset, cache_modifier=".cg").to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    if BC == 128 and N >= 16384:
        g = tl.load(
            G + tl.broadcast_to(col[None, :] * GS, (BR, BC)),
            col[None, :] < C,
            0,
        ).to(tl.float32)
    else:
        g = tl.load(G + col * GS, col < C, 0).to(tl.float32)[None, :]
    if BC == 128 and N >= 4096:
        a, b = tl.split(tl.reshape(x, (BR, BC // 4, 2, 2)))
        a0, a1 = tl.split(a)
        b0, b1 = tl.split(b)
        total = tl.sum(
            tl.fma(a0, a0, tl.fma(a1, a1, tl.fma(b0, b0, b1 * b1))), 1
        )
    else:
        total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g
    elif EPS >= 1e-30:
        inv = tl.minimum(_hardware_rsq421(total), 1.0 / EPS)
        if (
            BC == 256
            and N >= 4096
            and (SCALE >= -32)
            and (SCALE <= 32)
            and (EPS >= 1e-18)
            and ((SCALE if SCALE >= 0 else -SCALE) / EPS < 1.7014117e38)
            and (
                (SCALE if SCALE >= 0 else -SCALE)
                / (EPS if EPS >= 1.8446744e19 else 1.8446744e19)
                > 2.3509887e-38
            )
        ):
            gain = inv * SCALE
            y = x * gain[:, None] * g
        else:
            y = x * inv[:, None] * SCALE * g
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = libdevice.fast_dividef(y, 1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(Out + offset, _output551(y, Out, P), cache_modifier=".cs")
    else:
        tl.store(
            Out + offset,
            _output551(y, Out, P),
            (row[:, None] < N) & (col[None, :] < C),
        )


@triton.jit
def _tile_inplace0(X, G, B, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset, cache_modifier=".cg").to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    if BC == 128 and N >= 16384:
        g = tl.load(
            G + tl.broadcast_to(col[None, :] * GS, (BR, BC)),
            col[None, :] < C,
            0,
        ).to(tl.float32)
    else:
        g = tl.load(G + col * GS, col < C, 0).to(tl.float32)[None, :]
    if BC == 128 and N >= 4096:
        a, b = tl.split(tl.reshape(x, (BR, BC // 4, 2, 2)))
        a0, a1 = tl.split(a)
        b0, b1 = tl.split(b)
        total = tl.sum(
            tl.fma(a0, a0, tl.fma(a1, a1, tl.fma(b0, b0, b1 * b1))), 1
        )
    else:
        total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g
    elif EPS >= 1e-30:
        inv = tl.minimum(_hardware_rsq421(total), 1.0 / EPS)
        if (
            BC == 256
            and N >= 4096
            and (SCALE >= -32)
            and (SCALE <= 32)
            and (EPS >= 1e-18)
            and ((SCALE if SCALE >= 0 else -SCALE) / EPS < 1.7014117e38)
            and (
                (SCALE if SCALE >= 0 else -SCALE)
                / (EPS if EPS >= 1.8446744e19 else 1.8446744e19)
                > 2.3509887e-38
            )
        ):
            gain = inv * SCALE
            y = x * gain[:, None] * g
        else:
            y = x * inv[:, None] * SCALE * g
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = libdevice.fast_dividef(y, 1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(X + offset, _output551(y, X, P), cache_modifier=".cs")
    else:
        tl.store(
            X + offset,
            _output551(y, X, P),
            (row[:, None] < N) & (col[None, :] < C),
        )


@triton.jit
def _parts(X, G, B, Out, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    INPLACE: tl.constexpr = P[9]
    if N >= 16384:
        row = tl.program_id(0) * BR + tl.arange(0, BR)
        col = tl.arange(0, BC // 8)
        if N * C + BR * BC < 2147483648:
            offset = row[:, None] * C + col[None, :]
        else:
            offset = row[:, None].to(tl.int64) * C + col[None, :]
        if N % BR == 0:
            x0 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 0 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x0 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 0 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x1 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 1 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x1 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 1 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x2 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 2 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x2 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 2 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x3 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 3 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x3 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 3 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x4 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 4 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x4 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 4 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x5 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 5 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x5 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 5 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x6 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 6 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x6 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 6 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        if N % BR == 0:
            x7 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 7 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                )
            ).to(tl.float32)
        else:
            x7 = tl.load(
                tl.make_block_ptr(
                    base=X,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 7 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                boundary_check=(0,),
                padding_option="zero",
            ).to(tl.float32)
        total = tl.sum(
            x0 * x0
            + x1 * x1
            + (x2 * x2 + x3 * x3)
            + (x4 * x4 + x5 * x5 + (x6 * x6 + x7 * x7)),
            1,
        )
        if EPS < 1e-18:
            norm = tl.maximum(libdevice.sqrt(total), EPS)
            v0 = tl.div_rn(x0, norm[:, None]) * SCALE
            v1 = tl.div_rn(x1, norm[:, None]) * SCALE
            v2 = tl.div_rn(x2, norm[:, None]) * SCALE
            v3 = tl.div_rn(x3, norm[:, None]) * SCALE
            v4 = tl.div_rn(x4, norm[:, None]) * SCALE
            v5 = tl.div_rn(x5, norm[:, None]) * SCALE
            v6 = tl.div_rn(x6, norm[:, None]) * SCALE
            v7 = tl.div_rn(x7, norm[:, None]) * SCALE
        else:
            inv = tl.minimum(_hardware_rsq421(total), 1.0 / EPS)
            v0 = x0 * inv[:, None] * SCALE
            v1 = x1 * inv[:, None] * SCALE
            v2 = x2 * inv[:, None] * SCALE
            v3 = x3 * inv[:, None] * SCALE
            v4 = x4 * inv[:, None] * SCALE
            v5 = x5 * inv[:, None] * SCALE
            v6 = x6 * inv[:, None] * SCALE
            v7 = x7 * inv[:, None] * SCALE
        g0 = tl.load(G + (col + 0 * (BC // 8)) * GS).to(tl.float32)
        y0 = v0 * g0[None, :]
        g1 = tl.load(G + (col + 1 * (BC // 8)) * GS).to(tl.float32)
        y1 = v1 * g1[None, :]
        g2 = tl.load(G + (col + 2 * (BC // 8)) * GS).to(tl.float32)
        y2 = v2 * g2[None, :]
        g3 = tl.load(G + (col + 3 * (BC // 8)) * GS).to(tl.float32)
        y3 = v3 * g3[None, :]
        g4 = tl.load(G + (col + 4 * (BC // 8)) * GS).to(tl.float32)
        y4 = v4 * g4[None, :]
        g5 = tl.load(G + (col + 5 * (BC // 8)) * GS).to(tl.float32)
        y5 = v5 * g5[None, :]
        g6 = tl.load(G + (col + 6 * (BC // 8)) * GS).to(tl.float32)
        y6 = v6 * g6[None, :]
        g7 = tl.load(G + (col + 7 * (BC // 8)) * GS).to(tl.float32)
        y7 = v7 * g7[None, :]
        if BIAS:
            b0 = tl.load(B + (col + 0 * (BC // 8)) * BS).to(tl.float32)
            y0 = y0 + b0[None, :]
            b1 = tl.load(B + (col + 1 * (BC // 8)) * BS).to(tl.float32)
            y1 = y1 + b1[None, :]
            b2 = tl.load(B + (col + 2 * (BC // 8)) * BS).to(tl.float32)
            y2 = y2 + b2[None, :]
            b3 = tl.load(B + (col + 3 * (BC // 8)) * BS).to(tl.float32)
            y3 = y3 + b3[None, :]
            b4 = tl.load(B + (col + 4 * (BC // 8)) * BS).to(tl.float32)
            y4 = y4 + b4[None, :]
            b5 = tl.load(B + (col + 5 * (BC // 8)) * BS).to(tl.float32)
            y5 = y5 + b5[None, :]
            b6 = tl.load(B + (col + 6 * (BC // 8)) * BS).to(tl.float32)
            y6 = y6 + b6[None, :]
            b7 = tl.load(B + (col + 7 * (BC // 8)) * BS).to(tl.float32)
            y7 = y7 + b7[None, :]
        y0 = libdevice.fast_dividef(y0, 1.0 + tl.exp(-y0))
        y1 = libdevice.fast_dividef(y1, 1.0 + tl.exp(-y1))
        y2 = libdevice.fast_dividef(y2, 1.0 + tl.exp(-y2))
        y3 = libdevice.fast_dividef(y3, 1.0 + tl.exp(-y3))
        y4 = libdevice.fast_dividef(y4, 1.0 + tl.exp(-y4))
        y5 = libdevice.fast_dividef(y5, 1.0 + tl.exp(-y5))
        y6 = libdevice.fast_dividef(y6, 1.0 + tl.exp(-y6))
        y7 = libdevice.fast_dividef(y7, 1.0 + tl.exp(-y7))
        if INPLACE:
            destination = X
        else:
            destination = Out
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 0 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y0, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 0 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y0, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 1 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y1, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 1 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y1, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 2 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y2, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 2 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y2, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 3 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y3, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 3 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y3, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 4 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y4, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 4 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y4, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 5 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y5, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 5 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y5, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 6 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y6, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 6 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y6, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
        if N % BR == 0:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 7 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y7, destination, P).to(
                    destination.dtype.element_ty
                ),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                tl.make_block_ptr(
                    base=destination,
                    shape=(N, C),
                    strides=(C, 1),
                    offsets=(tl.program_id(0) * BR, 7 * (BC // 8)),
                    block_shape=(BR, BC // 8),
                    order=(1, 0),
                ),
                _output551(y7, destination, P).to(
                    destination.dtype.element_ty
                ),
                boundary_check=(0,),
            )
    else:
        row = tl.program_id(0) * BR + tl.arange(0, BR)
        col = tl.arange(0, BC // 8)
        if N * C + BR * BC < 2147483648:
            offset = row[:, None] * C + col[None, :]
        else:
            offset = row[:, None].to(tl.int64) * C + col[None, :]
        if N % BR == 0:
            x0 = tl.load(X + offset + 0 * (BC // 8)).to(tl.float32)
        else:
            x0 = tl.load(X + offset + 0 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x1 = tl.load(X + offset + 1 * (BC // 8)).to(tl.float32)
        else:
            x1 = tl.load(X + offset + 1 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x2 = tl.load(X + offset + 2 * (BC // 8)).to(tl.float32)
        else:
            x2 = tl.load(X + offset + 2 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x3 = tl.load(X + offset + 3 * (BC // 8)).to(tl.float32)
        else:
            x3 = tl.load(X + offset + 3 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x4 = tl.load(X + offset + 4 * (BC // 8)).to(tl.float32)
        else:
            x4 = tl.load(X + offset + 4 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x5 = tl.load(X + offset + 5 * (BC // 8)).to(tl.float32)
        else:
            x5 = tl.load(X + offset + 5 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x6 = tl.load(X + offset + 6 * (BC // 8)).to(tl.float32)
        else:
            x6 = tl.load(X + offset + 6 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        if N % BR == 0:
            x7 = tl.load(X + offset + 7 * (BC // 8)).to(tl.float32)
        else:
            x7 = tl.load(X + offset + 7 * (BC // 8), row[:, None] < N, 0).to(
                tl.float32
            )
        total = tl.sum(
            x0 * x0
            + x1 * x1
            + (x2 * x2 + x3 * x3)
            + (x4 * x4 + x5 * x5 + (x6 * x6 + x7 * x7)),
            1,
        )
        if EPS < 1e-18:
            norm = tl.maximum(libdevice.sqrt(total), EPS)
            v0 = tl.div_rn(x0, norm[:, None]) * SCALE
            v1 = tl.div_rn(x1, norm[:, None]) * SCALE
            v2 = tl.div_rn(x2, norm[:, None]) * SCALE
            v3 = tl.div_rn(x3, norm[:, None]) * SCALE
            v4 = tl.div_rn(x4, norm[:, None]) * SCALE
            v5 = tl.div_rn(x5, norm[:, None]) * SCALE
            v6 = tl.div_rn(x6, norm[:, None]) * SCALE
            v7 = tl.div_rn(x7, norm[:, None]) * SCALE
        else:
            inv = tl.minimum(_hardware_rsq421(total), 1.0 / EPS)
            v0 = x0 * inv[:, None] * SCALE
            v1 = x1 * inv[:, None] * SCALE
            v2 = x2 * inv[:, None] * SCALE
            v3 = x3 * inv[:, None] * SCALE
            v4 = x4 * inv[:, None] * SCALE
            v5 = x5 * inv[:, None] * SCALE
            v6 = x6 * inv[:, None] * SCALE
            v7 = x7 * inv[:, None] * SCALE
        g0 = tl.load(G + (col + 0 * (BC // 8)) * GS).to(tl.float32)
        y0 = v0 * g0[None, :]
        g1 = tl.load(G + (col + 1 * (BC // 8)) * GS).to(tl.float32)
        y1 = v1 * g1[None, :]
        g2 = tl.load(G + (col + 2 * (BC // 8)) * GS).to(tl.float32)
        y2 = v2 * g2[None, :]
        g3 = tl.load(G + (col + 3 * (BC // 8)) * GS).to(tl.float32)
        y3 = v3 * g3[None, :]
        g4 = tl.load(G + (col + 4 * (BC // 8)) * GS).to(tl.float32)
        y4 = v4 * g4[None, :]
        g5 = tl.load(G + (col + 5 * (BC // 8)) * GS).to(tl.float32)
        y5 = v5 * g5[None, :]
        g6 = tl.load(G + (col + 6 * (BC // 8)) * GS).to(tl.float32)
        y6 = v6 * g6[None, :]
        g7 = tl.load(G + (col + 7 * (BC // 8)) * GS).to(tl.float32)
        y7 = v7 * g7[None, :]
        if BIAS:
            b0 = tl.load(B + (col + 0 * (BC // 8)) * BS).to(tl.float32)
            y0 = y0 + b0[None, :]
            b1 = tl.load(B + (col + 1 * (BC // 8)) * BS).to(tl.float32)
            y1 = y1 + b1[None, :]
            b2 = tl.load(B + (col + 2 * (BC // 8)) * BS).to(tl.float32)
            y2 = y2 + b2[None, :]
            b3 = tl.load(B + (col + 3 * (BC // 8)) * BS).to(tl.float32)
            y3 = y3 + b3[None, :]
            b4 = tl.load(B + (col + 4 * (BC // 8)) * BS).to(tl.float32)
            y4 = y4 + b4[None, :]
            b5 = tl.load(B + (col + 5 * (BC // 8)) * BS).to(tl.float32)
            y5 = y5 + b5[None, :]
            b6 = tl.load(B + (col + 6 * (BC // 8)) * BS).to(tl.float32)
            y6 = y6 + b6[None, :]
            b7 = tl.load(B + (col + 7 * (BC // 8)) * BS).to(tl.float32)
            y7 = y7 + b7[None, :]
        y0 = libdevice.fast_dividef(y0, 1.0 + tl.exp(-y0))
        y1 = libdevice.fast_dividef(y1, 1.0 + tl.exp(-y1))
        y2 = libdevice.fast_dividef(y2, 1.0 + tl.exp(-y2))
        y3 = libdevice.fast_dividef(y3, 1.0 + tl.exp(-y3))
        y4 = libdevice.fast_dividef(y4, 1.0 + tl.exp(-y4))
        y5 = libdevice.fast_dividef(y5, 1.0 + tl.exp(-y5))
        y6 = libdevice.fast_dividef(y6, 1.0 + tl.exp(-y6))
        y7 = libdevice.fast_dividef(y7, 1.0 + tl.exp(-y7))
        if INPLACE:
            destination = X
        else:
            destination = Out
        if N % BR == 0:
            tl.store(
                destination + offset + 0 * (BC // 8),
                _output551(y0, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 0 * (BC // 8),
                _output551(y0, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 1 * (BC // 8),
                _output551(y1, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 1 * (BC // 8),
                _output551(y1, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 2 * (BC // 8),
                _output551(y2, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 2 * (BC // 8),
                _output551(y2, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 3 * (BC // 8),
                _output551(y3, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 3 * (BC // 8),
                _output551(y3, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 4 * (BC // 8),
                _output551(y4, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 4 * (BC // 8),
                _output551(y4, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 5 * (BC // 8),
                _output551(y5, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 5 * (BC // 8),
                _output551(y5, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 6 * (BC // 8),
                _output551(y6, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 6 * (BC // 8),
                _output551(y6, destination, P),
                row[:, None] < N,
            )
        if N % BR == 0:
            tl.store(
                destination + offset + 7 * (BC // 8),
                _output551(y7, destination, P),
                cache_modifier=".cg",
            )
        else:
            tl.store(
                destination + offset + 7 * (BC // 8),
                _output551(y7, destination, P),
                row[:, None] < N,
            )


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    c = x.size(1)
    n = x.numel() // c
    if c != 512 or n < 4096:
        dtype = (
            x.dtype
            if x.dtype == gamma.dtype
            else torch.promote_types(x.dtype, gamma.dtype)
        )
        if (
            dtype == x.dtype
            and (not torch._C._overlaps(x, gamma))
            and (bias is None or not torch._C._overlaps(x, bias))
        ):
            out = x
        else:
            out = torch.empty_strided(
                x.shape, x.stride(), dtype=dtype, device=x.device
            )
        b = gamma if bias is None else bias
        gs874 = gamma.stride(0)
        bs874 = gs874 if bias is None else b.stride(0)
        bc = triton.next_power_of_2(c)
        elements = (
            1024
            if bc == 256 and n >= 4096
            else 4096 if bc >= 512 and n >= 16384 else 2048
        )
        br = max(1, min(16, elements // bc))
        if n:
            params = (
                n,
                c,
                gs874,
                bs874,
                c**0.5 if rms_scale is None else rms_scale,
                eps,
                bias is not None,
                bc,
                br,
            )
            warps = 4 if bc >= 512 and n >= 16384 else 2
            if out is x:
                if bias is None:
                    _tile_inplace0_nb.run(
                        x,
                        gamma,
                        params,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=warps,
                        enable_fp_fusion=False,
                        allow_flush_denorm=False,
                    )
                else:
                    _tile_inplace0.run(
                        x,
                        gamma,
                        b,
                        params,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=warps,
                        enable_fp_fusion=False,
                        allow_flush_denorm=False,
                    )
            elif bias is None:
                if out is x:
                    _tile0_nbip.run(
                        x,
                        gamma,
                        params,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=warps,
                        enable_fp_fusion=False,
                        allow_flush_denorm=False,
                    )
                else:
                    _tile0_nb.run(
                        x,
                        gamma,
                        out,
                        params,
                        grid=((n + br - 1) // br,),
                        warmup=False,
                        num_warps=warps,
                        enable_fp_fusion=False,
                        allow_flush_denorm=False,
                    )
            elif out is x:
                _tile0_ip.run(
                    x,
                    gamma,
                    b,
                    params,
                    grid=((n + br - 1) // br,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                    allow_flush_denorm=False,
                )
            else:
                _tile0.run(
                    x,
                    gamma,
                    b,
                    out,
                    params,
                    grid=((n + br - 1) // br,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                    allow_flush_denorm=False,
                )
        return out
    dtype = (
        x.dtype
        if x.dtype == gamma.dtype
        else torch.promote_types(x.dtype, gamma.dtype)
    )
    if (
        dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            x.shape, x.stride(), dtype=dtype, device=x.device
        )
    b = gamma if bias is None else bias
    gs874 = gamma.stride(0)
    bs874 = gs874 if bias is None else b.stride(0)
    br = 8
    if bias is None:
        if out is x:
            _parts_nbip.run(
                x,
                gamma,
                (
                    n,
                    c,
                    gs874,
                    bs874,
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                    bias is not None,
                    c,
                    br,
                    out is x,
                ),
                grid=((n + br - 1) // br,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
                allow_flush_denorm=False,
            )
        else:
            _parts_nb.run(
                x,
                gamma,
                out,
                (
                    n,
                    c,
                    gs874,
                    bs874,
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                    bias is not None,
                    c,
                    br,
                    out is x,
                ),
                grid=((n + br - 1) // br,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
                allow_flush_denorm=False,
            )
    elif out is x:
        _parts_ip.run(
            x,
            gamma,
            b,
            (
                n,
                c,
                gs874,
                bs874,
                c**0.5 if rms_scale is None else rms_scale,
                eps,
                bias is not None,
                c,
                br,
                out is x,
            ),
            grid=((n + br - 1) // br,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
            allow_flush_denorm=False,
        )
    else:
        _parts.run(
            x,
            gamma,
            b,
            out,
            (
                n,
                c,
                gs874,
                bs874,
                c**0.5 if rms_scale is None else rms_scale,
                eps,
                bias is not None,
                c,
                br,
                out is x,
            ),
            grid=((n + br - 1) // br,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
            allow_flush_denorm=False,
        )
    return out


@triton.jit
def _tile_inplace0_nb(X, G, P: tl.constexpr):
    _tile_inplace0(X, G, G, P)


@triton.jit
def _parts_nb(X, G, Out, P: tl.constexpr):
    _parts(X, G, G, Out, P)


@triton.jit
def _parts_ip(X, G, B, P: tl.constexpr):
    _parts(X, G, B, X, P)


@triton.jit
def _parts_nbip(X, G, P: tl.constexpr):
    _parts(X, G, G, X, P)


@triton.jit
def _tile0_nb(X, G, Out, P: tl.constexpr):
    _tile0(X, G, G, Out, P)


@triton.jit
def _tile0_ip(X, G, B, P: tl.constexpr):
    _tile0(X, G, B, X, P)


@triton.jit
def _tile0_nbip(X, G, P: tl.constexpr):
    _tile0(X, G, G, X, P)


@triton.jit
def _hardware_rsq421(value):
    return tl.inline_asm_elementwise(
        "v_rsq_f32 $0, $1",
        constraints="=v,v",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _output551(value, pointer, P: tl.constexpr):
    if pointer.dtype.element_ty == tl.bfloat16:
        value = tl.where(value >= 3.39617752923046e38, float("inf"), value)
        return value.to(tl.bfloat16, fp_downcast_rounding="rtz")
    else:
        return value


__all__ = ["wan_rmsnorm_silu"]
