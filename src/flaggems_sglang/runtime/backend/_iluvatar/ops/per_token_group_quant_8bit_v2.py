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

import triton
import triton.language as tl


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _encode_magnitude(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits
    normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
    sub = (v + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    result = tl.where(mag < 1015021568, sub, tl.minimum(normal, 127))
    return result


@triton.jit
def _quantize_value(
    x, scale, EPS: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr
):
    if LO == -HI and HI > 0 and (EPS > 0):
        if HI == 448.0 and EPS >= 5.266214694141234e-36:
            q = tl.div_rn(tl.abs(x), scale[:, None])
        else:
            q = tl.minimum(
                tl.div_rn(tl.abs(x), scale[:, None]),
                HI,
                propagate_nan=tl.PropagateNan.ALL,
            )
        sign = x.to(tl.uint32, bitcast=True) >> 24 & 128
    else:
        signed = tl.minimum(
            tl.maximum(
                tl.div_rn(x, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
        sign = signed.to(tl.uint32, bitcast=True) >> 24 & 128
        q = tl.abs(signed)
    return (_encode_magnitude(q) | sign).to(tl.uint8)


@triton.jit
def _quant(
    X,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    EPS: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    BG: tl.constexpr,
    BR: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BG)
    ng = H // G
    row = r // ng
    col = (r % ng)[:, None] * G + c[None, :]
    x = tl.load(
        X + row[:, None] * X0 + col * X1,
        (r[:, None] < N * ng) & (c[None, :] < G),
        other=0,
    ).to(tl.float32)
    if EPS >= 0:
        amax = tl.maximum(
            tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1)
            .to(tl.float32, bitcast=True)
            .to(tl.int32, bitcast=True),
            tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
        ).to(tl.float32, bitcast=True)
    else:
        amax = tl.maximum(
            tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
                tl.float32, bitcast=True
            ),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if HI == 448.0 and EPS >= 1e-10:
        scale = tl.fma(
            amax, 0.002232142724096775, amax * 1.3304608803554885e-10
        )
    else:
        scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI)
    tl.store(
        Q + row[:, None] * Q0 + col * Q1,
        bits,
        (r[:, None] < N * ng) & (c[None, :] < G),
    )
    tl.store(S + row * S0 + r % ng * S1, scale, r < N * ng)


@triton.jit
def _dense(X, Q: tl.pointer_type(tl.uint8), S, P: tl.constexpr):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    BG: tl.constexpr = P[5]
    BR: tl.constexpr = P[6]
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.max_contiguous(tl.arange(0, BG), 4 if BG >= 4 else BG)
    if R % BR == 0 and G == BG:
        x = tl.load(X + r[:, None] * G + c[None, :]).to(tl.float32)
    else:
        x = tl.load(
            X + r[:, None] * G + c[None, :],
            (r[:, None] < R) & (c[None, :] < G),
            other=0,
        ).to(tl.float32)
    if EPS >= 0:
        amax = tl.maximum(
            tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1)
            .to(tl.float32, bitcast=True)
            .to(tl.int32, bitcast=True),
            tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
        ).to(tl.float32, bitcast=True)
    else:
        amax = tl.maximum(
            tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
                tl.float32, bitcast=True
            ),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if HI == 448.0 and EPS >= 1e-10:
        scale = tl.fma(
            amax, 0.002232142724096775, amax * 1.3304608803554885e-10
        )
    else:
        scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI)
    if R % BR == 0 and G == BG:
        tl.store(Q + r[:, None] * G + c[None, :], bits)
        tl.store(S + r, scale)
    else:
        tl.store(
            Q + r[:, None] * G + c[None, :],
            bits,
            (r[:, None] < R) & (c[None, :] < G),
        )
        tl.store(S + r, scale, r < R)


@triton.jit
def _loop(X, Q: tl.pointer_type(tl.uint8), S, P: tl.constexpr):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    BG: tl.constexpr = P[5]
    BR: tl.constexpr = P[6]
    for block in range(tl.program_id(0), tl.cdiv(R, BR), tl.num_programs(0)):
        r = block * BR + tl.arange(0, BR)
        c = tl.max_contiguous(tl.arange(0, BG), 4 if BG >= 4 else BG)
        if R % BR == 0 and G == BG:
            x = tl.load(X + r[:, None] * G + c[None, :]).to(tl.float32)
        else:
            x = tl.load(
                X + r[:, None] * G + c[None, :],
                (r[:, None] < R) & (c[None, :] < G),
                other=0,
            ).to(tl.float32)
        if EPS >= 0:
            amax = tl.maximum(
                tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1)
                .to(tl.float32, bitcast=True)
                .to(tl.int32, bitcast=True),
                tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
            ).to(tl.float32, bitcast=True)
        else:
            amax = tl.maximum(
                tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
                    tl.float32, bitcast=True
                ),
                EPS,
                propagate_nan=tl.PropagateNan.ALL,
            )
        if HI == 448.0 and EPS >= 1e-10:
            scale = tl.fma(
                amax, 0.002232142724096775, amax * 1.3304608803554885e-10
            )
        else:
            scale = tl.div_rn(amax, HI)
        bits = _quantize_value(x, scale, EPS, LO, HI)
        if R % BR == 0 and G == BG:
            tl.store(Q + r[:, None] * G + c[None, :], bits)
            tl.store(S + r, scale)
        else:
            tl.store(
                Q + r[:, None] * G + c[None, :],
                bits,
                (r[:, None] < R) & (c[None, :] < G),
            )
            tl.store(S + r, scale, r < R)


@triton.jit
def _scale_pair_loop(X, Q: tl.pointer_type(tl.uint8), S, P: tl.constexpr):
    R: tl.constexpr = P[0]
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    for block in range(tl.program_id(0), R // 4, tl.num_programs(0)):
        c = tl.max_contiguous(tl.arange(0, 128), 4)
        r0 = block * 4 + 0 + tl.arange(0, 1)
        x0 = tl.load(X + r0[:, None] * 128 + c[None, :]).to(tl.float32)
        r1 = block * 4 + 1 + tl.arange(0, 1)
        x1 = tl.load(X + r1[:, None] * 128 + c[None, :]).to(tl.float32)
        r2 = block * 4 + 2 + tl.arange(0, 1)
        x2 = tl.load(X + r2[:, None] * 128 + c[None, :]).to(tl.float32)
        r3 = block * 4 + 3 + tl.arange(0, 1)
        x3 = tl.load(X + r3[:, None] * 128 + c[None, :]).to(tl.float32)
        if EPS >= 0:
            amax0 = tl.maximum(
                tl.max(tl.abs(x0).to(tl.int32, bitcast=True), 1)
                .to(tl.float32, bitcast=True)
                .to(tl.int32, bitcast=True),
                tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
            ).to(tl.float32, bitcast=True)
        else:
            amax0 = tl.maximum(
                tl.max(tl.abs(x0).to(tl.int32, bitcast=True), 1).to(
                    tl.float32, bitcast=True
                ),
                EPS,
                propagate_nan=tl.PropagateNan.ALL,
            )
        if HI == 448.0 and EPS >= 1e-10:
            scale0 = tl.fma(
                amax0, 0.002232142724096775, amax0 * 1.3304608803554885e-10
            )
        else:
            scale0 = tl.div_rn(amax0, HI)
        bits0 = _quantize_value(x0, scale0, EPS, LO, HI)
        if EPS >= 0:
            amax1 = tl.maximum(
                tl.max(tl.abs(x1).to(tl.int32, bitcast=True), 1)
                .to(tl.float32, bitcast=True)
                .to(tl.int32, bitcast=True),
                tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
            ).to(tl.float32, bitcast=True)
        else:
            amax1 = tl.maximum(
                tl.max(tl.abs(x1).to(tl.int32, bitcast=True), 1).to(
                    tl.float32, bitcast=True
                ),
                EPS,
                propagate_nan=tl.PropagateNan.ALL,
            )
        if HI == 448.0 and EPS >= 1e-10:
            scale1 = tl.fma(
                amax1, 0.002232142724096775, amax1 * 1.3304608803554885e-10
            )
        else:
            scale1 = tl.div_rn(amax1, HI)
        bits1 = _quantize_value(x1, scale1, EPS, LO, HI)
        if EPS >= 0:
            amax2 = tl.maximum(
                tl.max(tl.abs(x2).to(tl.int32, bitcast=True), 1)
                .to(tl.float32, bitcast=True)
                .to(tl.int32, bitcast=True),
                tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
            ).to(tl.float32, bitcast=True)
        else:
            amax2 = tl.maximum(
                tl.max(tl.abs(x2).to(tl.int32, bitcast=True), 1).to(
                    tl.float32, bitcast=True
                ),
                EPS,
                propagate_nan=tl.PropagateNan.ALL,
            )
        if HI == 448.0 and EPS >= 1e-10:
            scale2 = tl.fma(
                amax2, 0.002232142724096775, amax2 * 1.3304608803554885e-10
            )
        else:
            scale2 = tl.div_rn(amax2, HI)
        bits2 = _quantize_value(x2, scale2, EPS, LO, HI)
        if EPS >= 0:
            amax3 = tl.maximum(
                tl.max(tl.abs(x3).to(tl.int32, bitcast=True), 1)
                .to(tl.float32, bitcast=True)
                .to(tl.int32, bitcast=True),
                tl.full((), EPS, tl.float32).to(tl.int32, bitcast=True),
            ).to(tl.float32, bitcast=True)
        else:
            amax3 = tl.maximum(
                tl.max(tl.abs(x3).to(tl.int32, bitcast=True), 1).to(
                    tl.float32, bitcast=True
                ),
                EPS,
                propagate_nan=tl.PropagateNan.ALL,
            )
        if HI == 448.0 and EPS >= 1e-10:
            scale3 = tl.fma(
                amax3, 0.002232142724096775, amax3 * 1.3304608803554885e-10
            )
        else:
            scale3 = tl.div_rn(amax3, HI)
        bits3 = _quantize_value(x3, scale3, EPS, LO, HI)
        tl.store(Q + r0[:, None] * 128 + c[None, :], bits0)
        tl.store(Q + r1[:, None] * 128 + c[None, :], bits1)
        tl.store(Q + r2[:, None] * 128 + c[None, :], bits2)
        tl.store(Q + r3[:, None] * 128 + c[None, :], bits3)
        packed0 = scale0.to(tl.uint32, bitcast=True).to(tl.uint64) | (
            scale1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        )
        tl.store(
            S.to(tl.pointer_type(tl.uint64)) + block * 2 + 0 + tl.arange(0, 1),
            packed0,
        )
        packed1 = scale2.to(tl.uint32, bitcast=True).to(tl.uint64) | (
            scale3.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        )
        tl.store(
            S.to(tl.pointer_type(tl.uint64)) + block * 2 + 1 + tl.arange(0, 1),
            packed1,
        )


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    n, h = input.shape
    rows = n * (h // group_size)
    br = (
        1
        if n * h >= 12582912
        else 4 if 16 <= rows <= 32 else 2 if rows <= 2048 else 4
    )
    warps = 1
    if (
        rows
        and h % group_size == 0
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
    ):
        cap = 16384 if n * h >= 12582912 else 4096
        kernel = _loop if rows > cap * br else _dense
        if (
            n * h >= 8388608
            and group_size == 128
            and rows % 4 == 0
            and output_s.data_ptr() % 8 == 0
        ):
            _scale_pair_loop.run(
                input,
                output_q,
                output_s,
                (rows, group_size, eps, min_8bit, max_8bit, 128, 1),
                grid=(min(rows // 4, cap * 2 if n * h < 12582912 else cap),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            kernel.run(
                input,
                output_q,
                output_s,
                (
                    rows,
                    group_size,
                    eps,
                    min_8bit,
                    max_8bit,
                    triton.next_power_of_2(group_size),
                    br,
                ),
                grid=(min(triton.cdiv(rows, br), cap),),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    elif rows:
        _quant.run(
            input,
            output_q,
            output_s,
            n,
            h,
            group_size,
            *input.stride(),
            *output_q.stride(),
            *output_s.stride(),
            eps,
            min_8bit,
            max_8bit,
            triton.next_power_of_2(group_size),
            br,
            grid=(triton.cdiv(rows, br),),
            warmup=False,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return (output_q, output_s)


__all__ = ["per_token_group_quant_8bit_v2"]
