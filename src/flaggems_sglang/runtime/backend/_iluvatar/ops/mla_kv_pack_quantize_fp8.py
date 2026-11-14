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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(value, E4: tl.constexpr, original, SIGN: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    raw = value.to(tl.uint32, bitcast=True)
    sign = (
        original.to(tl.uint16, bitcast=True).to(tl.uint32) >> 8 ^ SIGN
    ) & 128
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True),
    )
    exponent = tl.maximum(mag & 2139095040, 128 - BIAS << 23)
    magic = (exponent + (23 - M << 23)).to(tl.float32, bitcast=True)
    rounded = (mag.to(tl.float32, bitcast=True) + magic).to(
        tl.uint32, bitcast=True
    ) & (1 << M + 2) - 1
    code = rounded + (exponent >> 23 - M) - (128 - BIAS << M)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _linear_work(
    K,
    P,
    V,
    K_OUT,
    W,
    tile,
    K_WIDTH: tl.constexpr,
    K_TOTAL: tl.constexpr,
    V_TOTAL: tl.constexpr,
    K_TILES: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    PS: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    E4: tl.constexpr,
    K_SIGN: tl.constexpr,
    V_SIGN: tl.constexpr,
    BLOCK: tl.constexpr,
    PE: tl.constexpr,
):
    if tile < K_TILES:
        if K_WIDTH > 0:
            index = tile * BLOCK + tl.arange(0, BLOCK)
            row = index // K_WIDTH
            column = index % K_WIDTH
            token = row // H
            head = row % H
            live = index < K_TOTAL
            a = tl.load(
                K
                + (
                    index - row * PE
                    if KS == H * N and KH == N
                    else token * KS + head * KH + column
                ),
                live & (column < N),
                other=0,
            )
            b = tl.load(
                P + token * PS + column - N, live & (column >= N), other=0
            )
            original = tl.where(column < N, a, b)
            encoded = _encode(
                original.to(tl.float32) * K_SCALE, E4, original, K_SIGN
            )
            tl.store(K_OUT + index, encoded, live)
    elif D > 0:
        index = (tile - K_TILES) * BLOCK + tl.arange(0, BLOCK)
        row = index // D
        column = index % D
        token = row // H
        head = row % H
        original = tl.load(
            V
            + (
                index
                if VS == H * D and VH == D
                else token * VS + head * VH + column
            ),
            index < V_TOTAL,
            other=0,
        )
        encoded = _encode(
            original.to(tl.float32) * V_SCALE, E4, original, V_SIGN
        )
        tl.store(W + index, encoded, index < V_TOTAL)


@triton.jit
def _linear(K, P, V, K_OUT, W, META: tl.constexpr, BLOCK: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PE: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    KS: tl.constexpr = META[5]
    KH: tl.constexpr = META[6]
    PS: tl.constexpr = META[7]
    VS: tl.constexpr = META[8]
    VH: tl.constexpr = META[9]
    K_SCALE: tl.constexpr = META[10]
    V_SCALE: tl.constexpr = META[11]
    E4: tl.constexpr = META[12]
    K_SIGN: tl.constexpr = META[13]
    V_SIGN: tl.constexpr = META[14]
    K_WIDTH: tl.constexpr = N + PE
    K_TOTAL: tl.constexpr = R * K_WIDTH
    V_TOTAL: tl.constexpr = R * D
    K_TILES: tl.constexpr = triton.cdiv(K_TOTAL, BLOCK)
    V_TILES: tl.constexpr = triton.cdiv(V_TOTAL, BLOCK)
    WIDE: tl.constexpr = (
        K_TOTAL + BLOCK - 1 > 2147483647
        or V_TOTAL + BLOCK - 1 > 2147483647
        or (R // H - 1) * KS + (H - 1) * KH + N - 1 > 2147483647
        or ((R // H - 1) * PS + PE - 1 > 2147483647)
        or ((R // H - 1) * VS + (H - 1) * VH + D - 1 > 2147483647)
        or (KS > 2147483647)
        or (KH > 2147483647)
        or (PS > 2147483647)
        or (VS > 2147483647)
        or (VH > 2147483647)
        or (H > 2147483647)
        or (K_WIDTH > 2147483647)
        or (D > 2147483647)
    )
    start = tl.program_id(0)
    if WIDE:
        start = start.to(tl.int64)
    if K_TILES + V_TILES <= 16777215:
        _linear_work(
            K,
            P,
            V,
            K_OUT,
            W,
            start,
            K_WIDTH,
            K_TOTAL,
            V_TOTAL,
            K_TILES,
            N,
            D,
            H,
            KS,
            KH,
            PS,
            VS,
            VH,
            K_SCALE,
            V_SCALE,
            E4,
            K_SIGN,
            V_SIGN,
            BLOCK,
            PE,
        )
    elif WIDE:
        for tile in range(
            start, tl.full((), K_TILES + V_TILES, tl.int64), tl.num_programs(0)
        ):
            _linear_work(
                K,
                P,
                V,
                K_OUT,
                W,
                tile,
                K_WIDTH,
                K_TOTAL,
                V_TOTAL,
                K_TILES,
                N,
                D,
                H,
                KS,
                KH,
                PS,
                VS,
                VH,
                K_SCALE,
                V_SCALE,
                E4,
                K_SIGN,
                V_SIGN,
                BLOCK,
                PE,
            )
    else:
        for tile in range(start, K_TILES + V_TILES, tl.num_programs(0)):
            _linear_work(
                K,
                P,
                V,
                K_OUT,
                W,
                tile,
                K_WIDTH,
                K_TOTAL,
                V_TOTAL,
                K_TILES,
                N,
                D,
                H,
                KS,
                KH,
                PS,
                VS,
                VH,
                K_SCALE,
                V_SCALE,
                E4,
                K_SIGN,
                V_SIGN,
                BLOCK,
                PE,
            )


@triton.jit(do_not_specialize=["O_ADDR", "W_ADDR"])
def _address_linear(
    K, P, V, O_ADDR, W_ADDR, META: tl.constexpr, BLOCK: tl.constexpr
):
    output = tl.multiple_of(
        O_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    values = tl.multiple_of(
        W_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    _linear(K, P, V, output, values, META, BLOCK)


def mla_kv_pack_quantize_fp8(
    k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
):
    s, heads, nope = k_nope.shape
    rope, width = (k_pe.shape[-1], v.shape[-1])
    k_out = torch.empty(
        (s, heads, nope + rope), device=k_nope.device, dtype=fp8_dtype
    )
    v_out = torch.empty((s, heads, width), device=v.device, dtype=fp8_dtype)
    rows = s * heads
    block = 1024
    total = (rows * (nope + rope) + block - 1) // block + (
        rows * width + block - 1
    ) // block
    k_nope_strides = k_nope.stride()
    v_strides = v.stride()
    meta = (
        rows,
        heads,
        nope,
        rope,
        width,
        k_nope_strides[0],
        k_nope_strides[1],
        k_pe.stride(0),
        v_strides[0],
        v_strides[1],
        k_scale_inv,
        v_scale_inv,
        fp8_dtype == torch.float8_e4m3fn,
        128 if math.copysign(1.0, k_scale_inv) < 0 else 0,
        128 if math.copysign(1.0, v_scale_inv) < 0 else 0,
    )
    if total:
        k_address = k_out.data_ptr()
        v_address = v_out.data_ptr()
        assert k_address % 16 == 0 and v_address % 16 == 0
        programs = total
        if total > 16777215:
            iterations = (total + 16777214) // 16777215
            programs = (total + iterations - 1) // iterations
        _address_linear.run(
            k_nope,
            k_pe,
            v,
            k_address,
            v_address,
            meta,
            block,
            grid=(programs,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (k_out, v_out)


__all__ = ["mla_kv_pack_quantize_fp8"]
