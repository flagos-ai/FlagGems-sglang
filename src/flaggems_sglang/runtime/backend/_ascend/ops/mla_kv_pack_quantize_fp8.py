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
from triton.language.extra.cann import extension as al


@triton.jit
def _encode(
    value, original, SIGN, E4: tl.constexpr, PACK: tl.constexpr = False
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    raw = value.to(tl.int32, bitcast=True)
    sign = (
        original.to(tl.float32).to(tl.int32, bitcast=True) >> 24 ^ SIGN
    ).to(tl.int16) & tl.full((), 128, tl.int16)
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.int32, bitcast=True),
    )
    exponent = tl.maximum(mag & 2139095040, 128 - BIAS << 23)
    magic = (exponent + (23 - M << 23)).to(tl.float32, bitcast=True)
    rounded = (
        (mag.to(tl.float32, bitcast=True) + magic)
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    exponent_code = (exponent >> 23 - M).to(tl.int16)
    code = rounded + exponent_code - (128 - BIAS << M)
    code = tl.where(value != value, tl.full((), 127, tl.int16), code)
    if PACK:
        return code | sign.to(tl.int16)
    else:
        return (code | sign.to(tl.int16)).to(tl.uint8)


@triton.jit
def _encode_bf16_unit(
    original, SIGN: tl.constexpr, E4: tl.constexpr, PACK: tl.constexpr = False
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    SHIFT: tl.constexpr = 7 - M
    MAX_BITS: tl.constexpr = 17376 if E4 else 18272
    raw = original.to(tl.int16, bitcast=True)
    magnitude = tl.minimum(
        raw & tl.full((), 32767, tl.int16), tl.full((), MAX_BITS, tl.int16)
    )
    parity = magnitude >> SHIFT & tl.full((), 1, tl.int16)
    normal = (
        magnitude
        + tl.full((), (1 << SHIFT - 1) - 1 - (127 - BIAS << 7), tl.int16)
        + parity
        >> SHIFT
    )
    half_mag = tl.abs(original.to(tl.float16))
    magic = tl.full((), 2.0 if E4 else 0.015625, tl.float16)
    sub = (half_mag + magic).to(tl.int16, bitcast=True)
    code = tl.where(
        half_mag
        < tl.full((), 0.015625 if E4 else 6.103515625e-05, tl.float16),
        sub,
        normal,
    )
    sign = (raw >> 8 ^ SIGN) & tl.full((), 128, tl.int16)
    if PACK:
        return tl.where(
            half_mag != half_mag,
            tl.full((), 127 | SIGN, tl.int16),
            code | sign,
        )
    else:
        return tl.where(
            half_mag != half_mag,
            tl.full((), 127 | SIGN, tl.int16),
            code | sign,
        ).to(tl.uint8)


@triton.jit
def _encode_bf16_unit_pair(
    k_original,
    v_original,
    k_SIGN: tl.constexpr,
    v_SIGN: tl.constexpr,
    E4: tl.constexpr,
    k_PACK: tl.constexpr,
    v_PACK: tl.constexpr,
):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    SHIFT: tl.constexpr = 7 - M
    MAX_BITS: tl.constexpr = 17376 if E4 else 18272
    k_raw = k_original.to(tl.int16, bitcast=True)
    v_raw = v_original.to(tl.int16, bitcast=True)
    k_sign_shifted = k_raw >> 8
    v_sign_shifted = v_raw >> 8
    k_sign_xored = k_sign_shifted ^ k_SIGN
    v_sign_xored = v_sign_shifted ^ v_SIGN
    k_sign = k_sign_xored & tl.full((), 128, tl.int16)
    v_sign = v_sign_xored & tl.full((), 128, tl.int16)
    k_masked = k_raw & tl.full((), 32767, tl.int16)
    v_masked = v_raw & tl.full((), 32767, tl.int16)
    k_magnitude = tl.minimum(k_masked, tl.full((), MAX_BITS, tl.int16))
    v_magnitude = tl.minimum(v_masked, tl.full((), MAX_BITS, tl.int16))
    k_half_mag = k_masked.to(tl.bfloat16, bitcast=True).to(tl.float16)
    v_half_mag = v_masked.to(tl.bfloat16, bitcast=True).to(tl.float16)
    k_shifted = k_magnitude >> SHIFT
    v_shifted = v_magnitude >> SHIFT
    k_parity = k_shifted & tl.full((), 1, tl.int16)
    v_parity = v_shifted & tl.full((), 1, tl.int16)
    k_biased = k_magnitude + tl.full(
        (), (1 << SHIFT - 1) - 1 - (127 - BIAS << 7), tl.int16
    )
    v_biased = v_magnitude + tl.full(
        (), (1 << SHIFT - 1) - 1 - (127 - BIAS << 7), tl.int16
    )
    k_biased_parity = k_biased + k_parity
    v_biased_parity = v_biased + v_parity
    k_normal = k_biased_parity >> SHIFT
    v_normal = v_biased_parity >> SHIFT
    k_magic = tl.full((), 2.0 if E4 else 0.015625, tl.float16)
    v_magic = tl.full((), 2.0 if E4 else 0.015625, tl.float16)
    k_sub_added = k_half_mag + k_magic
    v_sub_added = v_half_mag + v_magic
    k_sub = k_sub_added.to(tl.int16, bitcast=True)
    v_sub = v_sub_added.to(tl.int16, bitcast=True)
    k_small = k_half_mag < tl.full(
        (), 0.015625 if E4 else 6.103515625e-05, tl.float16
    )
    v_small = v_half_mag < tl.full(
        (), 0.015625 if E4 else 6.103515625e-05, tl.float16
    )
    k_code = tl.where(k_small, k_sub, k_normal)
    v_code = tl.where(v_small, v_sub, v_normal)
    k_is_nan = k_half_mag != k_half_mag
    v_is_nan = v_half_mag != v_half_mag
    k_signed = k_code | k_sign
    v_signed = v_code | v_sign
    k_encoded = tl.where(
        k_is_nan, tl.full((), 127 | k_SIGN, tl.int16), k_signed
    )
    v_encoded = tl.where(
        v_is_nan, tl.full((), 127 | v_SIGN, tl.int16), v_signed
    )
    if k_PACK:
        k_result = k_encoded
    else:
        k_result = k_encoded.to(tl.uint8)
    if v_PACK:
        v_result = v_encoded
    else:
        v_result = v_encoded.to(tl.uint8)
    return (k_result, v_result)


@triton.jit
def _scaled_encode(
    original,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    PACK: tl.constexpr = False,
):
    if original.dtype == tl.bfloat16 and (SCALE == 1.0 or SCALE == -1.0):
        return _encode_bf16_unit(original, SIGN, E4, PACK)
    return _encode(original.to(tl.float32) * SCALE, original, SIGN, E4, PACK)


@triton.jit
def _pack_codes(code):
    if len(code.shape) == 1:
        lo = al.extract_slice(code, (0,), (code.shape[0] // 2,), (2,))
        hi = al.extract_slice(code, (1,), (code.shape[0] // 2,), (2,))
    else:
        lo = al.extract_slice(
            code, (0, 0), (code.shape[0], code.shape[1] // 2), (1, 2)
        )
        hi = al.extract_slice(
            code, (0, 1), (code.shape[0], code.shape[1] // 2), (1, 2)
        )
    return (lo & tl.full((), 255, tl.int16) | hi << 8).to(tl.uint16)


@triton.jit
def _copy_stream_tile(
    X,
    K_OUT,
    work,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_OFFSET: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(WIDTH, BLOCK) if WIDTH > 0 else 1
    row = work // TILES
    column = work % TILES * BLOCK + tl.arange(0, BLOCK)
    token = row // H
    head = row % H
    if WIDTH % BLOCK == 0:
        original = tl.load(X + token * TS + head * HS + column)
    else:
        original = tl.load(
            X + token * TS + head * HS + column, column < WIDTH, other=0
        )
    PACK: tl.constexpr = K_OUT.dtype.element_ty == tl.uint16
    encoded = _scaled_encode(original, SCALE, SIGN, E4, PACK)
    if PACK:
        pairs = work % TILES * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        pointers = K_OUT + row * (OUT_WIDTH // 2) + OUT_OFFSET // 2 + pairs
        if WIDTH % BLOCK == 0:
            tl.store(pointers, _pack_codes(encoded))
        else:
            tl.store(pointers, _pack_codes(encoded), pairs < WIDTH // 2)
    elif WIDTH % BLOCK == 0:
        tl.store(K_OUT + row * OUT_WIDTH + OUT_OFFSET + column, encoded)
    else:
        tl.store(
            K_OUT + row * OUT_WIDTH + OUT_OFFSET + column,
            encoded,
            column < WIDTH,
        )


@triton.jit
def _copy_pe_shared(
    P,
    K_OUT,
    work,
    H: tl.constexpr,
    PE: tl.constexpr,
    N: tl.constexpr,
    PS: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
    BR: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(PE, BLOCK) if PE > 0 else 1
    GROUPS: tl.constexpr = triton.cdiv(H, BR)
    row = work // TILES
    token = row // GROUPS
    heads = row % GROUPS * BR + tl.arange(0, BR)
    columns = work % TILES * BLOCK + tl.arange(0, BLOCK)
    if PE % BLOCK == 0:
        original = tl.load(P + token * PS + columns)
    else:
        original = tl.load(P + token * PS + columns, columns < PE, other=0)
    PACK: tl.constexpr = K_OUT.dtype.element_ty == tl.uint16
    encoded = _scaled_encode(original, SCALE, SIGN, E4, PACK)
    if PACK:
        pairs = work % TILES * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        pointers = (
            K_OUT
            + (token * H + heads[:, None]) * ((N + PE) // 2)
            + N // 2
            + pairs[None, :]
        )
        packed = _pack_codes(encoded)
        if H % BR == 0 and PE % BLOCK == 0:
            tl.store(pointers, packed[None, :])
        else:
            tl.store(
                pointers,
                packed[None, :],
                (heads[:, None] < H) & (pairs[None, :] < PE // 2),
            )
    else:
        pointers = (
            K_OUT
            + (token * H + heads[:, None]) * (N + PE)
            + N
            + columns[None, :]
        )
        if H % BR == 0 and PE % BLOCK == 0:
            tl.store(pointers, encoded[None, :])
        else:
            tl.store(
                pointers,
                encoded[None, :],
                (heads[:, None] < H) & (columns[None, :] < PE),
            )


@triton.jit
def _copy_pe_tokens(
    P,
    K_OUT,
    work,
    H: tl.constexpr,
    PE: tl.constexpr,
    N: tl.constexpr,
    PS: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    TB: tl.constexpr,
):
    tokens = work * TB + tl.arange(0, TB)
    columns = tl.arange(0, PE)
    original = tl.load(P + tokens[:, None] * PS + columns[None, :])
    PACK: tl.constexpr = K_OUT.dtype.element_ty == tl.uint16
    encoded = _scaled_encode(original, SCALE, SIGN, E4, PACK)
    if PACK:
        packed = _pack_codes(encoded)
        expanded = tl.broadcast_to(packed[:, None, :], (TB, H, PE // 2))
        values = tl.reshape(expanded, (TB * H, PE // 2))
        rows = work * (TB * H) + tl.arange(0, TB * H)
        pairs = tl.arange(0, PE // 2)
        pointers = (
            K_OUT + rows[:, None] * ((N + PE) // 2) + N // 2 + pairs[None, :]
        )
        tl.store(pointers, values)
    else:
        expanded = tl.broadcast_to(encoded[:, None, :], (TB, H, PE))
        values = tl.reshape(expanded, (TB * H, PE))
        rows = work * (TB * H) + tl.arange(0, TB * H)
        pointers = K_OUT + rows[:, None] * (N + PE) + N + columns[None, :]
        tl.store(pointers, values)


@triton.jit
def _copy_v_flat(
    V,
    W,
    work,
    SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = work * BLOCK + tl.arange(0, BLOCK)
    if SIZE % BLOCK == 0:
        original = tl.load(V + offsets)
    else:
        original = tl.load(V + offsets, offsets < SIZE, other=0)
    PACK: tl.constexpr = W.dtype.element_ty == tl.uint16
    encoded = _scaled_encode(original, SCALE, SIGN, E4, PACK)
    if PACK:
        pairs = work * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        if SIZE % BLOCK == 0:
            tl.store(W + pairs, _pack_codes(encoded))
        else:
            tl.store(W + pairs, _pack_codes(encoded), pairs < SIZE // 2)
    elif SIZE % BLOCK == 0:
        tl.store(W + offsets, encoded)
    else:
        tl.store(W + offsets, encoded, offsets < SIZE)


@triton.jit
def _copy_k_group(
    K,
    K_OUT,
    work,
    R: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    PE: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(N, BLOCK) if N > 0 else 1
    lanes = tl.arange(0, 8 * BLOCK)
    rows = work // TILES * 8 + lanes // BLOCK
    columns = work % TILES * BLOCK + lanes % BLOCK
    pointers = K + work * (8 * BLOCK) + lanes
    original = tl.load(pointers)
    PACK: tl.constexpr = K_OUT.dtype.element_ty == tl.uint16
    encoded = _scaled_encode(original, SCALE, SIGN, E4, PACK)
    if PACK:
        pairs = tl.arange(0, 8 * BLOCK // 2)
        pair_rows = work // TILES * 8 + pairs // (BLOCK // 2)
        pair_columns = work % TILES * (BLOCK // 2) + pairs % (BLOCK // 2)
        output = K_OUT + pair_rows * ((N + PE) // 2) + pair_columns
        tl.store(output, _pack_codes(encoded))
    else:
        output = K_OUT + rows * (N + PE) + columns
        tl.store(output, encoded)


@triton.jit
def _copy_kv_prefetch(
    K,
    V,
    K_OUT,
    W,
    work,
    R: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    PE: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    SIZE: tl.constexpr,
    K_SCALE: tl.constexpr,
    K_SIGN: tl.constexpr,
    V_SCALE: tl.constexpr,
    V_SIGN: tl.constexpr,
    E4: tl.constexpr,
    KBLOCK: tl.constexpr,
    VBLOCK: tl.constexpr,
):
    k_TILES: tl.constexpr = triton.cdiv(N, KBLOCK) if N > 0 else 1
    k_lanes = tl.arange(0, 8 * KBLOCK)
    k_rows = work // k_TILES * 8 + k_lanes // KBLOCK
    k_columns = work % k_TILES * KBLOCK + k_lanes % KBLOCK
    k_pointers = K + work * (8 * KBLOCK) + k_lanes
    k_original = tl.load(k_pointers)
    v_offsets = work * VBLOCK + tl.arange(0, VBLOCK)
    if SIZE % VBLOCK == 0:
        v_original = tl.load(V + v_offsets)
    else:
        v_original = tl.load(V + v_offsets, v_offsets < SIZE, other=0)
    k_PACK: tl.constexpr = K_OUT.dtype.element_ty == tl.uint16
    v_PACK: tl.constexpr = W.dtype.element_ty == tl.uint16
    UNIT_PAIR: tl.constexpr = (
        k_original.dtype == tl.bfloat16
        and v_original.dtype == tl.bfloat16
        and (K_SCALE == 1.0 or K_SCALE == -1.0)
        and (V_SCALE == 1.0 or V_SCALE == -1.0)
    )
    if UNIT_PAIR:
        k_encoded, v_encoded = _encode_bf16_unit_pair(
            k_original, v_original, K_SIGN, V_SIGN, E4, k_PACK, v_PACK
        )
    else:
        k_encoded = _scaled_encode(k_original, K_SCALE, K_SIGN, E4, k_PACK)
    if k_PACK:
        k_pairs = tl.arange(0, 8 * KBLOCK // 2)
        k_pair_rows = work // k_TILES * 8 + k_pairs // (KBLOCK // 2)
        k_pair_columns = work % k_TILES * (KBLOCK // 2) + k_pairs % (
            KBLOCK // 2
        )
        k_output = K_OUT + k_pair_rows * ((N + PE) // 2) + k_pair_columns
        tl.store(k_output, _pack_codes(k_encoded))
    else:
        k_output = K_OUT + k_rows * (N + PE) + k_columns
        tl.store(k_output, k_encoded)
    if not UNIT_PAIR:
        v_encoded = _scaled_encode(v_original, V_SCALE, V_SIGN, E4, v_PACK)
    if v_PACK:
        v_pairs = work * (VBLOCK // 2) + tl.arange(0, VBLOCK // 2)
        if SIZE % VBLOCK == 0:
            tl.store(W + v_pairs, _pack_codes(v_encoded))
        else:
            tl.store(W + v_pairs, _pack_codes(v_encoded), v_pairs < SIZE // 2)
    elif SIZE % VBLOCK == 0:
        tl.store(W + v_offsets, v_encoded)
    else:
        tl.store(W + v_offsets, v_encoded, v_offsets < SIZE)


@triton.jit
def _streams(K, P, V, K_OUT, W, META: tl.constexpr, BLOCK: tl.constexpr):
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
    NT: tl.constexpr = triton.cdiv(N, BLOCK)
    PT: tl.constexpr = triton.cdiv(PE, BLOCK)
    PE_BLOCK: tl.constexpr = META[15]
    BR: tl.constexpr = META[16]
    SHARE_PE: tl.constexpr = META[17]
    FLAT_V: tl.constexpr = META[18]
    DT: tl.constexpr = triton.cdiv(D, BLOCK)
    TOTAL: tl.constexpr = R * (NT + PT + DT)
    WIDE: tl.constexpr = (
        (R // H - 1) * KS + (H - 1) * KH + N - 1 > 2147483647
        or (R // H - 1) * PS + PE - 1 > 2147483647
        or (R // H - 1) * VS + (H - 1) * VH + D - 1 > 2147483647
        or (R * (N + PE) - 1 > 2147483647)
        or (R * D - 1 > 2147483647)
        or (TOTAL + 39 > 2147483647)
        or (FLAT_V and R * D + 4095 > 2147483647)
        or (KS > 2147483647)
        or (KH > 2147483647)
        or (PS > 2147483647)
        or (VS > 2147483647)
        or (VH > 2147483647)
        or (H > 2147483647)
        or (N + PE > 2147483647)
        or (D > 2147483647)
    )
    PAIR_KV: tl.constexpr = (
        N == BLOCK
        and KH == N
        and (KS == H * N)
        and FLAT_V
        and (D == N)
        and (R % 8 == 0)
    )
    start = tl.program_id(0)
    if WIDE:
        start = start.to(tl.int64)
    if N == BLOCK and KH == N and (KS == H * N):
        if WIDE:
            stop_0 = tl.full((), R // 8, tl.int64)
        else:
            stop_0 = R // 8
        for work in range(start, stop_0, tl.num_programs(0)):
            if PAIR_KV:
                _copy_kv_prefetch(
                    K,
                    V,
                    K_OUT,
                    W,
                    work,
                    R,
                    H,
                    N,
                    PE,
                    KS,
                    KH,
                    R * D,
                    K_SCALE,
                    K_SIGN,
                    V_SCALE,
                    V_SIGN,
                    E4,
                    BLOCK,
                    4096,
                )
            else:
                _copy_k_group(
                    K,
                    K_OUT,
                    work,
                    R,
                    H,
                    N,
                    PE,
                    KS,
                    KH,
                    K_SCALE,
                    K_SIGN,
                    E4,
                    BLOCK,
                )
        if R % 8:
            if WIDE:
                stop_1 = tl.full((), R % 8, tl.int64)
            else:
                stop_1 = R % 8
            for tail in range(start, stop_1, tl.num_programs(0)):
                _copy_stream_tile(
                    K,
                    K_OUT,
                    tail + R // 8 * 8,
                    H,
                    N,
                    N + PE,
                    0,
                    KS,
                    KH,
                    K_SCALE,
                    K_SIGN,
                    E4,
                    BLOCK,
                )
    else:
        if WIDE:
            stop_2 = tl.full((), R * NT, tl.int64)
        else:
            stop_2 = R * NT
        for work in range(start, stop_2, tl.num_programs(0)):
            _copy_stream_tile(
                K,
                K_OUT,
                work,
                H,
                N,
                N + PE,
                0,
                KS,
                KH,
                K_SCALE,
                K_SIGN,
                E4,
                BLOCK,
            )
    if SHARE_PE:
        if PE == PE_BLOCK and H & H - 1 == 0 and (H * PE_BLOCK <= 4096):
            TB: tl.constexpr = 4096 // (H * PE_BLOCK)
            FULL: tl.constexpr = R // H // TB
            if WIDE:
                stop_3 = tl.full((), FULL, tl.int64)
            else:
                stop_3 = FULL
            for work in range(start, stop_3, tl.num_programs(0)):
                _copy_pe_tokens(
                    P, K_OUT, work, H, PE, N, PS, K_SCALE, K_SIGN, E4, TB
                )
            if R // H % TB:
                BASE_PE: tl.constexpr = FULL * TB * triton.cdiv(H, BR)
                TAIL_PE: tl.constexpr = R // H % TB * triton.cdiv(H, BR)
                if WIDE:
                    stop_tail = tl.full((), TAIL_PE, tl.int64)
                else:
                    stop_tail = TAIL_PE
                for work in range(start, stop_tail, tl.num_programs(0)):
                    _copy_pe_shared(
                        P,
                        K_OUT,
                        work + BASE_PE,
                        H,
                        PE,
                        N,
                        PS,
                        K_SCALE,
                        K_SIGN,
                        E4,
                        PE_BLOCK,
                        BR,
                    )
        else:
            if WIDE:
                stop_3 = tl.full(
                    (), R // H * triton.cdiv(H, BR) * PT, tl.int64
                )
            else:
                stop_3 = R // H * triton.cdiv(H, BR) * PT
            for work in range(start, stop_3, tl.num_programs(0)):
                _copy_pe_shared(
                    P,
                    K_OUT,
                    work,
                    H,
                    PE,
                    N,
                    PS,
                    K_SCALE,
                    K_SIGN,
                    E4,
                    PE_BLOCK,
                    BR,
                )
    else:
        if WIDE:
            stop_4 = tl.full((), R * PT, tl.int64)
        else:
            stop_4 = R * PT
        for work in range(start, stop_4, tl.num_programs(0)):
            _copy_stream_tile(
                P,
                K_OUT,
                work,
                H,
                PE,
                N + PE,
                N,
                PS,
                0,
                K_SCALE,
                K_SIGN,
                E4,
                BLOCK,
            )
    if not PAIR_KV:
        if FLAT_V:
            if WIDE:
                stop_5 = tl.full((), triton.cdiv(R * D, 4096), tl.int64)
            else:
                stop_5 = triton.cdiv(R * D, 4096)
            for work in range(start, stop_5, tl.num_programs(0)):
                _copy_v_flat(V, W, work, R * D, V_SCALE, V_SIGN, E4, 4096)
        else:
            if WIDE:
                stop_6 = tl.full((), R * DT, tl.int64)
            else:
                stop_6 = R * DT
            for work in range(start, stop_6, tl.num_programs(0)):
                _copy_stream_tile(
                    V, W, work, H, D, D, 0, VS, VH, V_SCALE, V_SIGN, E4, BLOCK
                )


@triton.jit(do_not_specialize=["K_OUT", "W"])
def _streams_addresses(
    K,
    P,
    V,
    K_OUT,
    W,
    COMPACT_META: tl.constexpr,
    BLOCK: tl.constexpr,
    O_ALIGN: tl.constexpr,
    W_ALIGN: tl.constexpr,
):
    LAYOUT: tl.constexpr = COMPACT_META[5]
    STRIDES: tl.constexpr = (
        (
            COMPACT_META[1] * COMPACT_META[2],
            COMPACT_META[2],
            COMPACT_META[3],
            COMPACT_META[1] * COMPACT_META[4],
            COMPACT_META[4],
            True,
        )
        if LAYOUT is None
        else LAYOUT
    )
    META: tl.constexpr = (
        COMPACT_META[0],
        COMPACT_META[1],
        COMPACT_META[2],
        COMPACT_META[3],
        COMPACT_META[4],
        STRIDES[0],
        STRIDES[1],
        STRIDES[2],
        STRIDES[3],
        STRIDES[4],
        COMPACT_META[6],
        COMPACT_META[7],
        COMPACT_META[8],
        COMPACT_META[9],
        COMPACT_META[10],
        STRIDES[5],
    )
    PE_BASE: tl.constexpr = META[3] if META[3] > 1 else 1
    PE_POWER: tl.constexpr = triton.next_power_of_2(PE_BASE)
    PE_BLOCK: tl.constexpr = PE_POWER if PE_POWER < BLOCK else BLOCK
    HEAD_POWER: tl.constexpr = triton.next_power_of_2(META[1])
    HEAD_LIMIT: tl.constexpr = BLOCK // PE_BLOCK
    PE_HEADS: tl.constexpr = (
        HEAD_POWER if HEAD_POWER < HEAD_LIMIT else HEAD_LIMIT
    )
    PE_TILES: tl.constexpr = triton.cdiv(META[3], BLOCK)
    SHARED_TILES: tl.constexpr = (
        META[0]
        // META[1]
        * triton.cdiv(META[1], PE_HEADS)
        * triton.cdiv(META[3], PE_BLOCK)
    )
    SAVED_LANES: tl.constexpr = (
        META[0] * PE_TILES * BLOCK - SHARED_TILES * PE_BLOCK
    )
    FULL_META: tl.constexpr = (
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        PE_BLOCK,
        PE_HEADS,
        SAVED_LANES >= 2097152,
        META[15],
    )
    if O_ALIGN and FULL_META[2] % 2 == 0 and (FULL_META[3] % 2 == 0):
        op = K_OUT.to(tl.uint64).to(tl.pointer_type(tl.uint16))
    else:
        op = K_OUT.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if W_ALIGN and (
        FULL_META[18]
        and FULL_META[0] * FULL_META[4] % 2 == 0
        or FULL_META[4] % 2 == 0
    ):
        wp = W.to(tl.uint64).to(tl.pointer_type(tl.uint16))
    else:
        wp = W.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if O_ALIGN:
        op = tl.multiple_of(op, 16)
    if W_ALIGN:
        wp = tl.multiple_of(wp, 16)
    _streams(K, P, V, op, wp, FULL_META, BLOCK)


def mla_kv_pack_quantize_fp8(
    k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
):
    s, heads, nope = k_nope.shape
    rope, width = (k_pe.shape[-1], v.shape[-1])
    k_out = k_nope.new_empty((s, heads, nope + rope), dtype=fp8_dtype)
    v_out = v.new_empty((s, heads, width), dtype=fp8_dtype)
    rows = s * heads
    block = 512
    total = rows * (
        (nope + block - 1) // block
        + (rope + block - 1) // block
        + (width + block - 1) // block
    )
    v_contiguous = v.is_contiguous()
    if v_contiguous and k_nope.is_contiguous() and k_pe.is_contiguous():
        layout = None
    else:
        k_strides, v_strides = (k_nope.stride(), v.stride())
        layout = (
            k_strides[0],
            k_strides[1],
            k_pe.stride(0),
            v_strides[0],
            v_strides[1],
            v_contiguous,
        )
    meta = (
        rows,
        heads,
        nope,
        rope,
        width,
        layout,
        k_scale_inv,
        v_scale_inv,
        fp8_dtype == torch.float8_e4m3fn,
        128 if math.copysign(1.0, k_scale_inv) < 0 else 0,
        128 if math.copysign(1.0, v_scale_inv) < 0 else 0,
    )
    if total:
        k_address, v_address = (k_out.data_ptr(), v_out.data_ptr())
        _streams_addresses.run(
            k_nope,
            k_pe,
            v,
            k_address,
            v_address,
            meta,
            block,
            k_address % 16 == 0,
            v_address % 16 == 0,
            grid=(min(total, 40),),
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
            warmup=False,
        )
    return (k_out, v_out)


__all__ = ["mla_kv_pack_quantize_fp8"]
