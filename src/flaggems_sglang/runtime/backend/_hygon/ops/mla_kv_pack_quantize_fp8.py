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
def _encode(
    value,
    E4: tl.constexpr,
    original,
    SIGN: tl.constexpr,
    CANON_NAN: tl.constexpr = False,
):
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
    )
    code = rounded + (exponent >> 23 - M) - (128 - BIAS << M)
    if CANON_NAN:
        return tl.where(
            value != value, tl.full((), 127, tl.uint32), code | sign
        ).to(tl.uint8)
    else:
        if SIGN == 128:
            sign = tl.where(original != original, sign ^ 128, sign)
        code = tl.where(value != value, 127, code)
        return (code | sign).to(tl.uint8)


@triton.jit
def _encode_select(
    original, SCALE: tl.constexpr, E4: tl.constexpr, SIGN: tl.constexpr
):
    if SCALE != SCALE:
        return tl.full(original.shape, 127 | SIGN, tl.uint8)
    if SCALE >= -(2.0 ** (-150)) and SCALE <= 2.0 ** (-150):
        encoded = _encode(
            original.to(tl.float32) * SCALE, E4, original, SIGN, False
        )
        return tl.where(
            tl.abs(original.to(tl.float32)) == float("inf"),
            tl.full((), 255, tl.uint8),
            encoded,
        )
    if SCALE >= 2.0**128 - 2.0**103 or SCALE <= -(2.0**128 - 2.0**103):
        encoded = _encode(
            original.to(tl.float32) * SCALE, E4, original, SIGN, False
        )
        return tl.where(original == 0, tl.full((), 255, tl.uint8), encoded)
    return _encode(original.to(tl.float32) * SCALE, E4, original, SIGN, False)


@triton.jit
def _shared(K, P, V, K_OUT, W, META: tl.constexpr):
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
    BR: tl.constexpr = META[13]
    K_SIGN: tl.constexpr = META[14]
    V_SIGN: tl.constexpr = META[15]
    WIDE: tl.constexpr = (
        (R // H - 1) * KS + (H - 1) * KH + N - 1 > 2147483647
        or (R // H - 1) * PS + PE - 1 > 2147483647
        or (R // H - 1) * VS + (H - 1) * VH + D - 1 > 2147483647
        or (R * (N + PE) - 1 > 2147483647)
        or (R * D - 1 > 2147483647)
        or (KS > 2147483647)
        or (KH > 2147483647)
        or (PS > 2147483647)
        or (VS > 2147483647)
        or (VH > 2147483647)
        or (H > 2147483647)
        or (N + PE > 2147483647)
        or (D > 2147483647)
    )
    token = tl.program_id(0)
    head_group = tl.program_id(1)
    if WIDE:
        token = token.to(tl.int64)
        head_group = head_group.to(tl.int64)
    head = head_group * BR + tl.arange(0, BR)
    rows = token * H + head
    n = tl.arange(0, triton.next_power_of_2(N) if N > 0 else 1)
    pe = tl.arange(0, triton.next_power_of_2(PE) if PE > 0 else 1)
    d = tl.arange(0, triton.next_power_of_2(D) if D > 0 else 1)
    kval = tl.load(
        K + token * KS + head[:, None] * KH + n[None, :],
        (head[:, None] < H) & (n[None, :] < N),
        other=0,
    )
    pval = tl.load(P + token * PS + pe, pe < PE, other=0)
    kout = K_OUT.to(tl.pointer_type(tl.uint8))
    vout = W.to(tl.pointer_type(tl.uint8))
    tl.store(
        kout + rows[:, None] * (N + PE) + n[None, :],
        _encode_select(kval, K_SCALE, E4, K_SIGN),
        (head[:, None] < H) & (n[None, :] < N),
    )
    tl.store(
        kout + rows[:, None] * (N + PE) + N + pe[None, :],
        _encode_select(pval, K_SCALE, E4, K_SIGN)[None, :],
        (head[:, None] < H) & (pe[None, :] < PE),
    )
    vval = tl.load(
        V + token * VS + head[:, None] * VH + d[None, :],
        (head[:, None] < H) & (d[None, :] < D),
        other=0,
    )
    tl.store(
        vout + rows[:, None] * D + d[None, :],
        _encode_select(vval, V_SCALE, E4, V_SIGN),
        (head[:, None] < H) & (d[None, :] < D),
    )


@triton.jit(do_not_specialize=["O_ADDR", "W_ADDR"])
def _address_shared(K, P, V, O_ADDR, W_ADDR, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PE: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    LAYOUT: tl.constexpr = META[5]
    if LAYOUT is None:
        STRIDES: tl.constexpr = (H * N, N, PE, H * D, D)
    else:
        STRIDES: tl.constexpr = LAYOUT
    FULL_META: tl.constexpr = (
        R,
        H,
        N,
        PE,
        D,
        STRIDES[0],
        STRIDES[1],
        STRIDES[2],
        STRIDES[3],
        STRIDES[4],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
    )
    output = tl.multiple_of(
        O_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    values = tl.multiple_of(
        W_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    _shared(K, P, V, output, values, FULL_META)


def mla_kv_pack_quantize_fp8(
    k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
):
    s, heads, nope = k_nope.shape
    rope = k_pe.shape[-1]
    width = v.shape[-1]
    k_out = torch.empty(
        (s, heads, nope + rope), device=k_nope.device, dtype=fp8_dtype
    )
    v_out = torch.empty((s, heads, width), device=v.device, dtype=fp8_dtype)
    rows = s * heads
    if rows == 0 or (nope + rope == 0 and width == 0):
        return (k_out, v_out)
    lane_width = (
        1 << (max(nope, rope, width) - 1).bit_length()
        if max(nope, rope, width)
        else 0
    )
    block_rows = min(
        32,
        1 << (heads - 1).bit_length() if heads else 0,
        max(1, 4096 // lane_width),
    )
    if k_nope.is_contiguous() and k_pe.is_contiguous() and v.is_contiguous():
        layout = None
    else:
        k_strides = k_nope.stride()
        v_strides = v.stride()
        layout = (
            k_strides[0],
            k_strides[1],
            k_pe.stride(0),
            v_strides[0],
            v_strides[1],
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
        block_rows,
        128 if math.copysign(1.0, k_scale_inv) < 0 else 0,
        128 if math.copysign(1.0, v_scale_inv) < 0 else 0,
    )
    if rows:
        k_address = k_out.data_ptr()
        v_address = v_out.data_ptr()
        assert k_address % 16 == 0 and v_address % 16 == 0
        _address_shared.run(
            k_nope,
            k_pe,
            v,
            k_address,
            v_address,
            meta,
            grid=(s, (heads + block_rows - 1) // block_rows),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (k_out, v_out)


__all__ = ["mla_kv_pack_quantize_fp8"]
