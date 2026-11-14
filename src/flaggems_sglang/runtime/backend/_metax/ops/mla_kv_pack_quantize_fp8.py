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
    magic_bits = (
        exponent + (exponent >> 23 - M) + ((23 - M << 23) - (128 - BIAS << M))
    )
    code = (
        mag.to(tl.float32, bitcast=True)
        + magic_bits.to(tl.float32, bitcast=True)
    ).to(tl.uint32, bitcast=True)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.uint8)


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
        or (max(KS, KH, PS, VS, VH, H, N + PE, D) > 2147483647)
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
        _encode(kval.to(tl.float32) * K_SCALE, E4, kval, K_SIGN),
        (head[:, None] < H) & (n[None, :] < N),
    )
    tl.store(
        kout + rows[:, None] * (N + PE) + N + pe[None, :],
        _encode(pval.to(tl.float32) * K_SCALE, E4, pval, K_SIGN)[None, :],
        (head[:, None] < H) & (pe[None, :] < PE),
    )
    vval = tl.load(
        V + token * VS + head[:, None] * VH + d[None, :],
        (head[:, None] < H) & (d[None, :] < D),
        other=0,
    )
    tl.store(
        vout + rows[:, None] * D + d[None, :],
        _encode(vval.to(tl.float32) * V_SCALE, E4, vval, V_SIGN),
        (head[:, None] < H) & (d[None, :] < D),
    )


def _compact_wide_layout(tensor):
    if tensor.is_contiguous():
        return tensor
    span = sum(
        (
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride())
        )
    )
    return tensor.contiguous() if span > 2147483647 else tensor


@triton.jit
def _shared_compact(K, P, V, K_OUT, W, COMPACT_META: tl.constexpr):
    LAYOUT: tl.constexpr = COMPACT_META[5]
    STRIDES: tl.constexpr = (
        (
            COMPACT_META[1] * COMPACT_META[2],
            COMPACT_META[2],
            COMPACT_META[3],
            COMPACT_META[1] * COMPACT_META[4],
            COMPACT_META[4],
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
        COMPACT_META[11],
    )
    _shared(K, P, V, K_OUT, W, META)


def mla_kv_pack_quantize_fp8(
    k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
):
    if k_nope.is_contiguous() and k_pe.is_contiguous() and v.is_contiguous():
        layout = None
    else:
        k_nope = _compact_wide_layout(k_nope)
        k_pe = _compact_wide_layout(k_pe)
        v = _compact_wide_layout(v)
        layout = (
            k_nope.stride(0),
            k_nope.stride(1),
            k_pe.stride(0),
            v.stride(0),
            v.stride(1),
        )
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
    lane_width = triton.next_power_of_2(max(nope, rope, width))
    block_rows = min(
        32, triton.next_power_of_2(heads), max(1, 4096 // lane_width)
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
        _shared_compact.run(
            k_nope,
            k_pe,
            v,
            k_out,
            v_out,
            meta,
            grid=(s, triton.cdiv(heads, block_rows)),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (k_out, v_out)


__all__ = ["mla_kv_pack_quantize_fp8"]
