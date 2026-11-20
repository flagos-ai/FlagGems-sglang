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
from triton.language import float32 as fp32


@triton.jit
def _baseline_rope(X, F, P, Out, META: tl.constexpr):
    H: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SX2: tl.constexpr = META[4]
    SF0: tl.constexpr = META[5]
    SF1: tl.constexpr = META[6]
    SP: tl.constexpr = META[7]
    FR: tl.constexpr = META[8]
    T: tl.constexpr = META[9]
    INV: tl.constexpr = META[10]
    B: tl.constexpr = META[11]
    NB: tl.constexpr = META[12]
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid // NB
    i = pid % NB * B + tl.arange(0, B)
    h = i // (D // 2)
    d = i % (D // 2)
    valid = i < H * D // 2
    row = token
    if P is not None:
        if not WIDE and FR <= 2147483647 and SF0 <= 2147483647:
            row = tl.load(P + token * SP).to(tl.int32)
            row = row + tl.where(row < 0, FR, 0)
        else:
            row = tl.load(P + token * SP).to(tl.int64)
            row = tl.where(row < 0, row + FR, row)
    base = token * SX0 + h * SX1 + 2 * d * SX2
    a = tl.load(X + base, valid, 0).to(tl.float32)
    b = tl.load(X + base + SX2, valid, 0).to(tl.float32)
    c = tl.load(F + row * SF0 + d * SF1, valid, 0)
    s = tl.load(F + row * SF0 + d * SF1 + 1, valid, 0)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if Out is None:
        tl.store(X + oi, real, valid, cache_modifier=".cs")
        tl.store(X + oi + 1, imag, valid, cache_modifier=".cs")
    else:
        tl.store(Out + oi, real, valid, cache_modifier=".cs")
        tl.store(Out + oi + 1, imag, valid, cache_modifier=".cs")


@triton.jit
def _baseline_rope_typed_fallback_frequency(
    X, F: "*fp32", P, Out, META: tl.constexpr
):
    _baseline_rope(X, F, P, Out, META)


@triton.jit
def _pair_bits(X, F, P, Out, META: tl.constexpr):
    H: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SX2: tl.constexpr = META[4]
    SF0: tl.constexpr = META[5]
    SF1: tl.constexpr = META[6]
    SP: tl.constexpr = META[7]
    FR: tl.constexpr = META[8]
    T: tl.constexpr = META[9]
    INV: tl.constexpr = META[10]
    B: tl.constexpr = META[11]
    NB: tl.constexpr = META[12]
    PACKX: tl.constexpr = META[13]
    PACKF: tl.constexpr = META[14]
    BF16: tl.constexpr = META[15]
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid // NB
    i = pid % NB * B + tl.arange(0, B)
    h = i // (D // 2)
    d = i % (D // 2)
    valid = i < H * D // 2
    row = token
    if P is not None:
        if not WIDE and FR <= 2147483647 and SF0 <= 2147483647:
            row = tl.load(P + token * SP).to(tl.int32)
            row = row + tl.where(row < 0, FR, 0)
        else:
            row = tl.load(P + token * SP).to(tl.int64)
            row = tl.where(row < 0, row + FR, row)
    base = token * SX0 + h * SX1 + 2 * d * SX2
    if PACKX:
        bits = tl.load(X.to(tl.pointer_type(tl.uint32)) + base // 2, valid, 0)
        if BF16:
            a = ((bits & 65535) << 16).to(tl.float32, bitcast=True)
            b = (bits & 4294901760).to(tl.float32, bitcast=True)
        else:
            a = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            b = (
                (bits >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
    else:
        a = tl.load(X + base, valid, 0).to(tl.float32)
        b = tl.load(X + base + SX2, valid, 0).to(tl.float32)
    if PACKF:
        fb = tl.load(
            F.to(tl.pointer_type(tl.uint64))
            + row * (SF0 // 2)
            + d * (SF1 // 2),
            valid,
            0,
        )
        c = fb.to(tl.uint32).to(tl.float32, bitcast=True)
        s = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    else:
        c = tl.load(F + row * SF0 + d * SF1, valid, 0)
        s = tl.load(F + row * SF0 + d * SF1 + 1, valid, 0)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if PACKX:
        if BF16:
            rb = real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ib = imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            rb = real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ib = imag.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        packed = rb | ib << 16
        if Out is None:
            tl.store(
                X.to(tl.pointer_type(tl.uint32)) + oi // 2,
                packed,
                valid,
                cache_modifier=".cs",
            )
        else:
            tl.store(
                Out.to(tl.pointer_type(tl.uint32)) + oi // 2,
                packed,
                valid,
                cache_modifier=".cs",
            )
    elif Out is None:
        tl.store(X + oi, real, valid, cache_modifier=".cs")
        tl.store(X + oi + 1, imag, valid, cache_modifier=".cs")
    else:
        tl.store(Out + oi, real, valid, cache_modifier=".cs")
        tl.store(Out + oi + 1, imag, valid, cache_modifier=".cs")


def _original_apply(x, freqs_cis, positions, inverse, n):
    if n == 0:
        return x
    shape = x.shape
    h = shape[1] if len(shape) == 3 else 1
    d = shape[-1]
    strides = x.stride()
    dtype = x.dtype
    packed = (
        n > 0
        and dtype in (torch.bfloat16, torch.float16)
        and strides[-1] == 1
        and x.storage_offset() % 2 == 0
        and all(s % 2 == 0 for s in strides[:-1])
    )
    fstrides = freqs_cis.stride()
    f = (
        freqs_cis.view(torch.float32)
        if freqs_cis.dtype == torch.complex64 and fstrides[-1] == 1
        else torch.view_as_real(freqs_cis)
    )
    meta = (
        h,
        d,
        strides[0],
        strides[1] if len(shape) == 3 else 0,
        strides[-1],
        fstrides[0] * 2,
        fstrides[1] * 2,
        positions.stride(0) if positions is not None else 0,
        freqs_cis.shape[0],
        shape[0],
    )
    out = (
        None
        if x.is_contiguous()
        else torch.empty(shape, device=x.device, dtype=dtype)
    )
    block = 512 if packed and shape[0] <= 4096 and h * d // 2 >= 512 else 256
    nb = (h * d // 2 + block - 1) // block
    if packed:
        _pair_bits.run(
            x,
            f,
            positions,
            out,
            meta
            + (
                inverse,
                block,
                nb,
                True,
                block == 256 and freqs_cis.dtype == torch.complex64,
                dtype == torch.bfloat16,
            ),
            grid=(shape[0] * nb,),
            warmup=False,
            enable_fp_fusion=False,
        )
    else:
        _baseline_rope_typed_fallback_frequency.run(
            x,
            f,
            positions,
            out,
            meta + (inverse, block, nb),
            grid=(shape[0] * nb,),
            warmup=False,
            enable_fp_fusion=False,
        )
    return x if out is None else out


@triton.jit
def _dense_baseline_rope(X, F, META: tl.constexpr):
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = META[0]
    T: tl.constexpr = META[0]
    INV: tl.constexpr = META[3]
    B: tl.constexpr = 256
    NB: tl.constexpr = (H * D // 2 + B - 1) // B
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid // NB
    i = pid % NB * B + tl.arange(0, B)
    h = i // (D // 2)
    d = i % (D // 2)
    valid = i < H * D // 2
    row = token
    if None is not None:
        row = tl.load(None + token * SP).to(tl.int64)
        row = tl.where(row < 0, row + FR, row)
    base = token * SX0 + h * SX1 + 2 * d * SX2
    a = tl.load(X + base, valid, 0).to(tl.float32)
    b = tl.load(X + base + SX2, valid, 0).to(tl.float32)
    if D == 64 and H > 1:
        producer = tl.load(F + row * 64 + i % 64, i // 64 * 64 < H * 32, 0)
        producer_bits = producer.to(tl.int32, bitcast=True)
        c = _frequency_shuffle(producer_bits, (2 * d).to(tl.int32)).to(
            tl.float32, bitcast=True
        )
        s = _frequency_shuffle(producer_bits, (2 * d + 1).to(tl.int32)).to(
            tl.float32, bitcast=True
        )
    else:
        c = tl.load(F + row * SF0 + d * SF1, valid, 0)
        s = tl.load(F + row * SF0 + d * SF1 + 1, valid, 0)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if None is None:
        tl.store(X + oi, real, valid)
        tl.store(X + oi + 1, imag, valid)
    else:
        tl.store(None + oi, real, valid)
        tl.store(None + oi + 1, imag, valid)


@tl.core.extern
def _frequency_shuffle(value, lane_index, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [value, lane_index],
        {(tl.int32, tl.int32): ("llvm.bi.slb.shfl.idx.b32.i32", tl.int32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _dense_pair_bits(X, F, META: tl.constexpr):
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = META[0]
    T: tl.constexpr = META[0]
    INV: tl.constexpr = META[3]
    B: tl.constexpr = 512 if T <= 4096 and H * D // 2 >= 512 else 256
    NB: tl.constexpr = (H * D // 2 + B - 1) // B
    PACKX: tl.constexpr = True
    PACKF: tl.constexpr = False
    BF16: tl.constexpr = X.dtype.element_ty == tl.bfloat16
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid // NB
    i = pid % NB * B + tl.arange(0, B)
    if D == 64 and B == 256:
        i = i & -81 | (i & 16) << 2 | (i & 64) >> 2
    d = i % (D // 2)
    valid = i < H * D // 2
    row = token
    if None is not None:
        row = tl.load(None + token * SP).to(tl.int64)
        row = tl.where(row < 0, row + FR, row)
    base = token * H * D + 2 * i
    if PACKX:
        bits = tl.load(X.to(tl.pointer_type(tl.uint32)) + base // 2, valid, 0)
        if BF16:
            a = ((bits & 65535) << 16).to(tl.float32, bitcast=True)
            b = (bits & 4294901760).to(tl.float32, bitcast=True)
        else:
            a = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            b = (
                (bits >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
    else:
        a = tl.load(X + base, valid, 0).to(tl.float32)
        b = tl.load(X + base + SX2, valid, 0).to(tl.float32)
    if PACKF:
        fb = tl.load(
            F.to(tl.pointer_type(tl.uint64))
            + row * (SF0 // 2)
            + d * (SF1 // 2),
            valid,
            0,
        )
        c = fb.to(tl.uint32).to(tl.float32, bitcast=True)
        s = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    elif D == 64 and B == 256:
        lanes = tl.arange(0, B)
        producer = tl.load(
            F + row * 64 + (i & 16) * 2 + lanes % 32, lanes % 64 < 32, 0
        )
        producer_bits = producer.to(tl.int32, bitcast=True)
        c = _frequency_shuffle(producer_bits, (2 * (d % 16)).to(tl.int32)).to(
            tl.float32, bitcast=True
        )
        s = _frequency_shuffle(
            producer_bits, (2 * (d % 16) + 1).to(tl.int32)
        ).to(tl.float32, bitcast=True)
    else:
        c = tl.load(F + row * SF0 + d * SF1, valid, 0)
        s = tl.load(F + row * SF0 + d * SF1 + 1, valid, 0)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if PACKX:
        if BF16:
            rb = real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ib = imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            rb = real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ib = imag.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        packed = rb | ib << 16
        if None is None:
            tl.store(
                X.to(tl.pointer_type(tl.uint32)) + oi // 2,
                packed,
                valid,
                cache_modifier=".cs",
            )
        else:
            tl.store(
                None.to(tl.pointer_type(tl.uint32)) + oi // 2, packed, valid
            )
    elif None is None:
        tl.store(X + oi, real, valid)
        tl.store(X + oi + 1, imag, valid)
    else:
        tl.store(None + oi, real, valid)
        tl.store(None + oi + 1, imag, valid)


@triton.jit
def _dense_baseline_rope_typed_frequency(X, F: "*fp32", META: tl.constexpr):
    _dense_baseline_rope(X, F, META)


@triton.jit
def _dense_pair_bits_typed_frequency(X, F: "*fp32", META: tl.constexpr):
    _dense_pair_bits(X, F, META)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    n = x.numel()
    if (
        n > 0
        and positions is None
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
    ):
        shape = x.shape
        t = shape[0]
        h = shape[1] if len(shape) == 3 else 1
        d = shape[-1]
        if (
            d % 2 == 0
            and freqs_cis.shape[-1] * 2 == d
            and ((t + 16) * h * d + 2048 <= 2147483647)
        ):
            f = (
                freqs_cis.view(torch.float32)
                if freqs_cis.dtype == torch.complex64
                else torch.view_as_real(freqs_cis)
            )
            meta = (t, h, d, inverse)
            if x.dtype in (torch.bfloat16, torch.float16) and (
                x.storage_offset() % 2 == 0
            ):
                block = 512 if t <= 4096 and h * d // 2 >= 512 else 256
                nb = (h * d // 2 + block - 1) // block
                _dense_pair_bits_typed_frequency.run(
                    x,
                    f,
                    meta,
                    grid=(t * nb,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
                return x
            nb = (h * d // 2 + 255) // 256
            _dense_baseline_rope_typed_frequency.run(
                x,
                f,
                meta,
                grid=(t * nb,),
                warmup=False,
                enable_fp_fusion=False,
            )
            return x
    return _original_apply(x, freqs_cis, positions, inverse, n)


__all__ = ["apply_rotary_emb"]
