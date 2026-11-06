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

_GRID_CAP = 65535


@triton.jit
def _e2m1_bits(v, exp):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    key = mag - (exp << 23)
    low = 1 + (key > -4194304).to(tl.int32)
    high = tl.minimum((key + 2097151 >> 22) + 2, 7)
    idx = tl.where(key < 0, low, high)
    idx = tl.where(mag == 2139095040, 7, idx)
    encoded = idx | raw >> 28 & 8
    return tl.where((key > -16777216) & (mag <= 2139095040), encoded, 0)


@triton.jit
def _quantize_packed_scale(X, Y, S, CONFIG: tl.constexpr):
    N: tl.constexpr = CONFIG >> 1
    HIP: tl.constexpr = CONFIG & 1 != 0
    BASE_ROWS: tl.constexpr = (
        1 if N <= 65535 else triton.next_power_of_2(triton.cdiv(N, 65535))
    )
    MIN_ROWS: tl.constexpr = 8 if HIP and N >= 4096 else 4 if N >= 4096 else 1
    ROWS: tl.constexpr = BASE_ROWS if BASE_ROWS > MIN_ROWS else MIN_ROWS
    GPP: tl.constexpr = ROWS * 4
    TAIL: tl.constexpr = N % ROWS != 0
    CG: tl.constexpr = HIP and N >= 4096
    WIDE: tl.constexpr = N > 16777216
    num_groups: tl.constexpr = N * 4
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    groups = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, 32)
    offsets = groups[:, None] * 32 + lanes[None, :]
    if TAIL:
        x = tl.load(
            X + offsets,
            groups[:, None] < num_groups,
            0,
            cache_modifier=".cg" if CG else "",
        ).to(tl.float32)
    else:
        x = tl.load(X + offsets, cache_modifier=".cg" if CG else "").to(
            tl.float32
        )
    magnitude = x.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.minimum(tl.max(magnitude, axis=1), 2139095040)
    amax = maximum.to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = (bits >> 23 & 255) + (bits & 8388607 != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    code = _e2m1_bits(x, exp[:, None])
    even, odd = tl.split(tl.reshape(code, (GPP, 16, 2)))
    packed = even | odd << 4
    shifts = tl.arange(0, 4) * 8
    exp_rows = tl.reshape(exp, (GPP // 4, 4))
    packed_exp = tl.sum(exp_rows << shifts[None, :], axis=1)
    rows = pid * (GPP // 4) + tl.arange(0, GPP // 4)
    target = groups[:, None] * 16 + tl.arange(0, 16)[None, :]
    if TAIL:
        tl.store(Y + target, packed.to(tl.int8), groups[:, None] < num_groups)
        tl.store(S + rows, packed_exp, rows < num_groups // 4)
    else:
        tl.store(Y + target, packed.to(tl.int8))
        tl.store(S + rows, packed_exp)


def _rows_per_program(n):
    if n <= _GRID_CAP:
        return 1
    return 1 << ((n + _GRID_CAP - 1) // _GRID_CAP - 1).bit_length()


def quantize_fp4_indexer(x):
    x = x.contiguous()
    n = x.numel() // 128
    y = x.new_empty((n, 64), dtype=torch.int8)
    sf = x.new_empty((n,), dtype=torch.int32)
    if n == 0:
        return (y, sf)
    hip = bool(torch.version.hip)
    dtype = x.dtype
    rows = 8 if hip and n >= 4096 else 4 if n >= 4096 else 1
    if n > _GRID_CAP:
        rows = max(
            rows, 1 << ((n + _GRID_CAP - 1) // _GRID_CAP - 1).bit_length()
        )
    warps = 4 if rows >= 8 else 2 if dtype == torch.float32 or rows >= 4 else 1
    if hip and n >= 4096:
        warps = 2
    if (
        dtype == torch.bfloat16 or (dtype == torch.float16 and n >= 4096)
    ) and (x.data_ptr() % 4 == 0):
        _quantize_cuda_pairs.run(
            x,
            y,
            sf,
            n << 1 | hip,
            num_warps=warps,
            num_stages=1,
            grid=((n + rows - 1) // rows,),
            warmup=False,
        )
    else:
        _quantize_packed_scale.run(
            x,
            y,
            sf,
            n << 1 | hip,
            num_warps=warps,
            num_stages=1,
            grid=((n + rows - 1) // rows,),
            warmup=False,
        )
    return (y, sf)


@triton.jit
def _quantize_cuda_pairs(X, Y, S, CONFIG: tl.constexpr):
    N: tl.constexpr = CONFIG >> 1
    HIP: tl.constexpr = CONFIG & 1 != 0
    BASE_ROWS: tl.constexpr = (
        1 if N <= 65535 else triton.next_power_of_2(triton.cdiv(N, 65535))
    )
    MIN_ROWS: tl.constexpr = 8 if HIP and N >= 4096 else 4 if N >= 4096 else 1
    ROWS: tl.constexpr = BASE_ROWS if BASE_ROWS > MIN_ROWS else MIN_ROWS
    GPP: tl.constexpr = ROWS * 4
    TAIL: tl.constexpr = N % ROWS != 0
    CG: tl.constexpr = HIP and N >= 4096
    WIDE: tl.constexpr = N > 16777216
    num_groups: tl.constexpr = N * 4
    HALF: tl.constexpr = X.dtype.element_ty == tl.float16
    X = X.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    groups = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, 16)
    offsets = groups[:, None] * 16 + lanes[None, :]
    if TAIL:
        words = tl.load(
            X + offsets,
            groups[:, None] < num_groups,
            0,
            cache_modifier=".cg" if CG else "",
        )
    else:
        words = tl.load(X + offsets, cache_modifier=".cg" if CG else "")
    if HALF:
        rawmax = tl.max(tl.maximum(words & 32767, words >> 16 & 32767), axis=1)
        exp = tl.where(
            rawmax >= 31744, 254, tl.maximum((rawmax + 511 >> 10) + 110, 114)
        )
    else:
        even = (words << 16).to(tl.float32, bitcast=True)
        odd = (words & -65536).to(tl.float32, bitcast=True)
        me = even.to(tl.int32, bitcast=True) & 2147483647
        mo = odd.to(tl.int32, bitcast=True) & 2147483647
        maximum = tl.minimum(tl.max(tl.maximum(me, mo), axis=1), 2139095040)
        exp = tl.where(
            maximum >= 2139095040,
            254,
            tl.maximum((maximum + 4194303 >> 23) - 2, 114),
        )
    if HALF:
        packed = _e2m1_full_fp16_pairs(words, exp[:, None])
    else:
        packed = _e2m1_bf16_pairs494(words, exp[:, None])
    shifts = tl.arange(0, 4) * 8
    exp_rows = tl.reshape(exp, (GPP // 4, 4))
    packed_exp = tl.sum(exp_rows << shifts[None, :], axis=1)
    rows = pid * (GPP // 4) + tl.arange(0, GPP // 4)
    if TAIL:
        tl.store(Y + offsets, packed.to(tl.int8), groups[:, None] < num_groups)
        if HIP and N >= 4096 and (ROWS == 8):
            tl.store(
                S + rows[:, None] + lanes[None, :],
                packed_exp[:, None],
                (rows[:, None] < num_groups // 4) & (lanes[None, :] == 0),
            )
        else:
            tl.store(S + rows, packed_exp, rows < num_groups // 4)
    else:
        tl.store(Y + offsets, packed.to(tl.int8))
        if HIP and N >= 4096 and (ROWS == 8):
            tl.store(
                S + rows[:, None] + lanes[None, :],
                packed_exp[:, None],
                lanes[None, :] == 0,
            )
        else:
            tl.store(S + rows, packed_exp)


@triton.jit
def _packed_lshr15_494(value):
    return tl.inline_asm_elementwise(
        "v_pk_lshrrev_b16 $0, 15, $1 op_sel_hi:[0,1]",
        constraints="=v,v",
        args=[value],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _packed_floor_quotient494(value, floor):
    return tl.inline_asm_elementwise(
        "v_pk_max_u16 $0, $1, $2\nv_pk_lshrrev_b16 $0, 6, $0 op_sel_hi:[0,1]",
        constraints="=v,v,v",
        args=[value, floor],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _packed_keep494(nan, nonzero):
    return tl.inline_asm_elementwise(
        "v_pk_sub_i16 $0, $1, $2",
        constraints="=v,v,v",
        args=[nan, nonzero],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1_bf16_pairs494(words, exp):
    raw = words.to(tl.uint32)
    guarded_raw = raw | 2147516416
    base = exp.to(tl.uint32) * 8388736
    biased = guarded_raw - base
    nonzero = _packed_lshr15_494(biased + 16711935)
    low2 = _packed_lshr15_494(biased + 4128831)
    floor = tl.full((), 2147516416, tl.uint32)
    quotient = _packed_floor_quotient494(biased + 2031647, floor)
    code = quotient + nonzero + low2
    special = _packed_lshr15_494(guarded_raw - 2139127680)
    nan = _packed_lshr15_494(guarded_raw - 2139193217)
    keep = _packed_keep494(nan, nonzero)
    code = (code | raw >> 12 & 524296 | special * 7) & keep
    return (code | code >> 12).to(tl.int32, bitcast=True)


@triton.jit
def _e2m1_finite_fp16_pairs(words, exp):
    raw = words.to(tl.uint32)
    base = (exp.to(tl.uint32) - 112) * 67109888
    biased = (raw | 2147516416) - base
    cutoff = tl.where(exp == 114, 100599295, 134154239)
    nonzero = _packed_lshr15_494(biased + cutoff)
    low2 = _packed_lshr15_494(biased + 33489407)
    floor = tl.full((), 2147516416, tl.uint32)
    quotient = _packed_floor_quotient554(biased + 16711935, floor)
    code = quotient + nonzero + low2 ^ 4194368
    code = code | raw >> 12 & nonzero << 3
    return (code | code >> 12).to(tl.int32, bitcast=True)


@triton.jit
def _e2m1_full_fp16_pairs(words, exp):
    raw = words.to(tl.uint32)
    finite = _e2m1_finite_fp16_pairs(words, tl.minimum(exp, 141))
    difference = raw & 2147450879 ^ 2080406528
    biased = difference + 2147450879
    inf_mask = (_packed_lshr15_494(biased) ^ 65537) * 15
    special = raw >> 12 & inf_mask
    packed_special = (special | special >> 12).to(tl.int32, bitcast=True)
    return tl.where(exp == 254, packed_special, finite)


@triton.jit
def _packed_floor_quotient554(value, floor):
    return tl.inline_asm_elementwise(
        "v_pk_max_u16 $0, $1, $2\nv_pk_lshrrev_b16 $0, 9, $0 op_sel_hi:[0,1]",
        constraints="=v,v,v",
        args=[value, floor],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


__all__ = ["quantize_fp4_indexer"]
