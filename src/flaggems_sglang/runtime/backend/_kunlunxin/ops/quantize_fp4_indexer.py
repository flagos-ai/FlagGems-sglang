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
def _max2(a0, a1, b0, b1):
    return (tl.maximum(a0, b0), tl.maximum(a1, b1))


@triton.jit
def _max4(a0, a1, a2, a3, b0, b1, b2, b3):
    return (
        tl.maximum(a0, b0),
        tl.maximum(a1, b1),
        tl.maximum(a2, b2),
        tl.maximum(a3, b3),
    )


_GRID_CAP = 65535


@triton.jit
def _e2m1_bits(v, exp):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    key = mag - (exp << 23)
    low = (key > -16777216).to(tl.int32) + (key > -4194304).to(tl.int32)
    high = tl.minimum((key + 2097151 >> 22) + 2, 7)
    idx = tl.where(key < 0, low, high)
    idx = tl.where(mag == 2139095040, 7, idx)
    idx = tl.where(mag > 2139095040, 0, idx)
    return idx | tl.where(idx != 0, raw >> 28 & 8, 0)


@triton.jit
def _quantize_fp4_indexer_kernel(
    x_ptr,
    x_fp4_ptr,
    x_sf_ptr,
    num_groups,
    GPP: tl.constexpr,
    TAIL: tl.constexpr,
):
    x_sf_ptr = x_sf_ptr.to(tl.pointer_type(tl.int8))
    pid = tl.program_id(0)
    groups = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, 16)
    even_off = groups[:, None] * 32 + lanes[None, :] * 2
    odd_off = even_off + 1
    if TAIL:
        mask = groups < num_groups
        xe = tl.load(x_ptr + even_off, mask=mask[:, None], other=0.0).to(
            tl.float32
        )
        xo = tl.load(x_ptr + odd_off, mask=mask[:, None], other=0.0).to(
            tl.float32
        )
    else:
        xe = tl.load(x_ptr + even_off).to(tl.float32)
        xo = tl.load(x_ptr + odd_off).to(tl.float32)
    me = xe.to(tl.int32, bitcast=True) & 2147483647
    mo = xo.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(tl.maximum(me, mo), axis=1)
    amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits >> 23 & 255
    exp = exp + (bits & 8388607 != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    even_code = _e2m1_bits(xe, exp[:, None])
    odd_code = _e2m1_bits(xo, exp[:, None])
    packed = even_code & 15 | (odd_code & 15) << 4
    out_off = groups[:, None] * 16 + lanes[None, :]
    if TAIL:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8), mask=mask[:, None])
        tl.store(x_sf_ptr + groups, exp.to(tl.int8), mask=mask)
    else:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8))
        tl.store(x_sf_ptr + groups, exp.to(tl.int8))


def _rows_per_program(n):
    if n <= _GRID_CAP:
        return 1
    return 1 << ((n + _GRID_CAP - 1) // _GRID_CAP - 1).bit_length()


@triton.jit
def _quantize_half_pairs(
    x_ptr,
    x_fp4_ptr,
    x_sf_ptr,
    num_groups,
    GPP: tl.constexpr,
    TAIL: tl.constexpr,
    HALF: tl.constexpr,
):
    x_ptr = x_ptr.to(tl.pointer_type(tl.int32))
    x_sf_ptr = x_sf_ptr.to(tl.pointer_type(tl.int8))
    pid = tl.program_id(0)
    groups = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, 16)
    offsets = groups[:, None] * 16 + lanes[None, :]
    if TAIL:
        mask = groups < num_groups
        words = tl.load(x_ptr + offsets, mask=mask[:, None], other=0)
    else:
        words = tl.load(x_ptr + offsets)
    if HALF:
        xe = words.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        xo = (
            (words >> 16)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        xe = (words << 16).to(tl.float32, bitcast=True)
        xo = (words & -65536).to(tl.float32, bitcast=True)
    me = xe.to(tl.int32, bitcast=True) & 2147483647
    mo = xo.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(tl.maximum(me, mo), axis=1)
    amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits >> 23 & 255
    exp = exp + (bits & 8388607 != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    even_code = _e2m1_bits(xe, exp[:, None])
    odd_code = _e2m1_bits(xo, exp[:, None])
    packed = even_code & 15 | (odd_code & 15) << 4
    out_off = groups[:, None] * 16 + lanes[None, :]
    if TAIL:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8), mask=mask[:, None])
        tl.store(x_sf_ptr + groups, exp.to(tl.int8), mask=mask)
    else:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8))
        tl.store(x_sf_ptr + groups, exp.to(tl.int8))


def quantize_fp4_indexer(x):
    x = x.contiguous()
    n = x.numel() // 128
    if x.ndim != 2:
        x = x.view(-1, 128)
    if (
        128 <= n < 4096
        and n % 128 != 0
        and (x.dtype in (torch.bfloat16, torch.float16))
        and (x.storage_offset() % 2 == 0)
        and (x.data_ptr() % 4 == 0)
    ):
        padded = (n + 128 - 1) // 128 * 128
        y = x.new_empty((padded, 64), dtype=torch.int8)
        sf = x.new_empty((padded,), dtype=torch.int32)
        if x.data_ptr() % 8 == 0:
            _quantize_padded_quad.run(
                x,
                y,
                sf,
                n,
                groups_per_cluster=64,
                grid=(padded // 64,),
                warmup=False,
            )
        else:
            _quantize_padded_rows.run(
                x,
                y,
                sf,
                n,
                R=128,
                V=64,
                HALF=x.dtype == torch.float16,
                F32=False,
                groups_per_cluster=64,
                grid=(padded // 128,),
                warmup=False,
            )
        return (y[:n], sf[:n])
    y = x.new_empty((n, 64), dtype=torch.int8)
    sf = x.new_empty((n,), dtype=torch.int32)
    if n == 0:
        return (y, sf)
    if n <= 1048576:
        return _dispatch_segment(x, y, sf, n)
    x = x.view(-1, 128)
    for start in range(0, n, 1048576):
        stop = min(start + 1048576, n)
        _dispatch_segment(
            x[start:stop], y[start:stop], sf[start:stop], stop - start
        )
    return (y, sf)


def _dispatch_segment(x, y, sf, n):
    if (
        (
            x.dtype == torch.bfloat16
            or x.dtype == torch.float16
            or x.dtype == torch.float32
        )
        and x.storage_offset() % 2 == 0
        and (x.data_ptr() % 4 == 0)
    ):
        vector = (
            64
            if x.dtype == torch.float32
            or n <= (16384 if x.dtype == torch.float16 else 131072)
            else 32
        )
        chunks = n * 64 // vector
        chunk_rows = 128
        if chunks % chunk_rows == 0:
            if x.dtype == torch.float32 and x.data_ptr() % 8 == 0:
                scales = sf.view(torch.int16) if vector == 32 else sf
                _quantize_multigroup_u64.run(
                    x,
                    y,
                    scales,
                    R=chunk_rows,
                    V=vector,
                    HALF=x.dtype == torch.float16,
                    F32=x.dtype == torch.float32,
                    groups_per_cluster=64,
                    grid=((chunks + chunk_rows - 1) // chunk_rows,),
                    warmup=False,
                )
            elif (
                x.dtype in (torch.bfloat16, torch.float16)
                and x.data_ptr() % 8 == 0
            ):
                if (
                    vector == 64
                    or (chunks + chunk_rows - 1) // chunk_rows % 2 == 0
                ):
                    _quantize_multigroup_quadwide.run(
                        x,
                        y,
                        sf,
                        groups_per_cluster=64,
                        grid=(n // chunk_rows,),
                        warmup=False,
                    )
                else:
                    scales = sf.view(torch.int16) if vector == 32 else sf
                    _quantize_multigroup_quads.run(
                        x,
                        y,
                        scales,
                        groups_per_cluster=64,
                        grid=((chunks + chunk_rows - 1) // chunk_rows,),
                        warmup=False,
                    )
            else:
                scales = sf.view(torch.int16) if vector == 32 else sf
                _quantize_multigroup.run(
                    x.view(torch.int32),
                    y,
                    scales,
                    R=chunk_rows,
                    V=vector,
                    HALF=x.dtype == torch.float16,
                    F32=x.dtype == torch.float32,
                    groups_per_cluster=64,
                    grid=((chunks + chunk_rows - 1) // chunk_rows,),
                    warmup=False,
                )
            return (y, sf)
        if (
            n >= (256 if x.dtype == torch.bfloat16 else 4096)
            and chunks >= chunk_rows
        ):
            full_chunks = chunks // chunk_rows * chunk_rows
            full_tokens = full_chunks * vector // 64
            if x.dtype == torch.float32 and x.data_ptr() % 8 == 0:
                scales = sf.view(torch.int16) if vector == 32 else sf
                _quantize_multigroup_u64.run(
                    x,
                    y,
                    scales,
                    R=chunk_rows,
                    V=vector,
                    HALF=x.dtype == torch.float16,
                    F32=x.dtype == torch.float32,
                    groups_per_cluster=64,
                    grid=(full_chunks // chunk_rows,),
                    warmup=False,
                )
            elif (
                x.dtype in (torch.bfloat16, torch.float16)
                and x.data_ptr() % 8 == 0
            ):
                if vector == 64 or full_chunks // chunk_rows % 2 == 0:
                    _quantize_multigroup_quadwide.run(
                        x,
                        y,
                        sf,
                        groups_per_cluster=64,
                        grid=(full_tokens // chunk_rows,),
                        warmup=False,
                    )
                else:
                    scales = sf.view(torch.int16) if vector == 32 else sf
                    _quantize_multigroup_quads.run(
                        x,
                        y,
                        scales,
                        groups_per_cluster=64,
                        grid=(full_chunks // chunk_rows,),
                        warmup=False,
                    )
            else:
                scales = sf.view(torch.int16) if vector == 32 else sf
                _quantize_multigroup.run(
                    x.view(torch.int32),
                    y,
                    scales,
                    R=chunk_rows,
                    V=vector,
                    HALF=x.dtype == torch.float16,
                    F32=x.dtype == torch.float32,
                    groups_per_cluster=64,
                    grid=(full_chunks // chunk_rows,),
                    warmup=False,
                )
            x = x.view(-1, 128)
            tail_tokens = n - full_tokens
            if x.dtype == torch.float32:
                _launch_narrow(
                    x[full_tokens:],
                    y[full_tokens:],
                    sf[full_tokens:],
                    tail_tokens,
                    16,
                    False,
                )
            else:
                _quantize_half_pairs.run(
                    x[full_tokens:],
                    y[full_tokens:],
                    sf[full_tokens:],
                    tail_tokens * 4,
                    GPP=64,
                    TAIL=tail_tokens % 16 != 0,
                    groups_per_cluster=64,
                    HALF=x.dtype == torch.float16,
                    grid=((tail_tokens + 16 - 1) // 16,),
                    warmup=False,
                )
            return (y, sf)
    rows = max(_rows_per_program(n), 16)
    paired = (
        (x.dtype == torch.bfloat16 or (x.dtype == torch.float16 and n >= 4096))
        and x.storage_offset() % 2 == 0
        and (x.data_ptr() % 4 == 0)
    )
    if n >= 256 and n % rows:
        x = x.view(-1, 128)
        full = n // rows * rows
        _launch_narrow(x, y, sf, full, rows, paired)
        _launch_narrow(x[full:], y[full:], sf[full:], n - full, rows, paired)
    else:
        _launch_narrow(x, y, sf, n, rows, paired)
    return (y, sf)


def _launch_narrow(x, y, sf, n, rows, paired):
    if paired:
        _quantize_half_pairs.run(
            x,
            y,
            sf,
            n * 4,
            GPP=rows * 4,
            TAIL=n % rows != 0,
            groups_per_cluster=64,
            HALF=x.dtype == torch.float16,
            grid=((n + rows - 1) // rows,),
            warmup=False,
        )
    else:
        _quantize_fp4_indexer_kernel.run(
            x,
            y,
            sf,
            n * 4,
            GPP=rows * 4,
            TAIL=n % rows != 0,
            groups_per_cluster=64,
            grid=((n + rows - 1) // rows,),
            warmup=False,
        )


@triton.jit
def _quantize_multigroup(
    X,
    Y,
    S,
    R: tl.constexpr,
    V: tl.constexpr,
    HALF: tl.constexpr,
    F32: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, V)
    offsets = rows[:, None] * V + lanes[None, :]
    if F32:
        even = tl.load(X + offsets * 2).to(tl.float32, bitcast=True)
        odd = tl.load(X + offsets * 2 + 1).to(tl.float32, bitcast=True)
        me = even.to(tl.int32, bitcast=True) & 2147483647
        mo = odd.to(tl.int32, bitcast=True) & 2147483647
    elif HALF:
        words = tl.load(X + offsets)
        even = words & 65535
        odd = words >> 16 & 65535
        me = even & 32767
        mo = odd & 32767
    else:
        words = tl.load(X + offsets)
        even = (words << 16).to(tl.float32, bitcast=True)
        odd = (words & -65536).to(tl.float32, bitcast=True)
        me = (words & 32767) << 16
        mo = words & 2147418112
    magnitude = tl.maximum(me, mo)
    exp_word = tl.full((R,), 0, tl.int32)
    m0 = tl.where((lanes // 16 == 0)[None, :], magnitude, 0)
    m1 = tl.where((lanes // 16 == 1)[None, :], magnitude, 0)
    if V == 64:
        m2 = tl.where((lanes // 16 == 2)[None, :], magnitude, 0)
        m3 = tl.where((lanes // 16 == 3)[None, :], magnitude, 0)
        maximum0, maximum1, maximum2, maximum3 = tl.reduce(
            (m0, m1, m2, m3), 1, _max4
        )
    else:
        maximum0, maximum1 = tl.reduce((m0, m1), 1, _max2)
    maximum = maximum0
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0 * 8
    maximum = maximum1
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 1 * 8
    if V == 64:
        maximum = maximum2
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 2 * 8
    if V == 64:
        maximum = maximum3
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 3 * 8
    exp_lanes = exp_word[:, None] >> (lanes // 16)[None, :] * 8 & 255
    if HALF:
        packed = _e2m1_fp16(even, exp_lanes) | _e2m1_fp16(odd, exp_lanes) << 4
    else:
        packed = _e2m1_bits(even, exp_lanes) | _e2m1_bits(odd, exp_lanes) << 4
    tl.store(Y + offsets, packed.to(tl.int8))
    tl.store(S + rows, exp_word)


@triton.jit
def _e2m1_fp16(bits, exp):
    mag = bits & 32767
    key = mag - (exp - 112 << 10)
    low = (key > -2048).to(tl.int32) + (key > -512).to(tl.int32)
    high = tl.minimum((key + 255 >> 9) + 2, 7)
    idx = tl.where(key < 0, low, high)
    idx = tl.where(mag < 1024, ((mag > 512) & (exp == 114)).to(tl.int32), idx)
    idx = tl.where(mag == 31744, 7, idx)
    idx = tl.where(mag > 31744, 0, idx)
    return idx | tl.where(idx != 0, bits >> 12 & 8, 0)


@triton.jit
def _quantize_padded_rows(
    X,
    Y,
    S,
    chunks,
    R: tl.constexpr,
    V: tl.constexpr,
    HALF: tl.constexpr,
    F32: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.int32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, V)
    offsets = rows[:, None] * V + lanes[None, :]
    input_offsets = tl.minimum(rows, chunks - 1)[:, None] * V + lanes[None, :]
    if F32:
        even = tl.load(X + input_offsets * 2).to(tl.float32, bitcast=True)
        odd = tl.load(X + input_offsets * 2 + 1).to(tl.float32, bitcast=True)
        me = even.to(tl.int32, bitcast=True) & 2147483647
        mo = odd.to(tl.int32, bitcast=True) & 2147483647
    elif HALF:
        words = tl.load(X + input_offsets)
        even = words & 65535
        odd = words >> 16 & 65535
        me = even & 32767
        mo = odd & 32767
    else:
        words = tl.load(X + input_offsets)
        even = (words << 16).to(tl.float32, bitcast=True)
        odd = (words & -65536).to(tl.float32, bitcast=True)
        me = (words & 32767) << 16
        mo = words & 2147418112
    magnitude = tl.maximum(me, mo)
    exp_word = tl.full((R,), 0, tl.int32)
    m0 = tl.where((lanes // 16 == 0)[None, :], magnitude, 0)
    m1 = tl.where((lanes // 16 == 1)[None, :], magnitude, 0)
    if V == 64:
        m2 = tl.where((lanes // 16 == 2)[None, :], magnitude, 0)
        m3 = tl.where((lanes // 16 == 3)[None, :], magnitude, 0)
        maximum0, maximum1, maximum2, maximum3 = tl.reduce(
            (m0, m1, m2, m3), 1, _max4
        )
    else:
        maximum0, maximum1 = tl.reduce((m0, m1), 1, _max2)
    maximum = maximum0
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0 * 8
    maximum = maximum1
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 1 * 8
    if V == 64:
        maximum = maximum2
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 2 * 8
    if V == 64:
        maximum = maximum3
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 3 * 8
    exp_lanes = exp_word[:, None] >> (lanes // 16)[None, :] * 8 & 255
    if HALF:
        packed = _e2m1_fp16(even, exp_lanes) | _e2m1_fp16(odd, exp_lanes) << 4
    else:
        packed = _e2m1_bits(even, exp_lanes) | _e2m1_bits(odd, exp_lanes) << 4
    tl.store(Y + offsets, packed.to(tl.int8))
    tl.store(S + rows, exp_word)


@triton.jit
def _quantize_multigroup_u64(
    X,
    Y,
    S,
    R: tl.constexpr,
    V: tl.constexpr,
    HALF: tl.constexpr,
    F32: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.uint64))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, V)
    offsets = rows[:, None] * V + lanes[None, :]
    if F32:
        words = tl.load(X + offsets)
        even = (words & 4294967295).to(tl.int32).to(tl.float32, bitcast=True)
        odd = (words >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        me = even.to(tl.int32, bitcast=True) & 2147483647
        mo = odd.to(tl.int32, bitcast=True) & 2147483647
    elif HALF:
        words = tl.load(X + offsets)
        even = words & 65535
        odd = words >> 16 & 65535
        me = even & 32767
        mo = odd & 32767
    else:
        words = tl.load(X + offsets)
        even = (words << 16).to(tl.float32, bitcast=True)
        odd = (words & -65536).to(tl.float32, bitcast=True)
        me = (words & 32767) << 16
        mo = words & 2147418112
    magnitude = tl.maximum(me, mo)
    exp_word = tl.full((R,), 0, tl.int32)
    m0 = tl.where((lanes // 16 == 0)[None, :], magnitude, 0)
    m1 = tl.where((lanes // 16 == 1)[None, :], magnitude, 0)
    if V == 64:
        m2 = tl.where((lanes // 16 == 2)[None, :], magnitude, 0)
        m3 = tl.where((lanes // 16 == 3)[None, :], magnitude, 0)
        maximum0, maximum1, maximum2, maximum3 = tl.reduce(
            (m0, m1, m2, m3), 1, _max4
        )
    else:
        maximum0, maximum1 = tl.reduce((m0, m1), 1, _max2)
    maximum = maximum0
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0 * 8
    maximum = maximum1
    if HALF:
        amax = (
            tl.minimum(maximum, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 1 * 8
    if V == 64:
        maximum = maximum2
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 2 * 8
    if V == 64:
        maximum = maximum3
        if HALF:
            amax = (
                tl.minimum(maximum, 31744)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        else:
            amax = tl.minimum(maximum, 2139095040).to(tl.float32, bitcast=True)
        sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
        bits = sf.to(tl.int32, bitcast=True)
        exp = bits + 8388607 >> 23
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        exp_word = exp_word | exp << 3 * 8
    exp_lanes = exp_word[:, None] >> (lanes // 16)[None, :] * 8 & 255
    if HALF:
        packed = _e2m1_fp16(even, exp_lanes) | _e2m1_fp16(odd, exp_lanes) << 4
    else:
        packed = _e2m1_bits(even, exp_lanes) | _e2m1_bits(odd, exp_lanes) << 4
    tl.store(Y + offsets, packed.to(tl.int8))
    tl.store(S + rows, exp_word)


@triton.jit
def _quantize_multigroup_quads(X, Y, S):
    R: tl.constexpr = 128
    HALF: tl.constexpr = X.dtype.element_ty == tl.float16
    Y = Y.to(tl.pointer_type(tl.int16))
    X = X.to(tl.pointer_type(tl.uint64))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, 16)
    offsets = rows[:, None] * 16 + lanes[None, :]
    words = tl.load(X + offsets)
    q0 = (words >> 0 & 65535).to(tl.int32)
    q1 = (words >> 16 & 65535).to(tl.int32)
    q2 = (words >> 32 & 65535).to(tl.int32)
    q3 = (words >> 48 & 65535).to(tl.int32)
    if HALF:
        m0 = q0 & 32767
        m1 = q1 & 32767
        m2 = q2 & 32767
        m3 = q3 & 32767
    else:
        m0 = (q0 & 32767) << 16
        q0 = (q0 << 16).to(tl.float32, bitcast=True)
        m1 = (q1 & 32767) << 16
        q1 = (q1 << 16).to(tl.float32, bitcast=True)
        m2 = (q2 & 32767) << 16
        q2 = (q2 << 16).to(tl.float32, bitcast=True)
        m3 = (q3 & 32767) << 16
        q3 = (q3 << 16).to(tl.float32, bitcast=True)
    magnitude = tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))
    g0 = tl.where((lanes // 8 == 0)[None, :], magnitude, 0)
    g1 = tl.where((lanes // 8 == 1)[None, :], magnitude, 0)
    maximum0, maximum1 = tl.reduce((g0, g1), 1, _max2)
    exp_word = tl.full((R,), 0, tl.int32)
    if HALF:
        amax = (
            tl.minimum(maximum0, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum0, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0
    if HALF:
        amax = (
            tl.minimum(maximum1, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum1, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 8
    exp_lanes = exp_word[:, None] >> (lanes // 8)[None, :] * 8 & 255
    if HALF:
        packed = (
            _e2m1_full_fp16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (
                _e2m1_full_fp16_pairs((words >> 32).to(tl.uint32), exp_lanes)
                & 255
            )
            << 8
        )
    else:
        packed = (
            _e2m1_bf16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (_e2m1_bf16_pairs((words >> 32).to(tl.uint32), exp_lanes) & 255)
            << 8
        )
    tl.store(Y + offsets, packed.to(tl.int16))
    tl.store(S + rows, exp_word.to(tl.int16))


@triton.jit
def _quantize_multigroup_quadwide(X, Y, S):
    R: tl.constexpr = 128
    HALF: tl.constexpr = X.dtype.element_ty == tl.float16
    Y = Y.to(tl.pointer_type(tl.int16))
    X = X.to(tl.pointer_type(tl.uint64))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, 32)
    offsets = rows[:, None] * 32 + lanes[None, :]
    words = tl.load(X + offsets)
    q0 = (words >> 0 & 65535).to(tl.int32)
    q1 = (words >> 16 & 65535).to(tl.int32)
    q2 = (words >> 32 & 65535).to(tl.int32)
    q3 = (words >> 48 & 65535).to(tl.int32)
    if HALF:
        m0 = q0 & 32767
        m1 = q1 & 32767
        m2 = q2 & 32767
        m3 = q3 & 32767
    else:
        m0 = (q0 & 32767) << 16
        q0 = (q0 << 16).to(tl.float32, bitcast=True)
        m1 = (q1 & 32767) << 16
        q1 = (q1 << 16).to(tl.float32, bitcast=True)
        m2 = (q2 & 32767) << 16
        q2 = (q2 << 16).to(tl.float32, bitcast=True)
        m3 = (q3 & 32767) << 16
        q3 = (q3 << 16).to(tl.float32, bitcast=True)
    magnitude = tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))
    g0 = tl.where((lanes // 8 == 0)[None, :], magnitude, 0)
    g1 = tl.where((lanes // 8 == 1)[None, :], magnitude, 0)
    g2 = tl.where((lanes // 8 == 2)[None, :], magnitude, 0)
    g3 = tl.where((lanes // 8 == 3)[None, :], magnitude, 0)
    maximum0, maximum1, maximum2, maximum3 = tl.reduce(
        (g0, g1, g2, g3), 1, _max4
    )
    exp_word = tl.full((R,), 0, tl.int32)
    if HALF:
        amax = (
            tl.minimum(maximum0, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum0, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0
    if HALF:
        amax = (
            tl.minimum(maximum1, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum1, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 8
    if HALF:
        amax = (
            tl.minimum(maximum2, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum2, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 16
    if HALF:
        amax = (
            tl.minimum(maximum3, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum3, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 24
    exp_lanes = exp_word[:, None] >> (lanes // 8)[None, :] * 8 & 255
    if HALF:
        packed = (
            _e2m1_full_fp16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (
                _e2m1_full_fp16_pairs((words >> 32).to(tl.uint32), exp_lanes)
                & 255
            )
            << 8
        )
    else:
        packed = (
            _e2m1_bf16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (_e2m1_bf16_pairs((words >> 32).to(tl.uint32), exp_lanes) & 255)
            << 8
        )
    tl.store(Y + offsets, packed.to(tl.int16))
    tl.store(S + rows, exp_word)


@triton.jit
def _quantize_padded_quad(X, Y, S, chunks):
    R: tl.constexpr = 64
    HALF: tl.constexpr = X.dtype.element_ty == tl.float16
    Y = Y.to(tl.pointer_type(tl.int16))
    X = X.to(tl.pointer_type(tl.uint64))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, 32)
    offsets = rows[:, None] * 32 + lanes[None, :]
    input_offsets = tl.minimum(rows, chunks - 1)[:, None] * 32 + lanes[None, :]
    words = tl.load(X + input_offsets)
    q0 = (words >> 0 & 65535).to(tl.int32)
    q1 = (words >> 16 & 65535).to(tl.int32)
    q2 = (words >> 32 & 65535).to(tl.int32)
    q3 = (words >> 48 & 65535).to(tl.int32)
    if HALF:
        m0 = q0 & 32767
        m1 = q1 & 32767
        m2 = q2 & 32767
        m3 = q3 & 32767
    else:
        m0 = (q0 & 32767) << 16
        q0 = (q0 << 16).to(tl.float32, bitcast=True)
        m1 = (q1 & 32767) << 16
        q1 = (q1 << 16).to(tl.float32, bitcast=True)
        m2 = (q2 & 32767) << 16
        q2 = (q2 << 16).to(tl.float32, bitcast=True)
        m3 = (q3 & 32767) << 16
        q3 = (q3 << 16).to(tl.float32, bitcast=True)
    magnitude = tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))
    g0 = tl.where((lanes // 8 == 0)[None, :], magnitude, 0)
    g1 = tl.where((lanes // 8 == 1)[None, :], magnitude, 0)
    g2 = tl.where((lanes // 8 == 2)[None, :], magnitude, 0)
    g3 = tl.where((lanes // 8 == 3)[None, :], magnitude, 0)
    maximum0, maximum1, maximum2, maximum3 = tl.reduce(
        (g0, g1, g2, g3), 1, _max4
    )
    exp_word = tl.full((R,), 0, tl.int32)
    if HALF:
        amax = (
            tl.minimum(maximum0, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum0, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 0
    if HALF:
        amax = (
            tl.minimum(maximum1, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum1, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 8
    if HALF:
        amax = (
            tl.minimum(maximum2, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum2, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 16
    if HALF:
        amax = (
            tl.minimum(maximum3, 31744)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    else:
        amax = tl.minimum(maximum3, 2139095040).to(tl.float32, bitcast=True)
    sf = tl.maximum(tl.div_rn(amax, 6.0), 0.0001)
    bits = sf.to(tl.int32, bitcast=True)
    exp = bits + 8388607 >> 23
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    exp_word = exp_word | exp << 24
    exp_lanes = exp_word[:, None] >> (lanes // 8)[None, :] * 8 & 255
    if HALF:
        packed = (
            _e2m1_full_fp16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (
                _e2m1_full_fp16_pairs((words >> 32).to(tl.uint32), exp_lanes)
                & 255
            )
            << 8
        )
    else:
        packed = (
            _e2m1_bf16_pairs(words.to(tl.uint32), exp_lanes) & 255
            | (_e2m1_bf16_pairs((words >> 32).to(tl.uint32), exp_lanes) & 255)
            << 8
        )
    tl.store(Y + offsets, packed.to(tl.int16))
    tl.store(S + rows, exp_word)


@triton.jit
def _e2m1_bf16_pairs(words, exp):
    raw = words.to(tl.uint32)
    base = exp.to(tl.uint32) * 8388736
    biased = (raw | 2147516416) - base
    nonzero = biased + 16711935 >> 15 & 65537
    low2 = biased + 4128831 >> 15 & 65537
    positive = biased >> 15 & 65537
    high = biased + 2031647 >> 6 & positive * 511
    code = nonzero + low2 + high
    code = code | raw >> 12 & nonzero << 3
    guarded_raw = raw | 2147516416
    special = guarded_raw - 2139127680 >> 15 & 65537
    nan = guarded_raw - 2139193217 >> 15 & 65537
    code = (code | special * 7) & ~(nan * 65535)
    return (code | code >> 12).to(tl.int32, bitcast=True)


@triton.jit
def _e2m1_finite_fp16_pairs(words, exp):
    raw = words.to(tl.uint32)
    base = (exp.to(tl.uint32) - 112) * 67109888
    biased = (raw | 2147516416) - base
    cutoff = tl.where(exp == 114, 100599295, 134154239)
    nonzero = biased + cutoff >> 15 & 65537
    low2 = biased + 33489407 >> 15 & 65537
    positive = (biased >> 15 & 65537) * 63
    high = biased + 16711935 >> 9 & positive
    code = nonzero + low2 + high
    code = code | raw >> 12 & nonzero << 3
    return (code | code >> 12).to(tl.int32, bitcast=True)


@triton.jit
def _e2m1_full_fp16_pairs(words, exp):
    raw = words.to(tl.uint32)
    finite = _e2m1_finite_fp16_pairs(words, tl.minimum(exp, 141))
    difference = raw & 2147450879 ^ 2080406528
    biased = difference + 2147450879
    inf_mask = (biased >> 15 & 65537 ^ 65537) * 15
    special = (458759 | raw >> 12 & 524296) & inf_mask
    packed_special = (special | special >> 12).to(tl.int32, bitcast=True)
    return tl.where(exp == 254, packed_special, finite)


__all__ = ["quantize_fp4_indexer"]
