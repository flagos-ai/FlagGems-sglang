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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _offset(
    index, SHAPE: tl.constexpr, STRIDES: tl.constexpr, CONTIG: tl.constexpr
):
    if CONTIG:
        return index
    rest = index
    offset = tl.full(index.shape, 0, index.dtype)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (rest % SHAPE[axis]) * STRIDES[axis]
        rest = rest // SHAPE[axis]
    return offset


@triton.jit
def _load(pointer, mask):
    if pointer.dtype.element_ty == tl.bfloat16:
        word = tl.load(pointer.to(tl.pointer_type(tl.uint16)), mask, other=0)
        return (word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(pointer, mask, other=0).to(tl.float32)


@triton.jit
def _encode_fp8(value, E5: tl.constexpr):
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 128
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    if E5:
        M: tl.constexpr = 2
        BIAS: tl.constexpr = 15
        MAX: tl.constexpr = 57344.0
        MIN_NORMAL: tl.constexpr = 2.0**-14
        ROUND_BIAS: tl.constexpr = 128.0
        ROUND_BITS: tl.constexpr = 0x43000000
    else:
        M: tl.constexpr = 3
        BIAS: tl.constexpr = 7
        MAX: tl.constexpr = 448.0
        MIN_NORMAL: tl.constexpr = 2.0**-6
        ROUND_BIAS: tl.constexpr = 16384.0
        ROUND_BITS: tl.constexpr = 0x46800000
    magnitude = tl.minimum(tl.abs(value), MAX)
    u = magnitude.to(tl.uint32, bitcast=True)
    shift: tl.constexpr = 23 - M
    rounded = u + ((1 << (shift - 1)) - 1) + ((u >> shift) & 1)
    normal = (rounded >> shift) - ((127 - BIAS) << M)
    subnormal = (magnitude + ROUND_BIAS).to(
        tl.uint32, bitcast=True
    ) - ROUND_BITS
    encoded = tl.where(magnitude < MIN_NORMAL, subnormal, normal)
    return tl.where(nan, 127, sign | encoded).to(tl.uint8)


@triton.jit
def _partial_max(
    X,
    P,
    COUNT: tl.constexpr,
    PARTS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(pid * BLOCK, COUNT, PARTS * BLOCK):
        index = start + tl.arange(0, BLOCK)
        x = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), index < COUNT)
        value = tl.where(index < COUNT, tl.abs(x), 0.0)
        maximum = tl.maximum(maximum, value)
    value = tl.max(maximum, 0)
    if FINAL:
        value = value / 448.0
    tl.store(P + pid, value)


@triton.jit
def _quant_tensor(
    X,
    Q,
    S,
    P,
    COUNT: tl.constexpr,
    PARTS: tl.constexpr,
    MODE: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    QSTRIDES: tl.constexpr,
    CONTIG: tl.constexpr,
    QCONTIG: tl.constexpr,
    WIDE: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    value = _load(X + _offset(index, SHAPE, STRIDES, CONTIG), index < COUNT)
    if MODE == 1:
        r = tl.arange(0, REDUCE)
        all_values = _load(X + _offset(r, SHAPE, STRIDES, CONTIG), r < COUNT)
        scale = tl.max(tl.where(r < COUNT, tl.abs(all_values), 0.0), 0) / 448.0
    elif MODE == 2:
        p = tl.arange(0, REDUCE)
        partial = tl.load(P + tl.minimum(p, PARTS - 1))
        scale = tl.max(tl.where(p < PARTS, partial, 0.0), 0) / 448.0
    else:
        scale = tl.load(S)
    value = tl.where(index < COUNT, value, 1.0)
    quantized = _encode_fp8(tl.div_rn(value, scale), False)
    offset = _offset(index, SHAPE, QSTRIDES, QCONTIG)
    tl.store(Q + offset, quantized, index < COUNT)
    if MODE != 0:
        if tl.program_id(0) == 0:
            tl.store(S, scale)


@triton.jit
def _offset_c(i, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    if len(SHAPE) == 1:
        return i * STRIDES[0]
    else:
        off = tl.full(i.shape, 0, tl.int32)
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            off += i % SHAPE[dim] * STRIDES[dim]
            i = i // SHAPE[dim]
        return off


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    nan = mag > 2139095040
    mag = tl.minimum(mag, 1138753536)
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    a = mag.to(tl.float32, bitcast=True)
    rounded = a + 16384.0
    sub = rounded.to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < 1015021568, sub, normal)
    return (tl.where(nan, 127, code) | sign).to(tl.uint8)


@triton.jit
def _small(
    X,
    Q,
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    x = tl.load(X + _offset_c(i, SHAPE, STRIDES), i < N, other=0).to(
        tl.float32
    )
    s = (
        tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    tl.store(S, s)
    tl.store(Q + i, _encode(_ratio_tiny(x, s)), i < N)


@triton.jit
def _partial_bf16(
    X,
    P,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    if N > 2097152 and N <= 4194304 and (X.dtype.element_ty == tl.bfloat16):
        lane = tl.arange(0, 4096)
        maximum = tl.full((4096,), 0, tl.uint16)
        for part in range(B // 4096):
            i = tl.program_id(0) * B + part * 4096 + lane
            x = tl.load(X + _offset_c(i, SHAPE, STRIDES), i < N, other=0)
            bits = x.to(tl.uint16, bitcast=True) & tl.full(
                (), 32767, tl.uint16
            )
            maximum = tl.maximum(maximum, bits)
        value = tl.reduce(maximum, 0, _maximum16)
        tl.static_assert(value.dtype == tl.uint16)
        a = (value.to(tl.int32) << 16).to(tl.float32, bitcast=True)
        tl.store(P + tl.program_id(0), a)
    else:
        pid = tl.program_id(0)
        if N > 4194304:
            pid = tl.cdiv(N, B) - 1 - tl.program_id(0)
        i = pid * B + tl.arange(0, B)
        x = tl.load(X + _offset_c(i, SHAPE, STRIDES), i < N, other=0).to(
            tl.float32
        )
        a = tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        tl.store(P + tl.program_id(0), a)


@triton.jit
def _finish_grouped(
    X,
    P,
    Q,
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.arange(0, R)
    a = tl.load(P + r, r < K, other=0)
    s = (
        tl.max(a.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    if tl.program_id(0) == 0:
        tl.store(S, s)
    for tile in range(4):
        i = (tl.program_id(0) * 4 + tile) * B + tl.arange(0, B)
        x = tl.load(X + _offset_c(i, SHAPE, STRIDES), i < N, other=0).to(
            tl.float32
        )
        tl.store(Q + i, _encode(_ratio_tiny(x, s)), i < N)


@triton.jit
def _maximum16(a, b):
    return tl.maximum(a, b)


@triton.jit
def _ratio_tiny(x, s):
    xb = x.to(tl.int32, bitcast=True)
    sb = s.to(tl.int32, bitcast=True)
    xm = xb & 2147483647
    sm = sb & 2147483647
    if sm < 8388608:
        sn = sm.to(tl.float32) * 2.350988701644575e-38
        sn = (sn.to(tl.int32, bitcast=True) | sb & -2147483648).to(
            tl.float32, bitcast=True
        )
        xt = xm.to(tl.float32) * 2.350988701644575e-38
        xt = (xt.to(tl.int32, bitcast=True) | xb & -2147483648).to(
            tl.float32, bitcast=True
        )
        xn = tl.where(xm < 8388608, xt, x * 16777216.0)
        value = xn / sn
    else:
        value = x / s
    magnitude = value.to(tl.int32, bitcast=True) & 2147483647
    return (magnitude | (xb ^ sb) & -2147483648).to(tl.float32, bitcast=True)


@triton.jit
def _encode_e4m3fn_byte(x):
    u_x = tl.cast(x, tl.int32, bitcast=True)
    sign = u_x >> 24 & 128
    u = u_x & 2147483647
    a = tl.cast(u, tl.float32, bitcast=True)
    u_rounded = u + 524288
    normal_byte = (u_rounded >> 20 & 2047) - 960
    dm = (a * 512.0 + 0.5).to(tl.int32)
    dm_overflow = dm >= 8
    dm = tl.where(dm_overflow, 0, dm)
    dm_exp = tl.where(dm_overflow, 1, 0)
    denorm_byte = dm_exp << 3 | dm
    is_denorm = u < 1015021568
    byte_val = tl.where(is_denorm, denorm_byte, normal_byte)
    byte_val = byte_val | sign
    return byte_val.to(tl.uint8)


@triton.jit
def _absmax_blocks(
    X, max_ptr, n_elements, total_blocks, num_ctas, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    max_val = 0.0
    for block in tl.range(pid, total_blocks, num_ctas):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK), BLOCK)
        mask = offsets < n_elements
        x = tl.load(X + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        block_max = tl.max(abs_x, axis=0)
        block_max_f32 = block_max.to(tl.float32)
        max_val = tl.maximum(max_val, block_max_f32)
    max_int = tl.cast(max_val, tl.int32, bitcast=True)
    tl.store(max_ptr + pid, max_int)


@triton.jit
def _finalize_absmax(
    max_ptr, S, inv_ptr, num_abs_blocks, MAX_ABS_CTAS: tl.constexpr
):
    offs = tl.arange(0, MAX_ABS_CTAS)
    mask = offs < num_abs_blocks
    max_ints = tl.load(max_ptr + offs, mask=mask, other=0)
    max_floats = tl.cast(max_ints, tl.float32, bitcast=True)
    max_val = tl.max(max_floats, axis=0)
    max_val_safe = tl.where(max_val > 0.0, max_val, 1.0)
    scale = max_val / 448.0
    inv_scale = 448.0 / max_val_safe
    tl.store(S, scale)
    tl.store(inv_ptr, inv_scale)


@triton.jit
def _quant_direct(
    X,
    Q8,
    S,
    max_ptr,
    inv_ptr,
    n_elements,
    num_abs_blocks,
    BLOCK: tl.constexpr,
    MAX_ABS_CTAS: tl.constexpr,
    USE_FINALIZED: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_FINALIZED:
        inv_scale = tl.load(inv_ptr)
    else:
        offs_abs = tl.arange(0, MAX_ABS_CTAS)
        mask_abs = offs_abs < num_abs_blocks
        max_ints = tl.load(max_ptr + offs_abs, mask=mask_abs, other=0)
        max_floats = tl.cast(max_ints, tl.float32, bitcast=True)
        max_val = tl.max(max_floats, axis=0)
        max_val_safe = tl.where(max_val > 0.0, max_val, 1.0)
        scale = max_val / 448.0
        inv_scale = 448.0 / max_val_safe
        if pid == 0:
            tl.store(S, scale)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK), BLOCK)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    q = x * inv_scale
    q_byte = _encode_e4m3fn_byte(q)
    tl.store(Q8 + offsets, q_byte, mask=mask)


@triton.jit
def _quant_static_words(
    X,
    Q,
    S,
    N: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    LARGE: tl.constexpr,
):
    scale = tl.load(S)
    for step in range(triton.cdiv(N, B * P)):
        pid = tl.program_id(0)
        if LARGE:
            pid = pid.to(tl.int64)
        idx = (pid + step * P) * B + tl.arange(0, B)
        x = tl.load(X + idx, idx < N, 0).to(tl.float32)
        v = x / scale
        code = _encode(v)
        tl.store(Q + idx, code, idx < N)


@triton.jit
def _divide(x, scale):
    inv = tl.div_rn(1.0, scale)
    q = x * inv
    residual = tl.fma(-q, scale, x)
    refined = tl.fma(residual, inv, q)
    refined = tl.where(tl.abs(q) > 448.0, q, refined)
    if tl.abs(scale) < 1.0e-30 or tl.abs(scale) > 1.0e30:
        refined = tl.div_rn(x, scale)
    return refined


@triton.jit
def _encode_i8(v):
    signed_bits = v.to(tl.int32, bitcast=True)
    abs_bits = signed_bits & 0x7FFFFFFF
    is_nan = abs_bits > 0x7F800000
    bits = tl.minimum(abs_bits, 0x43E00000)
    ax = bits.to(tl.float32, bitcast=True)
    normal = (bits + (0x7FFFF - (960 << 20)) + ((bits >> 20) & 1)) >> 20
    sub = (ax + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(bits < 0x3C800000, sub, normal)
    sign = (signed_bits >> 24) & 128
    byte = tl.where(is_nan, 127, code | sign)
    return tl.where(byte >= 128, byte - 256, byte).to(tl.int8)


@triton.jit
def _partial_strided(
    X,
    T,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    XC: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    LARGE: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0.0, tl.float32)
    for step in range(triton.cdiv(N, B * P)):
        pid = tl.program_id(0)
        if LARGE:
            pid = pid.to(tl.int64)
        idx = (pid + step * P) * B + lane
        off = idx if XC else _offset_c(idx, SHAPE, XS)
        x = tl.load(X + off, idx < N, 0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(T + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _quant_strided(
    X,
    Q,
    S,
    T,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    XC: tl.constexpr,
    PARTS: tl.constexpr,
    PB: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    LARGE: tl.constexpr,
):
    ids = tl.arange(0, PB)
    scale = tl.max(tl.load(T + ids, ids < PARTS, 0), 0) / 448.0
    for step in range(triton.cdiv(N, B * P)):
        pid = tl.program_id(0)
        if LARGE:
            pid = pid.to(tl.int64)
        idx = (pid + step * P) * B + tl.arange(0, B)
        off = idx if XC else _offset_c(idx, SHAPE, XS)
        x = tl.load(X + off, idx < N, 0).to(tl.float32)
        value = _encode_i8(_divide(x, scale))
        tl.store(Q + idx, value, idx < N)
    if tl.program_id(0) == 0:
        tl.store(S, scale)


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    n = input.numel()
    if n == 0:
        return output_q, output_s
    if not input.is_contiguous():
        input = input.contiguous()
    qbytes = output_q.view(torch.uint8)
    if is_static:
        _quant_static_words[(min(triton.cdiv(n, 2048), 65535),)](
            input,
            qbytes,
            output_s,
            n,
            2048,
            min(triton.cdiv(n, 2048), 65535),
            max(
                n,
                sum((d - 1) * s for d, s in zip(input.shape, input.stride())),
            )
            >= 2147483648,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return output_q, output_s
    if n <= 8192:
        _small[(1,)](
            input,
            qbytes,
            output_s,
            n,
            (n,),
            (1,),
            triton.next_power_of_2(n),
            num_warps=16,
        )
    elif n <= 262144:
        if n <= 16384:
            parts = 1
            partial = output_s
        else:
            parts = min(triton.cdiv(n, 4096), 1024)
            partial = torch.empty(
                (parts,), dtype=torch.float32, device=input.device
            )
            _partial_max[(parts,)](
                input,
                partial,
                n,
                parts,
                (n,),
                (1,),
                True,
                False,
                False,
                4096,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        block = 256 if n <= 8192 else 1024
        warps = 8 if n > 8192 else 4
        _quant_tensor[(triton.cdiv(n, block),)](
            input,
            qbytes,
            output_s,
            partial,
            n,
            parts,
            2 if n > 16384 else 1,
            (n,),
            (1,),
            (n,),
            True,
            True,
            False,
            triton.next_power_of_2(n if n <= 16384 else parts),
            block,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    elif n <= 2097152:
        partial_block = 32768 if n > 1048576 else 8192
        k = triton.cdiv(n, partial_block)
        p = torch.empty((k,), dtype=torch.float32, device=input.device)
        _partial_bf16[(k,)](input, p, n, (n,), (1,), partial_block)
        _finish_grouped[(triton.cdiv(n, 8192),)](
            input,
            p,
            qbytes,
            output_s,
            n,
            (n,),
            (1,),
            k,
            triton.next_power_of_2(k),
            2048,
        )
    elif n <= 8388608:
        max_buf = torch.empty(1024, dtype=torch.int32, device=input.device)
        BLOCK_ABS = 8192 if n > 1048576 else 2048
        num_warps_abs = 8 if n > 1048576 else 4
        BLOCK_Q = 4096 if n <= 1048576 else 2048
        total_blocks = triton.cdiv(n, BLOCK_ABS)
        grid_abs = min(total_blocks, 1024)
        use_finalized = n > 1048576
        inv_buf = torch.empty(1, dtype=torch.float32, device=input.device)
        _absmax_blocks[(grid_abs,)](
            input,
            max_buf,
            n,
            total_blocks,
            grid_abs,
            BLOCK=BLOCK_ABS,
            num_warps=num_warps_abs,
        )
        if use_finalized:
            _finalize_absmax[(1,)](
                max_buf,
                output_s,
                inv_buf,
                grid_abs,
                MAX_ABS_CTAS=1024,
                num_warps=1,
            )
        grid_q = (triton.cdiv(n, BLOCK_Q),)
        _quant_direct[grid_q](
            input,
            qbytes,
            output_s,
            max_buf,
            inv_buf,
            n,
            grid_abs,
            BLOCK=BLOCK_Q,
            num_warps=8,
            MAX_ABS_CTAS=1024,
            USE_FINALIZED=use_finalized,
        )
    else:
        red_b = 8192
        parts = min(triton.cdiv(n, red_b), 128)
        tmp = torch.empty((parts,), dtype=torch.float32, device=input.device)
        large = n >= 2147483648
        _partial_strided[(parts,)](
            input,
            tmp,
            n,
            (n,),
            (1,),
            True,
            red_b,
            parts,
            large,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        b = 8192
        p = min(triton.cdiv(n, b), 65535)
        _quant_strided[(p,)](
            input,
            qbytes,
            output_s,
            tmp,
            n,
            (n,),
            (1,),
            True,
            parts,
            triton.next_power_of_2(parts),
            b,
            p,
            large,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output_q, output_s


__all__ = ["per_tensor_quant_fp8"]
