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
def _decode(raw):
    raw = raw.to(tl.int32)
    mag = raw & 127
    normal = (mag + 960 << 20).to(tl.float32, bitcast=True)
    sub = normal + normal - 0.015625
    value = tl.minimum(sub, normal)
    return value


@triton.jit
def _magnitude_code(mag):
    exponent = tl.maximum(mag >> 23, 121)
    bias = exponent + 20 << 23
    rounded = mag.to(tl.float32, bitcast=True) + bias.to(
        tl.float32, bitcast=True
    )
    code = rounded.to(tl.int32, bitcast=True) - bias + (exponent - 121 << 3)
    return code


@triton.jit
def _encode(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 2147483647, 1139802112)
    code = _magnitude_code(mag)
    return (code | bits >> 24 & 128).to(tl.uint8)


@triton.jit
def _positive_encode(value):
    mag = tl.minimum(value, 480.0, propagate_nan=tl.PropagateNan.NONE).to(
        tl.int32, bitcast=True
    )
    return _magnitude_code(mag).to(tl.uint8)


@triton.jit
def _compact(C, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    PS: tl.constexpr = META[4]
    OS: tl.constexpr = META[5]
    G: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.arange(0, 512)
    tail = tl.arange(0, 64)
    if tl.program_id(0) * B + B <= N:
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(C + src[:, None] * CS + col[None, :])
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + col[None, :] // G)
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :]).to(tl.int32)
        scale_bits = scales.to(tl.int32, bitcast=True) & 2147483647
        value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
        sign = (
            raw.to(tl.int32) ^ scales.to(tl.int32, bitcast=True) >> 24
        ) & 128
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_encode(value).to(tl.int32) | sign,
        )
        encoded = encoded.to(tl.uint8)
        tl.store(OUT + row[:, None] * OS + col[None, :], encoded)
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
        )
    else:
        src = tl.load(P + row * PS, row < N, 0).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(
            C + src[:, None] * CS + col[None, :], row[:, None] < N, 0
        )
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + col[None, :] // G, row[:, None] < N, 0)
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :], row[:, None] < N, 0).to(tl.int32)
        scale_bits = scales.to(tl.int32, bitcast=True) & 2147483647
        value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
        sign = (
            raw.to(tl.int32) ^ scales.to(tl.int32, bitcast=True) >> 24
        ) & 128
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_encode(value).to(tl.int32) | sign,
        )
        encoded = tl.where(row[:, None] < N, encoded, 0).to(tl.uint8)
        tl.store(
            OUT + row[:, None] * OS + col[None, :],
            encoded,
            row[:, None] < TOTAL,
        )
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
            row[:, None] < TOTAL,
        )


@triton.jit
def _packed(C, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    PS: tl.constexpr = META[4]
    OS: tl.constexpr = META[5]
    G: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.arange(0, 128)
    tail = tl.arange(0, 32)
    if tl.program_id(0) * B + B <= N:
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        cp = (C + src[:, None] * CS).to(tl.pointer_type(tl.uint32))
        raw4 = tl.load(cp + col[None, :])
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + col[None, :] * 4 // G)
        result = tl.full((B, 128), 0, tl.uint32)
        for lane in tl.static_range(4):
            raw = (raw4 >> lane * 8 & 255).to(tl.int32)
            scale_bits = scales.to(tl.int32, bitcast=True) & 2147483647
            value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
            sign = (raw ^ scales.to(tl.int32, bitcast=True) >> 24) & 128
            encoded = tl.where(
                raw.to(tl.int32) & 127 == 127,
                raw.to(tl.int32),
                _positive_encode(value).to(tl.int32) | sign,
            ).to(tl.uint32)
            result = result | encoded << lane * 8
        result = result
        op = (OUT + row[:, None] * OS).to(tl.pointer_type(tl.uint32))
        tl.store(op + col[None, :], result)
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint32)
        )
        rb = tl.load(rp + tail[None, :])
        r0 = (rb << 16).to(tl.float32, bitcast=True)
        r1 = (rb & 4294901760).to(tl.float32, bitcast=True)
        e0 = _encode(r0).to(tl.uint32) & 127 | rb >> 8 & 128
        e1 = _encode(r1).to(tl.uint32) & 127 | rb >> 24 & 128
        rop = (OUT + row[:, None] * OS + 512).to(tl.pointer_type(tl.uint16))
        tl.store(rop + tail[None, :], (e0 | e1 << 8).to(tl.uint16))
    else:
        src = tl.load(P + row * PS, row < N, 0).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        cp = (C + src[:, None] * CS).to(tl.pointer_type(tl.uint32))
        raw4 = tl.load(cp + col[None, :], row[:, None] < N, 0)
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + col[None, :] * 4 // G, row[:, None] < N, 0)
        result = tl.full((B, 128), 0, tl.uint32)
        for lane in tl.static_range(4):
            raw = (raw4 >> lane * 8 & 255).to(tl.int32)
            scale_bits = scales.to(tl.int32, bitcast=True) & 2147483647
            value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
            sign = (raw ^ scales.to(tl.int32, bitcast=True) >> 24) & 128
            encoded = tl.where(
                raw.to(tl.int32) & 127 == 127,
                raw.to(tl.int32),
                _positive_encode(value).to(tl.int32) | sign,
            ).to(tl.uint32)
            result = result | encoded << lane * 8
        result = tl.where(row[:, None] < N, result, 0)
        op = (OUT + row[:, None] * OS).to(tl.pointer_type(tl.uint32))
        tl.store(op + col[None, :], result, row[:, None] < TOTAL)
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint32)
        )
        rb = tl.load(rp + tail[None, :], row[:, None] < N, 0)
        r0 = (rb << 16).to(tl.float32, bitcast=True)
        r1 = (rb & 4294901760).to(tl.float32, bitcast=True)
        e0 = _encode(r0).to(tl.uint32) & 127 | rb >> 8 & 128
        e1 = _encode(r1).to(tl.uint32) & 127 | rb >> 24 & 128
        rop = (OUT + row[:, None] * OS + 512).to(tl.pointer_type(tl.uint16))
        tl.store(
            rop + tail[None, :],
            (e0 | e1 << 8).to(tl.uint16),
            row[:, None] < TOTAL,
        )


@triton.jit
def _general_layout(Q, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    QC: tl.constexpr = META[4]
    PS: tl.constexpr = META[5]
    OS: tl.constexpr = META[6]
    OC: tl.constexpr = META[7]
    G: tl.constexpr = META[8]
    NORMALIZE_ZERO: tl.constexpr = META[9]
    col = tl.arange(0, 512).to(tl.int64)
    tail = tl.arange(0, 64).to(tl.int64)
    for row in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        if row < N:
            src = tl.load(P + row64 * PS).to(tl.int64)
            src = tl.where(src < 0, src + M, src)
            base = Q + src * QS
            raw = tl.load(base + col * QC)
            s = 512 + col // G * 4
            s0 = tl.load(base + s * QC).to(tl.uint32)
            s1 = tl.load(base + (s + 1) * QC).to(tl.uint32)
            s2 = tl.load(base + (s + 2) * QC).to(tl.uint32)
            s3 = tl.load(base + (s + 3) * QC).to(tl.uint32)
            bits = s0 | s1 << 8 | s2 << 16 | s3 << 24
            scale_bits = bits & 2147483647
            value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
            sign = (raw.to(tl.int32) ^ (bits >> 24).to(tl.int32)) & 128
            sign = tl.where(
                raw.to(tl.int32) & 127 == 127, raw.to(tl.int32) & 128, sign
            )
            if NORMALIZE_ZERO:
                exact_zero = (
                    (raw.to(tl.int32) & 127 == 0) | (bits & 2147483647 == 0)
                ) & (raw.to(tl.int32) & 127 != 127)
                sign = tl.where((row < N // 4 * 4) & exact_zero, 0, sign)
            encoded = (
                tl.where(
                    raw.to(tl.int32) & 127 == 127, 127, _encode(value)
                ).to(tl.int32)
                & 127
                | sign
            ).to(tl.uint8)
            tl.store(OUT + row64 * OS + col * OC, encoded)
            r = 512 + 512 // G * 4 + tail * 2
            r0 = tl.load(base + r * QC).to(tl.uint32)
            r1 = tl.load(base + (r + 1) * QC).to(tl.uint32)
            rb = r0 | r1 << 8
            rope = (rb << 16).to(tl.float32, bitcast=True)
            encoded_rope = (
                _encode(rope).to(tl.int32) & 127 | (rb >> 8).to(tl.int32) & 128
            ).to(tl.uint8)
            tl.store(OUT + row64 * OS + (512 + tail) * OC, encoded_rope)
        else:
            tl.store(OUT + row64 * OS + col * OC, tl.full((512,), 0, tl.uint8))
            tl.store(
                OUT + row64 * OS + (512 + tail) * OC,
                tl.full((64,), 0, tl.uint8),
            )


@triton.jit
def _grouped(C, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    PS: tl.constexpr = META[4]
    OS: tl.constexpr = META[5]
    G: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    row = tl.program_id(0) * B + tl.arange(0, B)
    col = tl.arange(0, 512)
    tail = tl.arange(0, 64)
    if tl.program_id(0) * B + B <= N:
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(C + src[:, None] * CS + col[None, :]).reshape(
            (B, 512 // G, G)
        )
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.uint32))
        scales = tl.load(sp + tl.arange(0, 512 // G)[None, :])[:, :, None]
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :]).to(tl.int32)
        scale_bits = scales & 2147483647
        value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
        sign = (raw.to(tl.int32) ^ (scales >> 24).to(tl.int32)) & 128
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_encode(value).to(tl.int32) | sign,
        )
        encoded = encoded.to(tl.uint8)
        tl.store(
            OUT + row[:, None] * OS + col[None, :], encoded.reshape((B, 512))
        )
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
        )
    else:
        src = tl.load(P + row * PS, row < N, 0).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(
            C + src[:, None] * CS + col[None, :], row[:, None] < N, 0
        ).reshape((B, 512 // G, G))
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.uint32))
        scales = tl.load(
            sp + tl.arange(0, 512 // G)[None, :], row[:, None] < N, 0
        )[:, :, None]
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :], row[:, None] < N, 0).to(tl.int32)
        scale_bits = scales & 2147483647
        value = _decode(raw) * scale_bits.to(tl.float32, bitcast=True)
        sign = (raw.to(tl.int32) ^ (scales >> 24).to(tl.int32)) & 128
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_encode(value).to(tl.int32) | sign,
        )
        encoded = tl.where(row[:, None, None] < N, encoded, 0).to(tl.uint8)
        tl.store(
            OUT + row[:, None] * OS + col[None, :],
            encoded.reshape((B, 512)),
            row[:, None] < TOTAL,
        )
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
            row[:, None] < TOTAL,
        )


def _compute(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total == 0:
        return out
    qs = quant_k_cache.stride()
    os = out.stride()
    ps = page_table_1_flattened.stride(0)
    m = quant_k_cache.shape[0]
    aligned = (
        qs[-1] == 1
        and os[-1] == 1
        and (
            (
                quant_k_cache.data_ptr()
                | out.data_ptr()
                | qs[0]
                | qs[1]
                | os[0]
                | os[1]
            )
            & 3
            == 0
        )
    )
    compatible = (
        aligned
        and m * qs[0] < 2147483648
        and (out.shape[0] * os[0] < 2147483648)
        and (n * ps < 2147483648)
    )
    if compatible:
        b = 2 if n >= 4096 and group_size == 128 else 1
        kernel = (
            _grouped
            if group_size == 128 and n >= 4096
            else _packed if n >= 8192 else _compact
        )
        kernel[triton.cdiv(total, b),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], ps, os[0], group_size, b),
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        _general_layout[min(total, 32768),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], qs[-1], ps, os[0], os[-1], group_size, False),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _alias_writeback(
    T, OUT, TOTAL: tl.constexpr, OS: tl.constexpr, OC: tl.constexpr
):
    col = tl.arange(0, 512).to(tl.int64)
    tail = tl.arange(0, 64).to(tl.int64)
    for row in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        r = row.to(tl.int64)
        first = tl.load(T + r * 576 + col)
        last = tl.load(T + r * 576 + 512 + tail)
        tl.store(OUT + r * OS + col * OC, first)
        tl.store(OUT + r * OS + (512 + tail) * OC, last)


def gather_dequant_requant_fp8_paged(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total == 0:
        return out
    qs = quant_k_cache.stride()
    os = out.stride()
    ps = page_table_1_flattened.stride(0)
    m = quant_k_cache.shape[0]
    qp = quant_k_cache.data_ptr()
    pp = page_table_1_flattened.data_ptr()
    start = out.data_ptr()
    end = start + (total - 1) * os[0] + 575 * os[-1] + 1
    qe = qp + (quant_k_cache.shape[0] - 1) * qs[0] + 655 * qs[-1] + 1
    pe = pp + (n - 1) * ps * 4 + 4
    if start < qe and qp < end or (start < pe and pp < end):
        temp = torch.empty((total, 1, 576), dtype=out.dtype, device=out.device)
        _compute(
            quant_k_cache, page_table_1_flattened, group_size, extra_rows, temp
        )
        _alias_writeback[min(total, 80),](
            temp.view(torch.uint8),
            out.view(torch.uint8),
            total,
            os[0],
            os[-1],
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    aligned = (
        qs[-1] == 1
        and os[-1] == 1
        and ((qp | start | qs[0] | qs[1] | os[0] | os[1]) & 3 == 0)
    )
    compatible = (
        aligned
        and m * qs[0] < 2147483648
        and (out.shape[0] * os[0] < 2147483648)
        and (n * ps < 2147483648)
    )
    if compatible:
        b = 2 if n >= 4096 and group_size == 128 else 1
        kernel = (
            _grouped
            if group_size == 128 and n >= 4096
            else _packed if n >= 8192 else _compact
        )
        kernel[triton.cdiv(total, b),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], ps, os[0], group_size, b),
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        _general_layout[min(total, 32768),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], qs[-1], ps, os[0], os[-1], group_size, False),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["gather_dequant_requant_fp8_paged"]
