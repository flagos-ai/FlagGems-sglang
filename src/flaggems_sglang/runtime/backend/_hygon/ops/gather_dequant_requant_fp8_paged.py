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
    mag = raw.to(tl.int32) & 127
    codes = tl.arange(0, 128)
    normal = (codes + 960 << 20).to(tl.float32, bitcast=True)
    table = tl.minimum(normal + normal - 0.015625, normal)
    return tl.gather(table, mag.reshape((mag.numel,)), axis=0).reshape(
        mag.shape
    )


@triton.jit
def _magnitude_code(mag):
    bias = (tl.maximum(mag, 1015021568) & 2139095040) + 167772160
    rounded = mag.to(tl.float32, bitcast=True) + bias.to(
        tl.float32, bitcast=True
    )
    return rounded.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128


@triton.jit
def _encode_magnitude(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 2147483647, 1139802112)
    return _magnitude_code(mag)


@triton.jit
def _positive_magnitude(value):
    bits = value.to(tl.uint32, bitcast=True)
    mag = tl.minimum(bits, 1139802112)
    return _magnitude_code(mag)


@triton.jit
def _encode(value):
    bits = value.to(tl.int32, bitcast=True)
    return (_encode_magnitude(value) | bits >> 24 & 128).to(tl.uint8)


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
        value = _decode(raw) * tl.abs(scales)
        sign = (
            raw.to(tl.int32) ^ scales.to(tl.int32, bitcast=True) >> 24
        ) & 128
        sign = sign
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_magnitude(value).to(tl.int32) | sign,
        )
        encoded = encoded.to(tl.uint8)
        tl.store(OUT + row[:, None] * OS + col[None, :], encoded)
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :]).to(tl.int32)
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
        )
    elif tl.program_id(0) * B < N:
        src = tl.load(P + row * PS, row < N, 0).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(
            C + src[:, None] * CS + col[None, :], row[:, None] < N, 0
        )
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + col[None, :] // G, row[:, None] < N, 0)
        value = _decode(raw) * tl.abs(scales)
        sign = (
            raw.to(tl.int32) ^ scales.to(tl.int32, bitcast=True) >> 24
        ) & 128
        sign = sign
        encoded = tl.where(
            raw.to(tl.int32) & 127 == 127,
            raw.to(tl.int32),
            _positive_magnitude(value).to(tl.int32) | sign,
        )
        encoded = tl.where(row[:, None] < N, encoded, 0).to(tl.uint8)
        tl.store(
            OUT + row[:, None] * OS + col[None, :],
            encoded,
            row[:, None] < TOTAL,
        )
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :], row[:, None] < N, 0).to(tl.int32)
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
            row[:, None] < TOTAL,
        )
    else:
        tl.store(
            OUT + row[:, None] * OS + col[None, :],
            tl.full((B, 512), 0, tl.uint8),
            row[:, None] < TOTAL,
        )
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            tl.full((B, 64), 0, tl.uint8),
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
            scale = bits.to(tl.float32, bitcast=True)
            value = _decode(raw) * tl.abs(scale)
            sign = (raw.to(tl.int32) ^ (bits >> 24).to(tl.int32)) & 128
            sign = sign
            if NORMALIZE_ZERO:
                exact_zero = (
                    (raw.to(tl.int32) & 127 == 0) | (bits & 2147483647 == 0)
                ) & (raw.to(tl.int32) & 127 != 127)
                sign = tl.where((row < N // 4 * 4) & exact_zero, 0, sign)
            encoded = tl.where(
                raw.to(tl.int32) & 127 == 127,
                raw.to(tl.int32),
                (_encode_magnitude(value).to(tl.int32) | sign).to(tl.uint8),
            )
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
        words = tl.load(
            (C + src[:, None] * CS).to(tl.pointer_type(tl.uint32))
            + tl.arange(0, 128)[None, :]
        ).reshape((B, 512 // G, G // 4))
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(sp + tl.arange(0, 512 // G)[None, :])[:, :, None]
        encoded = _requant_words(words, scales, B, G)
        tl.store(
            (OUT + row[:, None] * OS).to(tl.pointer_type(tl.uint32))
            + tl.arange(0, 128)[None, :],
            encoded.reshape((B, 128)),
        )
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :]).to(tl.int32)
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
        )
    elif tl.program_id(0) * B < N:
        src = tl.load(P + row * PS, row < N, 0).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        words = tl.load(
            (C + src[:, None] * CS).to(tl.pointer_type(tl.uint32))
            + tl.arange(0, 128)[None, :],
            row[:, None] < N,
            0,
        ).reshape((B, 512 // G, G // 4))
        sp = (C + src[:, None] * CS + 512).to(tl.pointer_type(tl.float32))
        scales = tl.load(
            sp + tl.arange(0, 512 // G)[None, :], row[:, None] < N, 0
        )[:, :, None]
        encoded = _requant_words(words, scales, B, G)
        encoded = tl.where(row[:, None, None] < N, encoded, 0)
        tl.store(
            (OUT + row[:, None] * OS).to(tl.pointer_type(tl.uint32))
            + tl.arange(0, 128)[None, :],
            encoded.reshape((B, 128)),
            row[:, None] < TOTAL,
        )
        rp = (C + src[:, None] * CS + 512 + 512 // G * 4).to(
            tl.pointer_type(tl.uint16)
        )
        rb = tl.load(rp + tail[None, :], row[:, None] < N, 0).to(tl.int32)
        rope = (rb << 16).to(tl.float32, bitcast=True)
        encoded_rope = _encode(rope).to(tl.int32) & 127 | rb >> 8 & 128
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            encoded_rope.to(tl.uint8),
            row[:, None] < TOTAL,
        )
    else:
        tl.store(
            OUT + row[:, None] * OS + col[None, :],
            tl.full((B, 512), 0, tl.uint8),
            row[:, None] < TOTAL,
        )
        tl.store(
            OUT + row[:, None] * OS + 512 + tail[None, :],
            tl.full((B, 64), 0, tl.uint8),
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
        b = (
            8
            if n >= 8192 and group_size == 128
            else (
                4 if n >= 2048 and group_size == 128 else 2 if n >= 2048 else 1
            )
        )
        (_grouped if group_size == 128 else _compact).run(
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], ps, os[0], group_size, b),
            num_warps=(
                4
                if n >= 8192 and group_size == 128
                else 2 if n >= 2048 and group_size == 128 else 1
            ),
            enable_fp_fusion=False,
            grid=(triton.cdiv(total, b),),
            warmup=False,
        )
    else:
        _general_layout.run(
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], qs[-1], ps, os[0], os[-1], group_size, False),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(min(total, 32768),),
            warmup=False,
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
        _alias_writeback.run(
            temp.view(torch.uint8),
            out.view(torch.uint8),
            total,
            os[0],
            os[-1],
            num_warps=4,
            enable_fp_fusion=False,
            grid=(min(total, 80),),
            warmup=False,
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
        b = (
            8
            if n >= 8192 and group_size == 128
            else (
                4 if n >= 2048 and group_size == 128 else 2 if n >= 2048 else 1
            )
        )
        (_grouped if group_size == 128 else _compact).run(
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], ps, os[0], group_size, b),
            num_warps=(
                4
                if n >= 8192 and group_size == 128
                else 2 if n >= 2048 and group_size == 128 else 1
            ),
            enable_fp_fusion=False,
            grid=(triton.cdiv(total, b),),
            warmup=False,
        )
    else:
        _general_layout.run(
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            (n, total, m, qs[0], qs[-1], ps, os[0], os[-1], group_size, False),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(min(total, 32768),),
            warmup=False,
        )
    return out


@triton.jit
def _requant_words(words, scales, B: tl.constexpr, G: tl.constexpr):
    shift = tl.arange(0, 4) * 8
    raw = (
        (words[:, :, :, None] >> shift[None, None, None, :] & 255)
        .to(tl.uint8)
        .reshape((B, 512 // G, G))
    )
    value = _decode(raw) * tl.abs(scales)
    codes = _positive_magnitude(value).to(tl.uint32).reshape((B, 128, 2, 2))
    even, odd = tl.split(codes)
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    packed = (c0 | c1 << 8 | c2 << 16 | c3 << 24).reshape(
        (B, 512 // G, G // 4)
    )
    signs = (
        words ^ (scales.to(tl.uint32, bitcast=True) >> 31) * 2155905152
    ) & 2155905152
    result = packed | signs
    absent = ~words & 2139062143
    carry = absent + 2139062143
    nanmask = ((~carry & 2155905152) >> 7) * 255
    return result ^ (result ^ words) & nanmask


__all__ = ["gather_dequant_requant_fp8_paged"]
