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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _encode(value):
    return (
        value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    )


@triton.jit
def _packed(
    X,
    F,
    P,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    FR: tl.constexpr,
    HAS_P: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
    BP: tl.constexpr,
    BF16: tl.constexpr,
    LINEAR: tl.constexpr,
    FULL: tl.constexpr,
):
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + D - 1 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    row = pid * BR + tl.arange(0, BR)
    token = row // H
    head = row % H
    p = tl.arange(0, BP)
    if WIDE:
        p = p.to(tl.int64)
    mask = (row[:, None] < T * H) & (p[None, :] < D // 2)
    xi = row * (D // 2) if LINEAR else token * (SX0 // 2) + head * (SX1 // 2)
    xp = X.to(tl.pointer_type(tl.uint32)) + xi[:, None] + p[None, :]
    if FULL:
        bits = tl.load(xp)
    else:
        bits = tl.load(xp, mask, 0)
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
    freq_row = token
    if HAS_P:
        freq_row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
        freq_row = tl.where(freq_row < 0, freq_row + FR, freq_row)
    fb = tl.load(
        F.to(tl.pointer_type(tl.uint64))
        + freq_row[:, None] * SF0
        + p[None, :] * SF1,
        mask,
        0,
    )
    c = fb.to(tl.uint32).to(tl.float32, bitcast=True)
    s = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    if BF16:
        out = _encode(real) | _encode(imag) << 16
    else:
        out = (
            real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            | imag.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            << 16
        )
    op = (
        O.to(tl.pointer_type(tl.uint32)) + row[:, None] * (D // 2) + p[None, :]
    )
    if FULL:
        tl.store(op, out)
    else:
        tl.store(op, out, mask)


@triton.jit
def _head_loop(
    X,
    F,
    P,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    FR: tl.constexpr,
    HAS_P: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
    BP: tl.constexpr,
    BF16: tl.constexpr,
    FULL: tl.constexpr,
):
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 > 2147483647
        or (T - 1) * SP > 2147483647
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid * BR + tl.arange(0, BR)
    p = tl.arange(0, BP)
    if WIDE:
        p = p.to(tl.int64)
    mask = (token[:, None] < T) & (p[None, :] < D // 2)
    freq_row = token
    if HAS_P:
        freq_row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
        freq_row = tl.where(freq_row < 0, freq_row + FR, freq_row)
    fb = tl.load(
        F.to(tl.pointer_type(tl.uint64))
        + freq_row[:, None] * SF0
        + p[None, :] * SF1,
        mask,
        0,
    )
    c = fb.to(tl.uint32).to(tl.float32, bitcast=True)
    s = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    for head in range(0, H, 2):
        xi0 = (token * H + head) * (D // 2)
        xp0 = X.to(tl.pointer_type(tl.uint32)) + xi0[:, None] + p[None, :]
        if FULL:
            bits0 = tl.load(xp0)
        else:
            bits0 = tl.load(xp0, mask, 0)
        xi1 = (token * H + (head + 1)) * (D // 2)
        xp1 = X.to(tl.pointer_type(tl.uint32)) + xi1[:, None] + p[None, :]
        if FULL:
            bits1 = tl.load(xp1)
        else:
            bits1 = tl.load(xp1, mask, 0)
        if BF16:
            a0 = ((bits0 & 65535) << 16).to(tl.float32, bitcast=True)
            b0 = (bits0 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a0 = (
                bits0.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b0 = (
                (bits0 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real0 = a0 * c + b0 * s
            imag0 = b0 * c - a0 * s
        else:
            real0 = a0 * c - b0 * s
            imag0 = b0 * c + a0 * s
        if BF16:
            out0 = _encode(real0) | _encode(imag0) << 16
        else:
            out0 = (
                real0.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag0.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op0 = O.to(tl.pointer_type(tl.uint32)) + xi0[:, None] + p[None, :]
        if FULL:
            tl.store(op0, out0)
        else:
            tl.store(op0, out0, mask)
        if BF16:
            a1 = ((bits1 & 65535) << 16).to(tl.float32, bitcast=True)
            b1 = (bits1 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a1 = (
                bits1.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b1 = (
                (bits1 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real1 = a1 * c + b1 * s
            imag1 = b1 * c - a1 * s
        else:
            real1 = a1 * c - b1 * s
            imag1 = b1 * c + a1 * s
        if BF16:
            out1 = _encode(real1) | _encode(imag1) << 16
        else:
            out1 = (
                real1.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag1.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op1 = O.to(tl.pointer_type(tl.uint32)) + xi1[:, None] + p[None, :]
        if FULL:
            tl.store(op1, out1)
        else:
            tl.store(op1, out1, mask)


@triton.jit
def _head_quad(
    X,
    F,
    P,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    FR: tl.constexpr,
    HAS_P: tl.constexpr,
    INV: tl.constexpr,
    BR: tl.constexpr,
    BP: tl.constexpr,
    BF16: tl.constexpr,
    FULL: tl.constexpr,
):
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 > 2147483647
        or (T - 1) * SP > 2147483647
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    token = pid * BR + tl.arange(0, BR)
    p = tl.arange(0, BP)
    if WIDE:
        p = p.to(tl.int64)
    mask = (token[:, None] < T) & (p[None, :] < D // 2)
    freq_row = token
    if HAS_P:
        freq_row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
        freq_row = tl.where(freq_row < 0, freq_row + FR, freq_row)
    fb = tl.load(
        F.to(tl.pointer_type(tl.uint64))
        + freq_row[:, None] * SF0
        + p[None, :] * SF1,
        mask,
        0,
    )
    c = fb.to(tl.uint32).to(tl.float32, bitcast=True)
    s = (fb >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    for head in range(0, H, 4):
        xi0 = (token * H + head) * (D // 2)
        xp0 = X.to(tl.pointer_type(tl.uint32)) + xi0[:, None] + p[None, :]
        if FULL:
            bits0 = tl.load(xp0)
        else:
            bits0 = tl.load(xp0, mask, 0)
        xi1 = (token * H + (head + 1)) * (D // 2)
        xp1 = X.to(tl.pointer_type(tl.uint32)) + xi1[:, None] + p[None, :]
        if FULL:
            bits1 = tl.load(xp1)
        else:
            bits1 = tl.load(xp1, mask, 0)
        xi2 = (token * H + (head + 2)) * (D // 2)
        xp2 = X.to(tl.pointer_type(tl.uint32)) + xi2[:, None] + p[None, :]
        if FULL:
            bits2 = tl.load(xp2)
        else:
            bits2 = tl.load(xp2, mask, 0)
        xi3 = (token * H + (head + 3)) * (D // 2)
        xp3 = X.to(tl.pointer_type(tl.uint32)) + xi3[:, None] + p[None, :]
        if FULL:
            bits3 = tl.load(xp3)
        else:
            bits3 = tl.load(xp3, mask, 0)
        if BF16:
            a0 = ((bits0 & 65535) << 16).to(tl.float32, bitcast=True)
            b0 = (bits0 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a0 = (
                bits0.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b0 = (
                (bits0 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real0 = a0 * c + b0 * s
            imag0 = b0 * c - a0 * s
        else:
            real0 = a0 * c - b0 * s
            imag0 = b0 * c + a0 * s
        if BF16:
            out0 = _encode(real0) | _encode(imag0) << 16
        else:
            out0 = (
                real0.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag0.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op0 = O.to(tl.pointer_type(tl.uint32)) + xi0[:, None] + p[None, :]
        if FULL:
            tl.store(op0, out0)
        else:
            tl.store(op0, out0, mask)
        if BF16:
            a1 = ((bits1 & 65535) << 16).to(tl.float32, bitcast=True)
            b1 = (bits1 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a1 = (
                bits1.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b1 = (
                (bits1 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real1 = a1 * c + b1 * s
            imag1 = b1 * c - a1 * s
        else:
            real1 = a1 * c - b1 * s
            imag1 = b1 * c + a1 * s
        if BF16:
            out1 = _encode(real1) | _encode(imag1) << 16
        else:
            out1 = (
                real1.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag1.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op1 = O.to(tl.pointer_type(tl.uint32)) + xi1[:, None] + p[None, :]
        if FULL:
            tl.store(op1, out1)
        else:
            tl.store(op1, out1, mask)
        if BF16:
            a2 = ((bits2 & 65535) << 16).to(tl.float32, bitcast=True)
            b2 = (bits2 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a2 = (
                bits2.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b2 = (
                (bits2 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real2 = a2 * c + b2 * s
            imag2 = b2 * c - a2 * s
        else:
            real2 = a2 * c - b2 * s
            imag2 = b2 * c + a2 * s
        if BF16:
            out2 = _encode(real2) | _encode(imag2) << 16
        else:
            out2 = (
                real2.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag2.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op2 = O.to(tl.pointer_type(tl.uint32)) + xi2[:, None] + p[None, :]
        if FULL:
            tl.store(op2, out2)
        else:
            tl.store(op2, out2, mask)
        if BF16:
            a3 = ((bits3 & 65535) << 16).to(tl.float32, bitcast=True)
            b3 = (bits3 & 4294901760).to(tl.float32, bitcast=True)
        else:
            a3 = (
                bits3.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            )
            b3 = (
                (bits3 >> 16)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
        if INV:
            real3 = a3 * c + b3 * s
            imag3 = b3 * c - a3 * s
        else:
            real3 = a3 * c - b3 * s
            imag3 = b3 * c + a3 * s
        if BF16:
            out3 = _encode(real3) | _encode(imag3) << 16
        else:
            out3 = (
                real3.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                | imag3.to(tl.float16)
                .to(tl.uint16, bitcast=True)
                .to(tl.uint32)
                << 16
            )
        op3 = O.to(tl.pointer_type(tl.uint32)) + xi3[:, None] + p[None, :]
        if FULL:
            tl.store(op3, out3)
        else:
            tl.store(op3, out3, mask)


def _baseline_apply(x, freqs_cis, positions, inverse):
    if x.shape[0] < 4096:
        return _baseline(x, freqs_cis, positions, inverse)
    t = x.shape[0]
    h = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    if (
        h % 2 == 0
        and t >= 4096
        and (x.numel() >= 1048576)
        and (1 < h <= 32)
        and (d <= 128)
        and (d & d - 1 == 0)
        and (x.dtype in (torch.bfloat16, torch.float16))
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
    ):
        f = triton.reinterpret(freqs_cis, tl.float32)
        p = positions if positions is not None else x
        sp = positions.stride(0) if positions is not None else 0
        kernel = _head_quad if h % 4 == 0 else _head_loop
        kernel.run(
            x,
            f,
            p,
            x,
            t,
            h,
            d,
            freqs_cis.stride(0),
            freqs_cis.stride(1),
            sp,
            freqs_cis.shape[0],
            positions is not None,
            inverse,
            64,
            d // 2,
            x.dtype == torch.bfloat16,
            t % 64 == 0,
            grid=((t + 64 - 1) // 64,),
            warmup=False,
            num_warps=1,
            groups_per_cluster=64,
            enable_fp_fusion=False,
            isCloseDtypeConvert=x.dtype == torch.bfloat16,
        )
        return x
    return _baseline(x, freqs_cis, positions, inverse)


RAW195 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits195 { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode195(f32v value) "
    "{\n"
    "    bits195 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32768u) >> 16;\n"
    "    return word;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_rope195(\n'
    "    gu16* X, gf32* F, gu16* O,\n"
    "    int T, int H, int INV, int COPY, int pid, int np) {\n"
    "    unsigned int xb[128] __attribute__((aligned(64)));\n"
    "    float fb[64] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    const int groups = H / 4;\n"
    "    const int blocks = T * groups;\n"
    "    for (int block = pid * 64 + core; block < blocks; block += np"
    " * 64) {\n"
    "        const int offset = block * 256;\n"
    "        __builtin_xpu_gm2lm_v3(xb, X + offset, 0, 512);\n"
    "        if (!COPY) {\n"
    "            const int token = block / groups;\n"
    "            __builtin_xpu_gm2lm_v3(fb, F + token * 64, 0, 256);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        if (!COPY) {\n"
    "            #pragma clang loop vectorize(disable)\n"
    "            for (int pair = 0; pair < 32; ++pair) {\n"
    "                fc[pair] = fb[pair * 2];\n"
    "                fs[pair] = fb[pair * 2 + 1];\n"
    "            }\n"
    "            for (int head = 0; head < 4; ++head) {\n"
    "                for (int pair = 0; pair < 32; pair += 16) {\n"
    "                    const int index = head * 32 + pair;\n"
    "                    f32v c = *((f32v*)(fc + pair));\n"
    "                    f32v s = *((f32v*)(fs + pair));\n"
    "                    u32v packed = *((u32v*)(xb + index));\n"
    "                    bits195 real, imag;\n"
    "                    real.u = packed << 16;\n"
    "                    imag.u = packed & 0xffff0000u;\n"
    "                    f32v ac = real.f * c;\n"
    "                    f32v bs = imag.f * s;\n"
    "                    f32v bc = imag.f * c;\n"
    "                    f32v as = real.f * s;\n"
    '                    __asm__("" : "+v"(ac), "+v"(bs), "+v"(bc), "+'
    'v"(as));\n'
    "                    if (INV) {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac + b"
    "s) | (encode195(bc - as) << 16);\n"
    "                    } else {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac - b"
    "s) | (encode195(bc + as) << 16);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + offset, xb, 0, 512);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW195, flags=("-ffp-contract=off",))
def _device_rope195(X, F, Out, T, H, INV, COPY, pid, np):
    pass


@triton.jit(do_not_specialize=["X", "F", "Out"])
def _entry195(X, F, Out, META: tl.constexpr):
    if Out is None:
        Out = X
    tle.raw.call(
        _device_rope195,
        (
            X,
            F,
            Out,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _raw_eligible(x, freqs_cis, positions):
    return (
        x.ndim == 3
        and x.dtype == torch.bfloat16
        and (x.shape[-1] == 64)
        and (x.shape[1] % 4 == 0)
        and (x.numel() >= 65536)
        and (x.numel() + 256 <= 2147483647)
        and (positions is None)
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
        and (freqs_cis.shape[-1] == 32)
        and ((x.data_ptr() | freqs_cis.data_ptr()) & 63 == 0)
    )


def _raw_launch(
    x, freqs_cis, positions, inverse, destination=None, copy=False
):
    t, h, d = x.shape
    f = torch.view_as_real(freqs_cis)
    out = x if destination is None else destination
    if h == 16 and t >= 512 and (t * h * d + 1024 <= 2147483647):
        programs = min(12, (t + 64 - 1) // 64)
        _entry_token195.run(
            x,
            f,
            destination,
            (t, h, int(inverse), int(copy)),
            grid=(programs,),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        programs = min(12, (t * (h // 4) + 64 - 1) // 64)
        _entry195.run(
            x,
            f,
            destination,
            (t, h, int(inverse), int(copy)),
            grid=(programs,),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out


def _current_apply(x, freqs_cis, positions, inverse):
    if _raw_eligible(x, freqs_cis, positions):
        return _raw_launch(x, freqs_cis, positions, inverse)
    return _baseline_apply(x, freqs_cis, positions, inverse)


def _small_eligible(x, freqs_cis, positions):
    return (
        x.ndim == 3
        and x.dtype == torch.bfloat16
        and ((shape := x.shape)[-1] == 64)
        and (shape[1] % 4 == 0)
        and (1 <= x.numel() <= 2147483391)
        and (positions is None)
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
        and (freqs_cis.shape[-1] == 32)
        and ((x.data_ptr() | freqs_cis.data_ptr()) & 63 == 0)
    )


def _small_apply(x, freqs_cis, positions, inverse):
    if _small_eligible(x, freqs_cis, positions):
        return _raw_launch(x, freqs_cis, positions, inverse)
    return _baseline_apply(x, freqs_cis, positions, inverse)


def _frozen_apply(x, freqs_cis, positions, inverse):
    if _small_eligible(x, freqs_cis, positions):
        return _raw_launch(x, freqs_cis, positions, inverse)
    return _baseline_apply(x, freqs_cis, positions, inverse)


RAW_TOKEN195 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits195 { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode195(f32v value) "
    "{\n"
    "    bits195 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32768u) >> 16;\n"
    "    return word;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_token195(\n'
    "    gu16* X, gf32* F, gu16* O,\n"
    "    int T, int H, int INV, int COPY, int pid, int np) {\n"
    "    unsigned int xb[512] __attribute__((aligned(64)));\n"
    "    float fb[64] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    const int groups = H / 16;\n"
    "    const int blocks = T * groups;\n"
    "    for (int block = pid * 64 + core; block < blocks; block += np"
    " * 64) {\n"
    "        const int offset = block * 1024;\n"
    "        __builtin_xpu_gm2lm_v3(xb, X + offset, 0, 2048);\n"
    "        if (!COPY) {\n"
    "            const int token = block / groups;\n"
    "            __builtin_xpu_gm2lm_v3(fb, F + token * 64, 0, 256);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        if (!COPY) {\n"
    "            #pragma clang loop vectorize(disable)\n"
    "            for (int pair = 0; pair < 32; ++pair) {\n"
    "                fc[pair] = fb[pair * 2];\n"
    "                fs[pair] = fb[pair * 2 + 1];\n"
    "            }\n"
    "            for (int head = 0; head < 16; ++head) {\n"
    "                for (int pair = 0; pair < 32; pair += 16) {\n"
    "                    const int index = head * 32 + pair;\n"
    "                    f32v c = *((f32v*)(fc + pair));\n"
    "                    f32v s = *((f32v*)(fs + pair));\n"
    "                    u32v packed = *((u32v*)(xb + index));\n"
    "                    bits195 real, imag;\n"
    "                    real.u = packed << 16;\n"
    "                    imag.u = packed & 0xffff0000u;\n"
    "                    f32v ac = real.f * c;\n"
    "                    f32v bs = imag.f * s;\n"
    "                    f32v bc = imag.f * c;\n"
    "                    f32v as = real.f * s;\n"
    '                    __asm__("" : "+v"(ac), "+v"(bs), "+v"(bc), "+'
    'v"(as));\n'
    "                    if (INV) {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac + b"
    "s) | (encode195(bc - as) << 16);\n"
    "                    } else {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac - b"
    "s) | (encode195(bc + as) << 16);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + offset, xb, 0, 2048);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_TOKEN195, flags=("-ffp-contract=off",))
def _device_token195(X, F, Out, T, H, INV, COPY, pid, np):
    pass


@triton.jit(do_not_specialize=["X", "F", "Out"])
def _entry_token195(X, F, Out, META: tl.constexpr):
    if Out is None:
        Out = X
    tle.raw.call(
        _device_token195,
        (
            X,
            F,
            Out,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _recommended_apply(x, freqs_cis, positions, inverse):
    if _small_eligible(x, freqs_cis, positions):
        return _raw_launch(x, freqs_cis, positions, inverse)
    return _baseline_apply(x, freqs_cis, positions, inverse)


@triton.jit
def _rope_pairs(
    X,
    F,
    P,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SX2: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    FR: tl.constexpr,
    HAS_P: tl.constexpr,
    INV: tl.constexpr,
    B: tl.constexpr,
    NB: tl.constexpr,
):
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
    pair = pid % NB * B + tl.arange(0, B)
    head = pair // (D // 2)
    dim = pair % (D // 2)
    valid = pair < H * (D // 2)
    row = token
    if HAS_P:
        row = tl.load(P + token * SP).to(tl.int64)
        row = tl.where(row < 0, row + FR, row)
    base = token * SX0 + head * SX1 + 2 * dim * SX2
    a = tl.load(X + base, valid, 0).to(tl.float32)
    b = tl.load(X + base + SX2, valid, 0).to(tl.float32)
    c = tl.load(F + row * SF0 + dim * SF1, valid, 0).to(tl.float32)
    s = tl.load(F + row * SF0 + dim * SF1 + 1, valid, 0).to(tl.float32)
    ac = tl.fma(a, c, 0.0)
    bs = tl.fma(b, s, 0.0)
    bc = tl.fma(b, c, 0.0)
    ass = tl.fma(a, s, 0.0)
    if INV:
        real = ac + bs
        imag = bc - ass
    else:
        real = ac - bs
        imag = bc + ass
    op = token * H * D + 2 * pair
    tl.store(O + op, real, valid)
    tl.store(O + op + 1, imag, valid)


def _baseline(x, freqs_cis, positions, inverse):
    t = x.shape[0]
    h = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    if x.numel() == 0:
        return x
    contiguous = x.is_contiguous()
    f = triton.reinterpret(freqs_cis, tl.float32)
    p = positions if positions is not None else x
    sp = positions.stride(0) if positions is not None else 0
    sh = x.stride(1) if x.ndim == 3 else 0
    packed = (
        d <= 128
        and x.dtype in (torch.bfloat16, torch.float16)
        and (x.stride(-1) == 1)
        and all((s % 2 == 0 for s in x.stride()[:-1]))
        and (x.storage_offset() % 2 == 0)
    )
    if packed:
        out = (
            x
            if contiguous
            else torch.empty(x.shape, dtype=x.dtype, device=x.device)
        )
        br = 64
        bp = triton.next_power_of_2(d // 2)
        _packed.run(
            x,
            f,
            p,
            out,
            t,
            h,
            d,
            x.stride(0),
            sh,
            freqs_cis.stride(0),
            freqs_cis.stride(1),
            sp,
            freqs_cis.shape[0],
            positions is not None,
            inverse,
            br,
            bp,
            x.dtype == torch.bfloat16,
            contiguous,
            t * h % br == 0 and bp == d // 2,
            grid=((t * h + br - 1) // br,),
            warmup=False,
            num_warps=1,
            groups_per_cluster=64,
            enable_fp_fusion=False,
            isCloseDtypeConvert=x.dtype == torch.bfloat16,
        )
        return out
    x = x.contiguous()
    sh = x.stride(1) if x.ndim == 3 else 0
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block = min(4096, triton.next_power_of_2(h * d))
    nb = (h * d + block - 1) // block
    _rope_pairs.run(
        x,
        f,
        p,
        out,
        t,
        h,
        d,
        x.stride(0),
        sh,
        x.stride(-1),
        freqs_cis.stride(0) * 2,
        freqs_cis.stride(1) * 2,
        sp,
        freqs_cis.shape[0],
        positions is not None,
        inverse,
        block // 2,
        nb,
        grid=(t * nb,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


RAW_PIPELINE195 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits195 { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode195(f32v value) "
    "{\n"
    "    bits195 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32767u + ((original >> 16) & 1u)) >> "
    "16;\n"
    "    u32v diff = 0x7f800000u - (original & 0x7fffffffu);\n"
    '    __asm__("" : "+v"(diff));\n'
    "    u32v nanmask = 0u - (diff >> 31);\n"
    '    __asm__("" : "+v"(nanmask));\n'
    "    return (word & ~nanmask) | (((original >> 16) | 64u) & nanmas"
    "k);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_pipeline195(\n'
    "    gu16* X, gf32* F, gu16* O,\n"
    "    int T, int H, int INV, int COPY, int pid, int np) {\n"
    "    unsigned int storage_x[1024] __attribute__((aligned(64)));\n"
    "    float storage_f[128] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    const int groups = H / 16;\n"
    "    const int blocks = T * groups;\n"
    "    int block = pid * 64 + core;\n"
    "    const int step = np * 64;\n"
    "    int current = 0;\n"
    "    if (block < blocks) {\n"
    "        __builtin_xpu_gm2lm_v3(storage_x, X + block * 1024, 0, 20"
    "48);\n"
    "        if (!COPY)\n"
    "            __builtin_xpu_gm2lm_v3(storage_f, F + (block / groups"
    ") * 64, 0, 256);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    for (; block < blocks; block += step) {\n"
    "        const int offset = block * 1024;\n"
    "        unsigned int* xb = storage_x + current * 512;\n"
    "        float* fb = storage_f + current * 64;\n"
    "        const int next = block + step;\n"
    "        if (next < blocks) {\n"
    "            __builtin_xpu_gm2lm_v3(storage_x + (1 - current) * 51"
    "2,\n"
    "                                  X + next * 1024, 0, 2048);\n"
    "            if (!COPY)\n"
    "                __builtin_xpu_gm2lm_v3(storage_f + (1 - current) "
    "* 64,\n"
    "                                      F + (next / groups) * 64, 0"
    ", 256);\n"
    "        }\n"
    "        if (!COPY) {\n"
    "            #pragma clang loop vectorize(disable)\n"
    "            for (int pair = 0; pair < 32; ++pair) {\n"
    "                fc[pair] = fb[pair * 2];\n"
    "                fs[pair] = fb[pair * 2 + 1];\n"
    "            }\n"
    "            for (int head = 0; head < 16; ++head) {\n"
    "                for (int pair = 0; pair < 32; pair += 16) {\n"
    "                    const int index = head * 32 + pair;\n"
    "                    f32v c = *((f32v*)(fc + pair));\n"
    "                    f32v s = *((f32v*)(fs + pair));\n"
    "                    u32v packed = *((u32v*)(xb + index));\n"
    "                    bits195 real, imag;\n"
    "                    real.u = packed << 16;\n"
    "                    imag.u = packed & 0xffff0000u;\n"
    "                    f32v ac = real.f * c;\n"
    "                    f32v bs = imag.f * s;\n"
    "                    f32v bc = imag.f * c;\n"
    "                    f32v as = real.f * s;\n"
    '                    __asm__("" : "+v"(ac), "+v"(bs), "+v"(bc), "+'
    'v"(as));\n'
    "                    if (INV) {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac + b"
    "s) | (encode195(bc - as) << 16);\n"
    "                    } else {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac - b"
    "s) | (encode195(bc + as) << 16);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + offset, xb, 0, 2048);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        current = 1 - current;\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_PIPELINE195, flags=("-ffp-contract=off",))
def _device_pipeline195(X, F, Out, T, H, INV, COPY, pid, np):
    pass


@triton.jit(do_not_specialize=["X", "F", "Out"])
def _entry_pipeline195(X, F, Out, META: tl.constexpr):
    tle.raw.call(
        _device_pipeline195,
        (
            X,
            F,
            Out,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _pipeline_launch(x, freqs_cis, inverse, destination=None, copy=False):
    t, h, d = x.shape
    out = x if destination is None else destination
    _entry_pipeline195.run(
        x,
        torch.view_as_real(freqs_cis),
        out,
        (t, h, int(inverse), int(copy)),
        grid=(min(12, (t + 64 - 1) // 64),),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


def _pipeline_apply195(x, freqs_cis, positions, inverse):
    if (
        x.ndim == 3
        and x.shape[0] >= 4096
        and (x.shape[1] == 16)
        and _small_eligible(x, freqs_cis, positions)
    ):
        return _pipeline_launch(x, freqs_cis, inverse)
    return _recommended_apply(x, freqs_cis, positions, inverse)


RAW_ENCODE195 = (
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits195 { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode195(f32v value) "
    "{\n"
    "    bits195 bits;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u;\n"
    "    u32v word = (original + 32768u) >> 16;\n"
    "    return word;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_encode195(\n'
    "    gu16* X, gf32* F, gu16* O,\n"
    "    int T, int H, int INV, int COPY, int pid, int np) {\n"
    "    unsigned int storage_x[1024] __attribute__((aligned(64)));\n"
    "    float storage_f[128] __attribute__((aligned(64)));\n"
    "    float fc[32] __attribute__((aligned(64)));\n"
    "    float fs[32] __attribute__((aligned(64)));\n"
    "    const int core = __builtin_xpu_core_id();\n"
    "    const int groups = H / 16;\n"
    "    const int blocks = T * groups;\n"
    "    int block = pid * 64 + core;\n"
    "    const int step = np * 64;\n"
    "    int current = 0;\n"
    "    if (block < blocks) {\n"
    "        __builtin_xpu_gm2lm_v3(storage_x, X + block * 1024, 0, 20"
    "48);\n"
    "        if (!COPY)\n"
    "            __builtin_xpu_gm2lm_v3(storage_f, F + (block / groups"
    ") * 64, 0, 256);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    for (; block < blocks; block += step) {\n"
    "        const int offset = block * 1024;\n"
    "        unsigned int* xb = storage_x + current * 512;\n"
    "        float* fb = storage_f + current * 64;\n"
    "        const int next = block + step;\n"
    "        if (next < blocks) {\n"
    "            __builtin_xpu_gm2lm_v3(storage_x + (1 - current) * 51"
    "2,\n"
    "                                  X + next * 1024, 0, 2048);\n"
    "            if (!COPY)\n"
    "                __builtin_xpu_gm2lm_v3(storage_f + (1 - current) "
    "* 64,\n"
    "                                      F + (next / groups) * 64, 0"
    ", 256);\n"
    "        }\n"
    "        if (!COPY) {\n"
    "            #pragma clang loop vectorize(disable)\n"
    "            for (int pair = 0; pair < 32; ++pair) {\n"
    "                fc[pair] = fb[pair * 2];\n"
    "                fs[pair] = fb[pair * 2 + 1];\n"
    "            }\n"
    "            for (int head = 0; head < 16; ++head) {\n"
    "                for (int pair = 0; pair < 32; pair += 16) {\n"
    "                    const int index = head * 32 + pair;\n"
    "                    f32v c = *((f32v*)(fc + pair));\n"
    "                    f32v s = *((f32v*)(fs + pair));\n"
    "                    u32v packed = *((u32v*)(xb + index));\n"
    "                    bits195 real, imag;\n"
    "                    real.u = packed << 16;\n"
    "                    imag.u = packed & 0xffff0000u;\n"
    "                    f32v ac = real.f * c;\n"
    "                    f32v bs = imag.f * s;\n"
    "                    f32v bc = imag.f * c;\n"
    "                    f32v as = real.f * s;\n"
    '                    __asm__("" : "+v"(ac), "+v"(bs), "+v"(bc), "+'
    'v"(as));\n'
    "                    if (INV) {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac + b"
    "s) | (encode195(bc - as) << 16);\n"
    "                    } else {\n"
    "                        *((u32v*)(xb + index)) = encode195(ac - b"
    "s) | (encode195(bc + as) << 16);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O + offset, xb, 0, 2048);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        current = 1 - current;\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_ENCODE195, flags=("-ffp-contract=off",))
def _device_encode195(X, F, Out, T, H, INV, COPY, pid, np):
    pass


@triton.jit(do_not_specialize=["X", "F"])
def _entry_encode195(X, F, META: tl.constexpr):
    tle.raw.call(
        _device_encode195,
        (
            X,
            F,
            X,
            tl.full((), META[0], tl.int32),
            tl.full((), META[1], tl.int32),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    if (
        x.ndim == 3
        and x.shape[0] >= 4096
        and (x.shape[1] == 16)
        and _small_eligible(x, freqs_cis, positions)
    ):
        t = x.shape[0]
        _entry_encode195.run(
            x,
            torch.view_as_real(freqs_cis),
            (t, 16, int(inverse), 0),
            grid=(min(12, (t + 64 - 1) // 64),),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return x
    return _recommended_apply(x, freqs_cis, positions, inverse)


__all__ = ["apply_rotary_emb"]
