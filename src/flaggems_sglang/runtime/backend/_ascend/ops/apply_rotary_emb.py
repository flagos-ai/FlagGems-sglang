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
from triton.language import bfloat16 as bf16
from triton.language import float32 as fp32
from triton.language import int32 as i32
from triton.language import int64 as i64
from triton.language import uint32 as u32


@triton.jit
def _fallback_rope(
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
    NP: tl.constexpr,
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
    for task in range(pid, T * NB, NP):
        token = task // NB
        block = task % NB
        i = block * (B // 2) + tl.arange(0, B // 2)
        h = i // (D // 2)
        d = i % (D // 2)
        valid = i < H * D // 2
        row = token
        if HAS_P:
            if not WIDE and FR <= 2147483647 and SF0 <= 2147483647:
                row = tl.load(P + token * SP).to(tl.int32)
                row = row + ((row >> 31) & FR)
            else:
                row = tl.load(P + token * SP).to(tl.int64)
                row = tl.where(row < 0, row + FR, row)
        base = token * SX0 + h * SX1
        a = tl.load(X + base + 2 * d * SX2, valid, 0).to(tl.float32)
        b = tl.load(X + base + (2 * d + 1) * SX2, valid, 0).to(tl.float32)
        c = tl.load(F + row * SF0 + d * SF1, valid, 0).to(tl.float32)
        s = tl.load(F + row * SF0 + d * SF1 + 1, valid, 0).to(tl.float32)
        if INV:
            real = a * c + 1.0 * b * s
            imag = b * c + -1.0 * a * s
        else:
            real = a * c + -1.0 * b * s
            imag = b * c + 1.0 * a * s
        out = tl.reshape(tl.join(real, imag), (B,))
        j = block * B + tl.arange(0, B)
        tl.store(O + token * H * D + j, out, j < H * D)


def _fallback(x, freqs_cis, positions, inverse, shape):
    t = shape[0]
    h = shape[1] if len(shape) == 3 else 1
    d = shape[-1]
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    f = torch.view_as_real(freqs_cis)
    p = positions if positions is not None else x
    sp = positions.stride(0) if positions is not None else 0
    sh = x.stride(1) if len(shape) == 3 else 0
    block = 1024
    nb = (h * d + block - 1) // block
    np = min(t * nb, 32768)
    _fallback_rope[np,](
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
        f.stride(0),
        f.stride(1),
        sp,
        freqs_cis.shape[0],
        positions is not None,
        inverse,
        block,
        nb,
        np,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _rows(X, F, P, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SF0: tl.constexpr = META[3]
    SP: tl.constexpr = META[4]
    FR: tl.constexpr = META[5]
    INV: tl.constexpr = META[6]
    BT: tl.constexpr = META[7]
    NP: tl.constexpr = META[8]
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (FR - 1) * SF0 + D - 1 > 2147483647
        or (T - 1) * SP > 2147483647
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    i = tl.arange(0, H * D)
    d = tl.arange(0, D)
    for block in range(pid, tl.cdiv(T, BT), NP):
        token = block * BT + tl.arange(0, BT)
        xp = X + token[:, None] * H * D + i[None, :]
        if T % BT == 0:
            values = tl.load(xp).to(tl.float32)
        else:
            values = tl.load(xp, token[:, None] < T, 0).to(tl.float32)
        row = token
        if P is not None:
            row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
            row = tl.where(row < 0, row + FR, row)
        fp = F + row[:, None] * SF0 + d[None, :]
        if T % BT == 0:
            freqs = tl.load(fp)
        else:
            freqs = tl.load(fp, token[:, None] < T, 0)
        a, b = tl.split(tl.reshape(values, (BT, H * D // 2, 2)))
        c, s = tl.split(tl.reshape(freqs, (BT, D // 2, 2)))
        c = tl.reshape(
            tl.broadcast_to(c[:, None, :], (BT, H, D // 2)), (BT, H * D // 2)
        )
        s = tl.reshape(
            tl.broadcast_to(s[:, None, :], (BT, H, D // 2)), (BT, H * D // 2)
        )
        if INV:
            real = a * c + b * s
            imag = b * c - a * s
        else:
            real = a * c - b * s
            imag = b * c + a * s
        out = tl.reshape(tl.join(real, imag), (BT, H * D))
        if T % BT == 0:
            tl.store(xp, out)
        else:
            tl.store(xp, out, token[:, None] < T)


@triton.jit
def _rows_current_bf16_x(X: "*bf16", F: "*fp32", P, META: tl.constexpr):
    _rows_typed_frequency(X, F, P, META)


@triton.jit
def _rows_current_bf16_p32(
    X: "*bf16", F: "*fp32", P: "*i32", META: tl.constexpr
):
    _rows_current_bf16_x(X, F, P, META)


@triton.jit
def _rows_current_bf16_p64(
    X: "*bf16", F: "*fp32", P: "*i64", META: tl.constexpr
):
    _rows_current_bf16_x(X, F, P, META)


def _unpacked(x, freqs_cis, positions, inverse, shape):
    if x.numel() == 0:
        return x
    t = shape[0]
    h = shape[1] if len(shape) == 3 else 1
    d = shape[-1]
    if (
        d <= 128
        and h * d <= 4096
        and x.is_contiguous()
        and (freqs_cis.stride(1) == 1)
        and (h & h - 1 == 0)
        and (d & d - 1 == 0)
    ):
        if freqs_cis.dtype == torch.complex64:
            f = freqs_cis.data_ptr()
            bt = min(16, max(1, 4096 // (h * d)))
            np = min((t + bt - 1) // bt, 32768)
            meta = (
                t,
                h,
                d,
                freqs_cis.stride(0) * 2,
                positions.stride(0) if positions is not None else 0,
                freqs_cis.shape[0],
                inverse,
                bt,
                np,
            )
            if x.dtype == torch.bfloat16:
                if positions is not None and positions.dtype == torch.int32:
                    _rows_current_bf16_p32.run(
                        x.data_ptr(),
                        f,
                        positions.data_ptr(),
                        meta,
                        grid=(np,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                elif positions is not None and positions.dtype == torch.int64:
                    _rows_current_bf16_p64.run(
                        x.data_ptr(),
                        f,
                        positions.data_ptr(),
                        meta,
                        grid=(np,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                else:
                    _rows_current_bf16_x.run(
                        x.data_ptr(),
                        f,
                        positions,
                        meta,
                        grid=(np,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
            else:
                _rows_typed_frequency.run(
                    x,
                    f,
                    positions,
                    meta,
                    grid=(np,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return x
        f = (
            freqs_cis.view(torch.float32)
            if freqs_cis.dtype == torch.complex64
            else torch.view_as_real(freqs_cis)
        )
        bt = min(16, max(1, 4096 // (h * d)))
        np = min((t + bt - 1) // bt, 32768)
        meta = (
            t,
            h,
            d,
            f.stride(0),
            positions.stride(0) if positions is not None else 0,
            freqs_cis.shape[0],
            inverse,
            bt,
            np,
        )
        _rows.run(
            x,
            f,
            positions,
            meta,
            grid=(np,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return x
    return _fallback(x, freqs_cis, positions, inverse, shape)


@triton.jit
def _encode(value):
    return (
        value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    )


@triton.jit
def _packed_rows(X, F, P, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SF0: tl.constexpr = META[3]
    SP: tl.constexpr = META[4]
    FR: tl.constexpr = META[5]
    INV: tl.constexpr = META[6]
    BT: tl.constexpr = META[7]
    NP: tl.constexpr = META[8]
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (FR - 1) * SF0 + D - 1 > 2147483647
        or (T - 1) * SP > 2147483647
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    i = tl.arange(0, H * D // 2)
    d = tl.arange(0, D)
    for block in range(pid, tl.cdiv(T, BT), NP):
        token = block * BT + tl.arange(0, BT)
        xp = X + token[:, None] * (H * D // 2) + i[None, :]
        row = token
        if P is not None:
            row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
            row = tl.where(row < 0, row + FR, row)
        fp = F + row[:, None] * SF0 + d[None, :]
        if T % BT == 0:
            bits = tl.load(xp)
            freqs = tl.load(fp)
        else:
            bits = tl.load(xp, token[:, None] < T, 0)
            freqs = tl.load(fp, token[:, None] < T, 0)
        a = (bits << 16).to(tl.float32, bitcast=True)
        b = (bits & 4294901760).to(tl.float32, bitcast=True)
        c, s = tl.split(tl.reshape(freqs, (BT, D // 2, 2)))
        c = tl.reshape(
            tl.broadcast_to(c[:, None, :], (BT, H, D // 2)), (BT, H * D // 2)
        )
        s = tl.reshape(
            tl.broadcast_to(s[:, None, :], (BT, H, D // 2)), (BT, H * D // 2)
        )
        if INV:
            real = a * c + b * s
            imag = b * c - a * s
        else:
            real = a * c - b * s
            imag = b * c + a * s
        out = _encode(real) | (
            imag.to(tl.bfloat16).to(tl.float32).to(tl.int32, bitcast=True)
            & -65536
        )
        if T % BT == 0:
            tl.store(xp, out)
        else:
            tl.store(xp, out, token[:, None] < T)


@triton.jit
def _rows_typed_frequency(X, F: "*fp32", P, META: tl.constexpr):
    _rows(X, F, P, META)


@triton.jit
def _packed_rows_typed_frequency(X, F: "*fp32", P, META: tl.constexpr):
    _packed_rows(X, F, P, META)


@triton.jit
def _packed_rows_current_x(X: "*u32", F: "*fp32", P, META: tl.constexpr):
    _packed_rows_typed_frequency(X, F, P, META)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    shape = x.shape
    if x.numel() == 0:
        return x
    t = shape[0]
    h = shape[1] if len(shape) == 3 else 1
    d = shape[-1]
    if (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and (x.storage_offset() % 2 == 0)
        and (h * d <= 4096)
        and (freqs_cis.stride(1) == 1)
        and (h & h - 1 == 0)
        and (d & d - 1 == 0)
    ):
        if freqs_cis.dtype == torch.complex64:
            f = freqs_cis.data_ptr()
            budget = (
                8192
                if t >= 16384
                and d <= 128
                and (h * d >= 1024)
                and (positions is None)
                and freqs_cis.is_contiguous()
                and (x.data_ptr() % 16 == 0)
                and (freqs_cis.data_ptr() % 16 == 0)
                and (x.numel() + 8192 <= 2147483647)
                and (2 * freqs_cis.numel() + 8192 <= 2147483647)
                else 4096
            )
            bt = min(16, max(1, budget // (h * d)))
            np = min((t + bt - 1) // bt, 32768)
            meta = (
                t,
                h,
                d,
                freqs_cis.stride(0) * 2,
                positions.stride(0) if positions is not None else 0,
                freqs_cis.shape[0],
                inverse,
                bt,
                np,
            )
            _packed_rows_current_x.run(
                x.data_ptr(),
                f,
                positions,
                meta,
                grid=(np,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return x
        f = (
            freqs_cis.view(torch.float32)
            if freqs_cis.dtype == torch.complex64
            else torch.view_as_real(freqs_cis)
        )
        budget = (
            8192
            if t >= 16384
            and d <= 128
            and (h * d >= 1024)
            and (positions is None)
            and freqs_cis.is_contiguous()
            and (x.data_ptr() % 16 == 0)
            and (freqs_cis.data_ptr() % 16 == 0)
            and (x.numel() + 8192 <= 2147483647)
            and (2 * freqs_cis.numel() + 8192 <= 2147483647)
            else 4096
        )
        bt = min(16, max(1, budget // (h * d)))
        np = min((t + bt - 1) // bt, 32768)
        meta = (
            t,
            h,
            d,
            f.stride(0),
            positions.stride(0) if positions is not None else 0,
            freqs_cis.shape[0],
            inverse,
            bt,
            np,
        )
        _packed_rows.run(
            triton.reinterpret(x, tl.uint32),
            f,
            positions,
            meta,
            grid=(np,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return x
    return _unpacked(x, freqs_cis, positions, inverse, shape)


__all__ = ["apply_rotary_emb"]
