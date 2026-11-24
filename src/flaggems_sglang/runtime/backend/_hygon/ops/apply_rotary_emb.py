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
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.language import bfloat16 as bf16
from triton.language import float32 as fp32


@triton.jit
def _rope(X, F, P, Out, META: tl.constexpr):
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
        tl.store(X + oi, real, valid)
        tl.store(X + oi + 1, imag, valid)
    else:
        tl.store(Out + oi, real, valid)
        tl.store(Out + oi + 1, imag, valid)


def _fallback(x, freqs_cis, positions, inverse):
    if x.numel() == 0:
        return x
    shape = x.shape
    h = shape[1] if len(shape) == 3 else 1
    d = shape[-1]
    out = (
        None
        if x.is_contiguous()
        else torch.empty(shape, dtype=x.dtype, device=x.device)
    )
    fstrides = freqs_cis.stride()
    f = (
        freqs_cis.view(torch.float32)
        if freqs_cis.dtype == torch.complex64 and fstrides[-1] == 1
        else torch.view_as_real(freqs_cis)
    )
    block = 256
    nb = (h * d // 2 + block - 1) // block
    strides = x.stride()
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
        inverse,
        block,
        nb,
    )
    _rope.run(
        x,
        f,
        positions,
        out,
        meta,
        grid=(shape[0] * nb,),
        warmup=False,
        enable_fp_fusion=False,
    )
    return x if out is None else out


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
    block = pid
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


def _baseline_apply(x, freqs_cis, positions, inverse):
    if x.numel() >= 1048576:
        shape = x.shape
        t = shape[0]
        h = shape[1] if len(shape) == 3 else 1
        d = shape[-1]
        if (
            t <= 32768
            and h * d <= 4096
            and x.is_contiguous()
            and (freqs_cis.stride(1) == 1)
            and (h & h - 1 == 0)
            and (d & d - 1 == 0)
        ):
            f = torch.view_as_real(freqs_cis)
            bt = 4 if h * d <= 1024 and d <= 128 else 1
            meta = (
                t,
                h,
                d,
                f.stride(0),
                positions.stride(0) if positions is not None else 0,
                freqs_cis.shape[0],
                inverse,
                bt,
            )
            _rows.run(
                x,
                f,
                positions,
                meta,
                grid=((t + bt - 1) // bt,),
                warmup=False,
                enable_fp_fusion=False,
            )
            return x
    return _fallback(x, freqs_cis, positions, inverse)


@triton.jit
def _dense_rope(X, F, META: tl.constexpr):
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


@triton.jit
def _dense_rows(X, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SF0: tl.constexpr = D
    SP: tl.constexpr = 0
    FR: tl.constexpr = T
    INV: tl.constexpr = META[3]
    BT: tl.constexpr = 4 if H * D <= 1024 and D <= 128 else 1
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
    block = pid
    token = block * BT + tl.arange(0, BT)
    xp = X + token[:, None] * H * D + i[None, :]
    if T % BT == 0:
        values = tl.load(xp).to(tl.float32)
    else:
        values = tl.load(xp, token[:, None] < T, 0).to(tl.float32)
    row = token
    if None is not None:
        row = tl.load(None + token * SP, token < T, 0).to(tl.int64)
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
def _dense_rope_typed_frequency(X, F: "*fp32", META: tl.constexpr):
    _dense_rope(X, F, META)


@triton.jit
def _dense_rows_typed_frequency(X, F: "*fp32", META: tl.constexpr):
    _dense_rows(X, F, META)


@g.jit
def _forward_pair(a, b, c, s):
    return (a * c - b * s, b * c + a * s)


@g.jit
def _inverse_pair(a, b, c, s):
    return (a * c + b * s, b * c - a * s)


@g.jit
def _wave_frequency_1d(X, F: "*fp32", META: gl.constexpr):
    T: gl.constexpr = META[0]
    H: gl.constexpr = META[1]
    D: gl.constexpr = META[2]
    INV: gl.constexpr = META[3]
    gl.static_assert(H == 16 and D == 64)
    gl.static_assert(X.dtype.element_ty == gl.bfloat16)
    X_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        [[1], [2], [4], [2048]],
        [[8], [16], [32], [64], [128], [256]],
        [[512], [1024]],
        [],
        [4096],
    )
    index = gl.arange(0, 4096, layout=X_LAYOUT)
    token = gl.program_id(0) * 4 + index // 1024
    column = index % 1024
    xp = X + gl.program_id(0) * 4096 + index
    fp = (
        F
        + gl.program_id(0) * 256
        + (index // 1024 * 64 + column // 8 % 8 * 8)
        + column % 8
    )
    if T % 4 == 0:
        values = gl.load(xp).to(gl.float32)
        freqs = gl.load(fp)
    else:
        valid = token < T
        values = gl.load(xp, valid, 0).to(gl.float32)
        freqs = gl.load(fp, valid, 0)
    if INV:
        (out,) = gl.map_elementwise(_inverse_pair, values, freqs, pack=2)
    else:
        (out,) = gl.map_elementwise(_forward_pair, values, freqs, pack=2)
    if T % 4 == 0:
        bits = out.to(gl.uint32, bitcast=True)
        rounded = bits + 32767 + ((bits >> 16) & 1)
        rounded = gl.where(out == out, rounded, 0x7FFF0000)
        packed = gl.inline_asm_elementwise(
            "v_perm_b32 $0, $2, $1, $3",
            constraints="=v,v,v,s,s",
            args=[rounded, 0x07060302],
            dtype=gl.bfloat16,
            is_pure=True,
            pack=2,
        )
        gl.store(xp, packed)
    else:
        gl.store(xp, out, token < T)


@triton.jit
def _dense_rope_current_bf16_x(X: "*bf16", F: "*fp32", META: tl.constexpr):
    _dense_rope_typed_frequency(X, F, META)


@triton.jit
def _dense_rows_current_bf16_x(X: "*bf16", F: "*fp32", META: tl.constexpr):
    _dense_rows_typed_frequency(X, F, META)


@g.jit
def _wave_frequency_current_bf16_x(X: "*bf16", F: "*fp32", META: gl.constexpr):
    _wave_frequency_1d(X, F, META)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    if (
        x.numel() > 0
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
            if (
                4 <= t <= 32768
                and x.dtype == torch.bfloat16
                and h == 16
                and d == 64
                and (t % 4 == 0 or t >= 1024)
            ):
                if t % 4 == 0:
                    _wave_frequency_current_bf16_x.run(
                        x,
                        f,
                        meta,
                        grid=(t // 4,),
                        warmup=False,
                        enable_fp_fusion=False,
                    )
                else:
                    full_t = t // 4 * 4
                    _wave_frequency_current_bf16_x.run(
                        x,
                        f,
                        (full_t, h, d, inverse),
                        grid=(full_t // 4,),
                        warmup=False,
                        enable_fp_fusion=False,
                    )
                    _dense_rope_current_bf16_x.run(
                        x[full_t:],
                        f[full_t:],
                        (t - full_t, h, d, inverse),
                        grid=((t - full_t) * ((h * d // 2 + 255) // 256),),
                        warmup=False,
                        enable_fp_fusion=False,
                    )
                return x
            if (
                t * h * d >= 1048576
                and t <= 32768
                and (h * d <= 4096)
                and (h & h - 1 == 0)
                and (d & d - 1 == 0)
            ):
                bt = 4 if h * d <= 1024 and d <= 128 else 1
                (
                    _dense_rows_current_bf16_x
                    if x.dtype == torch.bfloat16
                    else _dense_rows_typed_frequency
                ).run(
                    x if x.dtype == torch.bfloat16 else x,
                    f,
                    meta,
                    grid=((t + bt - 1) // bt,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
                return x
            nb = (h * d // 2 + 255) // 256
            (
                _dense_rope_current_bf16_x
                if x.dtype == torch.bfloat16
                else _dense_rope_typed_frequency
            ).run(
                x if x.dtype == torch.bfloat16 else x,
                f,
                meta,
                grid=(t * nb,),
                warmup=False,
                enable_fp_fusion=False,
            )
            return x
    return _baseline_apply(x, freqs_cis, positions, inverse)


__all__ = ["apply_rotary_emb"]
