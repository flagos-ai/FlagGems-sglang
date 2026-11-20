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
    FLAT: tl.constexpr = META[13]
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    if FLAT:
        i = pid * B + tl.arange(0, B)
        token = i // (H * D // 2)
        col = i % (H * D // 2)
        valid = i < T * H * D // 2
    else:
        token = pid // NB
        i = pid % NB * B + tl.arange(0, B)
        col = i
        valid = i < H * D // 2
    h = col // (D // 2)
    d = col % (D // 2)
    row = token
    if P is not None:
        row = tl.load(P + token * SP, token < T, 0).to(tl.int64)
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
    oi = 2 * i if FLAT else token * H * D + 2 * i
    if Out is None:
        tl.store(X + oi, real, valid)
        tl.store(X + oi + 1, imag, valid)
    else:
        out = tl.reshape(tl.join(real, imag), (2 * B,))
        j = tl.arange(0, 2 * B)
        output_base = pid * 2 * B if FLAT else token * H * D + pid % NB * 2 * B
        output_valid = (
            output_base + j < T * H * D
            if FLAT
            else pid % NB * 2 * B + j < H * D
        )
        tl.store(Out + output_base + j, out, output_valid)


def _fallback(x, freqs_cis, positions, inverse):
    if x.numel() == 0:
        return x
    h = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    out = (
        None
        if x.is_contiguous()
        else torch.empty(x.shape, dtype=x.dtype, device=x.device)
    )
    f = torch.view_as_real(freqs_cis)
    flat = x.numel() >= 1048576
    block = 1024 if flat else 256
    nb = (h * d // 2 + block - 1) // block
    tasks = (x.numel() // 2 + block - 1) // block if flat else x.shape[0] * nb
    meta = (
        h,
        d,
        x.stride(0),
        x.stride(1) if x.ndim == 3 else 0,
        x.stride(-1),
        f.stride(0),
        f.stride(1),
        positions.stride(0) if positions is not None else 0,
        freqs_cis.shape[0],
        x.shape[0],
        inverse,
        block,
        nb,
        flat,
    )
    _rope.run(
        x,
        f,
        positions,
        out,
        meta,
        grid=(tasks,),
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
        if not WIDE and FR <= 2147483647 and SF0 <= 2147483647:
            row = tl.load(P + token * SP, token < T, 0).to(tl.int32)
            row = row + tl.where(row < 0, FR, 0)
        else:
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
    if x.numel() == 0:
        return _fallback(x, freqs_cis, positions, inverse)
    t = x.shape[0]
    h = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    if (
        h * d <= 4096
        and x.is_contiguous()
        and (freqs_cis.stride(1) == 1)
        and (h & h - 1 == 0)
        and (d & d - 1 == 0)
    ):
        f = torch.view_as_real(freqs_cis)
        bt = min(16, max(1, 4096 // (h * d)))
        np = (t + bt - 1) // bt
        if np > 32768:
            return _fallback(x, freqs_cis, positions, inverse)
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
            grid=(np,),
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
    B: tl.constexpr = 1024 if T * H * D >= 1048576 else 256
    NB: tl.constexpr = (H * D // 2 + B - 1) // B
    FLAT: tl.constexpr = T * H * D >= 1048576
    WIDE: tl.constexpr = (
        T * H * D > 2147483647
        or (T - 1) * SX0 + (H - 1) * SX1 + (D - 1) * SX2 > 2147483647
        or (FR - 1) * SF0 + (D // 2 - 1) * SF1 + 1 > 2147483647
        or ((T - 1) * SP > 2147483647)
    )
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
    if FLAT:
        i = pid * B + tl.arange(0, B)
        token = i // (H * D // 2)
        col = i % (H * D // 2)
        valid = i < T * H * D // 2
    else:
        token = pid // NB
        i = pid % NB * B + tl.arange(0, B)
        col = i
        valid = i < H * D // 2
    h = col // (D // 2)
    d = col % (D // 2)
    row = token
    if None is not None:
        row = tl.load(None + token * SP, token < T, 0).to(tl.int64)
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
    oi = 2 * i if FLAT else token * H * D + 2 * i
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
    BT: tl.constexpr = 16 if H * D <= 256 else 4096 // (H * D)
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


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    if (
        x.numel() > 0
        and x.ndim in (2, 3)
        and (positions is None)
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
    ):
        t = x.shape[0]
        h = x.shape[1] if x.ndim == 3 else 1
        d = x.shape[-1]
        if (
            d % 2 == 0
            and freqs_cis.shape[-1] * 2 == d
            and ((t + 16) * h * d + 2048 <= 2147483647)
        ):
            if freqs_cis.dtype == torch.complex64:
                f = freqs_cis.view(torch.float32)
                meta = (t, h, d, inverse)
                if (
                    t * h * d >= 1048576
                    and h * d <= 4096
                    and (h & h - 1 == 0)
                    and (d & d - 1 == 0)
                ):
                    bt = 16 if h * d <= 256 else 4096 // (h * d)
                    np = (t + bt - 1) // bt
                    if np <= 32768:
                        _dense_rows_typed_frequency.run(
                            x,
                            f,
                            meta,
                            grid=(np,),
                            warmup=False,
                            enable_fp_fusion=False,
                        )
                        return x
                flat = t * h * d >= 1048576
                block = 1024 if flat else 256
                nb = (h * d // 2 + block - 1) // block
                tasks = (
                    (t * h * d // 2 + block - 1) // block if flat else t * nb
                )
                _dense_rope_typed_frequency.run(
                    x,
                    f,
                    meta,
                    grid=(tasks,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
                return x
            f = (
                freqs_cis.view(torch.float32)
                if freqs_cis.dtype == torch.complex64
                else torch.view_as_real(freqs_cis)
            )
            meta = (t, h, d, inverse)
            if (
                t * h * d >= 1048576
                and h * d <= 4096
                and (h & h - 1 == 0)
                and (d & d - 1 == 0)
            ):
                bt = 16 if h * d <= 256 else 4096 // (h * d)
                np = (t + bt - 1) // bt
                if np <= 32768:
                    _dense_rows.run(
                        x,
                        f,
                        meta,
                        grid=(np,),
                        warmup=False,
                        enable_fp_fusion=False,
                    )
                    return x
            flat = t * h * d >= 1048576
            block = 1024 if flat else 256
            nb = (h * d // 2 + block - 1) // block
            tasks = (t * h * d // 2 + block - 1) // block if flat else t * nb
            _dense_rope.run(
                x, f, meta, grid=(tasks,), warmup=False, enable_fp_fusion=False
            )
            return x
    return _baseline_apply(x, freqs_cis, positions, inverse)


__all__ = ["apply_rotary_emb"]
