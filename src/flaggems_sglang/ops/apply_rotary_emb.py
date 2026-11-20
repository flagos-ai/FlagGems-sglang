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

_CUDA_OUTPUT_JOIN195 = tl.constexpr(torch.version.hip is None)

if torch.version.hip is None:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def _dense_packed_head_owner195(
        X: "*bf16", F: "*fp32", META: gl.constexpr
    ):
        D: gl.constexpr = META[2]
        INV: gl.constexpr = META[3]
        Q: gl.constexpr = D // 4
        tile: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
        rows: gl.constexpr = gl.SliceLayout(1, tile)
        columns: gl.constexpr = gl.SliceLayout(0, tile)
        token = gl.program_id(0)
        i = gl.arange(0, 256, layout=rows)
        component = gl.arange(0, 4, layout=columns)
        quad = i
        if Q > 8:
            quad = (
                (i & ~(8 + Q)) | ((i & 8) * (Q // 8)) | ((i & Q) // (Q // 8))
            )
        high = (i & 8) != 0
        offsets = token * 1024 + quad[:, None] * 4 + component[None, :]
        values = gl.load(X + offsets, cache_modifier=".cg")
        decoded = gl.inline_asm_elementwise(
            "shl.b32 $0, $2, 16; and.b32 $1, $2, 0xffff0000;",
            constraints="=r,=r,r",
            args=[values.to(gl.uint16, bitcast=True)],
            dtype=gl.uint32,
            is_pure=True,
            pack=2,
        )
        values = decoded.to(gl.float32, bitcast=True)
        a, b = gl.split(gl.reshape(values, (256, 2, 2)))
        frequency_component = gl.arange(0, 2, layout=columns)
        frequency_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2], [32, 1], [4, 1], [1, 0]
        )
        if D == 2:
            frequency_index = (
                token * D + i[:, None] * 0 + frequency_component[None, :]
            )
        else:
            frequency_pair = (quad % Q) * 2 + high.to(gl.int32)
            frequency_index = (
                token * D
                + 2 * frequency_pair[:, None]
                + frequency_component[None, :]
            )
        frequency = gl.convert_layout(
            gl.load(F + frequency_index), frequency_layout
        )
        if D == 2:
            c, s = gl.split(frequency)
            cosine = gl.join(c, c)
            sine = gl.join(s, s)
        else:
            frequency_high = gl.convert_layout(
                high, gl.SliceLayout(1, frequency_layout)
            )
            peer = gl.inline_asm_elementwise(
                "shfl.sync.bfly.b32 $0, $1, 8, 31, -1;",
                constraints="=f,f",
                args=[frequency],
                dtype=gl.float32,
                is_pure=True,
                pack=1,
            )
            first = gl.where(frequency_high[:, None], peer, frequency)
            second = gl.where(frequency_high[:, None], frequency, peer)
            c0, s0 = gl.split(first)
            c1, s1 = gl.split(second)
            cosine = gl.join(c0, c1)
            sine = gl.join(s0, s1)
        cosine = gl.convert_layout(cosine, a.type.layout)
        sine = gl.convert_layout(sine, a.type.layout)
        ac = a * cosine
        bs = b * sine
        bc = b * cosine
        ass = a * sine
        if INV:
            real = ac + bs
            imag = bc - ass
        else:
            real = ac - bs
            imag = bc + ass
        output = gl.reshape(gl.join(real, imag), (256, 4)).to(gl.bfloat16)
        gl.store(X + offsets, output, cache_modifier=".cs")

    @gluon.jit
    def _dense_element_owner195(X: "*bf16", F: "*fp32", META: gl.constexpr):
        D: gl.constexpr = META[2]
        INV: gl.constexpr = META[3]
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
        token = gl.program_id(0)
        element = gl.arange(0, 1024, layout=layout)
        odd = (element & 1) != 0
        own = gl.load(X + token * 1024 + element).to(gl.float32)
        frequency = gl.load(F + token * D + element % D)
        peer = gl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, 1, 31, -1;",
            constraints="=f,f",
            args=[own],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        peer_frequency = gl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, 1, 31, -1;",
            constraints="=f,f",
            args=[frequency],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        cosine = gl.where(odd, peer_frequency, frequency)
        sine = gl.where(odd, frequency, peer_frequency)
        primary = own * cosine
        secondary = peer * sine
        if INV:
            result = gl.where(odd, primary - secondary, primary + secondary)
        else:
            result = gl.where(odd, primary + secondary, primary - secondary)
        gl.store(X + token * 1024 + element, result, cache_modifier=".cs")


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
    FULL: tl.constexpr = Out is None and T >= 16384 and (H * D // 2 % B == 0)
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
    a = tl.load(X + base, None if FULL else valid, None if FULL else 0).to(
        tl.float32
    )
    b = tl.load(
        X + base + SX2, None if FULL else valid, None if FULL else 0
    ).to(tl.float32)
    c = tl.load(
        F + row * SF0 + d * SF1, None if FULL else valid, None if FULL else 0
    )
    s = tl.load(
        F + row * SF0 + d * SF1 + 1,
        None if FULL else valid,
        None if FULL else 0,
    )
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    if _CUDA_OUTPUT_JOIN195:
        output = tl.reshape(tl.join(real, imag), (2 * B,))
        component = tl.arange(0, 2 * B)
        output_base = token * H * D + pid % NB * 2 * B
        output_valid = pid % NB * 2 * B + component < 2 * (H * D // 2)
        if Out is None:
            tl.store(
                X + output_base + component,
                output,
                None if FULL else output_valid,
                cache_modifier=".cs" if FULL else "",
            )
        else:
            tl.store(
                Out + output_base + component,
                output,
                None if FULL else output_valid,
                cache_modifier=".cs" if FULL else "",
            )
    else:
        oi = token * H * D + 2 * i
        if Out is None:
            tl.store(
                X + oi,
                real,
                None if FULL else valid,
                cache_modifier=".cs" if FULL else "",
            )
            tl.store(
                X + oi + 1,
                imag,
                None if FULL else valid,
                cache_modifier=".cs" if FULL else "",
            )
        else:
            tl.store(
                Out + oi,
                real,
                None if FULL else valid,
                cache_modifier=".cs" if FULL else "",
            )
            tl.store(
                Out + oi + 1,
                imag,
                None if FULL else valid,
                cache_modifier=".cs" if FULL else "",
            )


@triton.jit
def _rope_typed_frequency(X, F: "*fp32", P, Out, META: tl.constexpr):
    _rope(X, F, P, Out, META)


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
    _rope_typed_frequency.run(
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
def _dense(X, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    INV: tl.constexpr = META[3]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = T
    B: tl.constexpr = 256
    NB: tl.constexpr = (H * D // 2 + 255) // 256
    FULL: tl.constexpr = None is None and T >= 16384 and (H * D // 2 % B == 0)
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
    a = tl.load(X + base, None if FULL else valid, None if FULL else 0).to(
        tl.float32
    )
    b = tl.load(
        X + base + SX2, None if FULL else valid, None if FULL else 0
    ).to(tl.float32)
    c = tl.load(
        F + row * SF0 + d * SF1, None if FULL else valid, None if FULL else 0
    )
    s = tl.load(
        F + row * SF0 + d * SF1 + 1,
        None if FULL else valid,
        None if FULL else 0,
    )
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if None is None:
        tl.store(
            X + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            X + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
    else:
        tl.store(
            None + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            None + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )


@triton.jit
def _dense_typed_frequency(X, F: "*fp32", META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    INV: tl.constexpr = META[3]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = T
    B: tl.constexpr = 256
    NB: tl.constexpr = (H * D // 2 + 255) // 256
    FULL: tl.constexpr = None is None and T >= 16384 and (H * D // 2 % B == 0)
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
    a = tl.load(X + base, None if FULL else valid, None if FULL else 0).to(
        tl.float32
    )
    b = tl.load(
        X + base + SX2, None if FULL else valid, None if FULL else 0
    ).to(tl.float32)
    c = tl.load(
        F + row * SF0 + d * SF1, None if FULL else valid, None if FULL else 0
    )
    s = tl.load(
        F + row * SF0 + d * SF1 + 1,
        None if FULL else valid,
        None if FULL else 0,
    )
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if None is None:
        tl.store(
            X + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            X + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
    else:
        tl.store(
            None + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            None + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )


@triton.jit
def _dense_x_pointer_bf16(X: "*bf16", F: "*fp32", META: tl.constexpr):
    _dense_typed_frequency(X, F, META)


@triton.jit
def _dense_typed_frequency_cta_pair(
    X, F: "*fp32", META: tl.constexpr, COLOCATE: tl.constexpr = False
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    INV: tl.constexpr = META[3]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = T
    B: tl.constexpr = 512 if COLOCATE else 256
    NB: tl.constexpr = (H * D // 2 + B - 1) // B
    FULL: tl.constexpr = None is None and T >= 16384 and (H * D // 2 % B == 0)
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
    lane = tl.arange(0, B)
    if COLOCATE:
        lane = (lane & -385) | ((lane & 128) << 1) | ((lane & 256) >> 1)
    i = pid % NB * B + lane
    h = i // (D // 2)
    d = i % (D // 2)
    valid = i < H * D // 2
    row = token
    if None is not None:
        row = tl.load(None + token * SP).to(tl.int64)
        row = tl.where(row < 0, row + FR, row)
    base = token * SX0 + h * SX1 + 2 * d * SX2
    a = tl.load(X + base, None if FULL else valid, None if FULL else 0).to(
        tl.float32
    )
    b = tl.load(
        X + base + SX2, None if FULL else valid, None if FULL else 0
    ).to(tl.float32)
    c = tl.load(
        F + row * SF0 + d * SF1, None if FULL else valid, None if FULL else 0
    )
    s = tl.load(
        F + row * SF0 + d * SF1 + 1,
        None if FULL else valid,
        None if FULL else 0,
    )
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if None is None:
        tl.store(
            X + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            X + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
    else:
        tl.store(
            None + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            None + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )


@triton.jit
def _dense_x_pointer_bf16_cta_colocated(
    X: "*bf16", F: "*fp32", META: tl.constexpr
):
    _dense_typed_frequency_cta_pair(X, F, META, True)


def apply_rotary_emb(x, freqs_cis, positions, inverse):
    n = x.numel()
    if n == 0:
        return x
    if positions is None and x.is_contiguous() and freqs_cis.is_contiguous():
        shape = x.shape
        t, d = (shape[0], shape[-1])
        h = shape[1] if len(shape) == 3 else 1
        if (
            d % 2 == 0
            and freqs_cis.shape[-1] * 2 == d
            and (n + 512 <= 2147483647)
            and (t * d + 512 <= 2147483647)
        ):
            f = (
                freqs_cis.view(torch.float32)
                if freqs_cis.dtype == torch.complex64
                else torch.view_as_real(freqs_cis)
            )
            if x.dtype == torch.bfloat16:
                if (
                    n >= 16 * 1024 * 1024
                    and torch.version.hip is not None
                    and x.storage_offset() % 2 == 0
                ):
                    _dense_hip_packed195.run(
                        x,
                        f,
                        (t, h, d, inverse),
                        grid=(t * ((h * d // 2 + 255) // 256),),
                        warmup=False,
                        enable_fp_fusion=False,
                    )
                    return x
                if (
                    t >= 16384
                    and h * d == 1024
                    and 128 % (d // 2) == 0
                    and x.device.type == "cuda"
                    and torch.version.cuda is not None
                    and torch.version.hip is None
                ):
                    _dense_packed_head_owner195.run(
                        x,
                        f,
                        (t, h, d, inverse),
                        grid=(t,),
                        warmup=False,
                        enable_fp_fusion=False,
                        num_warps=4,
                    )
                    return x
                dense = (
                    _dense_x_pointer_bf16_xcg
                    if torch.version.hip is not None
                    else _dense_x_pointer_bf16
                )
                dense.run(
                    x,
                    f,
                    (t, h, d, inverse),
                    grid=(t * ((h * d // 2 + 255) // 256),),
                    warmup=False,
                    enable_fp_fusion=False,
                )
                return x
            _dense_typed_frequency.run(
                x,
                f,
                (t, h, d, inverse),
                grid=(t * ((h * d // 2 + 255) // 256),),
                warmup=False,
                enable_fp_fusion=False,
            )
            return x
    return _fallback(x, freqs_cis, positions, inverse)


@triton.jit
def _dense_typed_frequency_xcg(X, F: "*fp32", META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    INV: tl.constexpr = META[3]
    SX0: tl.constexpr = H * D
    SX1: tl.constexpr = D
    SX2: tl.constexpr = 1
    SF0: tl.constexpr = D
    SF1: tl.constexpr = 2
    SP: tl.constexpr = 0
    FR: tl.constexpr = T
    B: tl.constexpr = 256
    NB: tl.constexpr = (H * D // 2 + 255) // 256
    FULL: tl.constexpr = None is None and T >= 16384 and (H * D // 2 % B == 0)
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
    a = tl.load(
        X + base,
        None if FULL else valid,
        None if FULL else 0,
        cache_modifier=".cg",
    ).to(tl.float32)
    b = tl.load(
        X + base + SX2,
        None if FULL else valid,
        None if FULL else 0,
        cache_modifier=".cg",
    ).to(tl.float32)
    c = tl.load(
        F + row * SF0 + d * SF1, None if FULL else valid, None if FULL else 0
    )
    s = tl.load(
        F + row * SF0 + d * SF1 + 1,
        None if FULL else valid,
        None if FULL else 0,
    )
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    oi = token * H * D + 2 * i
    if None is None:
        tl.store(
            X + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            X + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
    else:
        tl.store(
            None + oi,
            real,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )
        tl.store(
            None + oi + 1,
            imag,
            None if FULL else valid,
            cache_modifier=".cs" if FULL else "",
        )


@triton.jit
def _dense_x_pointer_bf16_xcg(X: "*bf16", F: "*fp32", META: tl.constexpr):
    _dense_typed_frequency_xcg(X, F, META)


@triton.jit
def _dense_hip_packed195(X: "*bf16", F: "*fp32", META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    INV: tl.constexpr = META[3]
    B: tl.constexpr = 256
    N: tl.constexpr = H * D // 2
    NB: tl.constexpr = (N + B - 1) // B
    FULL: tl.constexpr = T >= 16384 and N % B == 0
    pid = tl.program_id(0)
    token = pid // NB
    pair = pid % NB * B + tl.arange(0, B)
    valid = pair < N
    d = pair % (D // 2)
    packed = X.to(tl.pointer_type(tl.uint32))
    word = tl.load(
        packed + token * N + pair,
        None if FULL else valid,
        None if FULL else 0,
        cache_modifier=".cg",
    )
    a = (word << 16).to(tl.float32, bitcast=True)
    b = (word & 0xFFFF0000).to(tl.float32, bitcast=True)
    frequency = tl.load(
        F.to(tl.pointer_type(tl.uint64)) + token * (D // 2) + d,
        None if FULL else valid,
        None if FULL else 0,
    )
    c = frequency.to(tl.uint32).to(tl.float32, bitcast=True)
    s = (frequency >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    if INV:
        real = a * c + b * s
        imag = b * c - a * s
    else:
        real = a * c - b * s
        imag = b * c + a * s
    lo = real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi = imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(
        packed + token * N + pair,
        lo | (hi << 16),
        None if FULL else valid,
        cache_modifier=".cs" if FULL else "",
    )


__all__ = ["apply_rotary_emb"]
