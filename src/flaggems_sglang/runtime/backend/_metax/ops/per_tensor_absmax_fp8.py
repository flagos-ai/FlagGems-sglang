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
def _stride_native1k(
    X, Output, N: tl.constexpr, S: tl.constexpr, CACHE: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    if (tl.program_id(0).to(tl.int64) + 1) * 1024 <= N:
        x = tl.load(X + i * S, cache_modifier=CACHE)
    else:
        x = tl.load(X + i * S, i < N, other=0, cache_modifier=CACHE)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        v = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        v = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        v.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _reduce(
    X,
    P,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    CONTIG: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    FINAL: tl.constexpr,
    ADDR32: tl.constexpr = False,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (tl.program_id(0) + step.to(tl.int64) * G) * B + lane
        else:
            i = (tl.program_id(0) + step * G) * B + lane
        off = i
        if not CONTIG:
            if ADDR32:
                q = i.to(tl.int32)
                off = tl.full((B,), 0, tl.int32)
            else:
                q = i.to(tl.int64)
                off = tl.full((B,), 0, tl.int64)
            for d in tl.static_range(len(SH) - 1, -1, -1):
                if d == 0:
                    off += q * ST[d]
                else:
                    off += q % SH[d] * ST[d]
                    q = q // SH[d]
        x = tl.load(X + off, i < N, other=0).to(tl.float32)
        acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
    v = tl.max(acc, 0).to(tl.float32, bitcast=True)
    if FINAL:
        v = v / 448.0
    if FINAL:
        tl.store(P, v)
    else:
        bits = (v / 448.0).to(tl.uint32, bitcast=True)
        tl.atomic_max(P.to(tl.pointer_type(tl.uint32)), bits, sem="relaxed")


@triton.jit
def _packed(X, Output, N: tl.constexpr):
    B: tl.constexpr = 512 if N <= 512 else 1024 if N <= 1024 else 2048
    G: tl.constexpr = (N + B - 1) // B if N <= B * 512 else 512
    pid = tl.program_id(0)
    if N >= 2097152 and G >= 64 and (G % 64 == 0):
        pid = pid % 64 * (G // 64) + pid // 64
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (pid + step.to(tl.int64) * G) * B + lane
        else:
            i = (pid + step * G) * B + lane
        x = tl.load(X.to(tl.pointer_type(tl.int32)) + i, i < N, other=0)
        lo = x & 32767
        hi = x >> 16 & 32767
        acc = tl.maximum(acc, tl.maximum(lo, hi))
    bits = tl.max(acc, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (bits << 16).to(tl.float32, bitcast=True)
    else:
        maximum = (
            bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
    v = maximum / 448.0
    if G == 1:
        tl.store(Output, v)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)),
            v.to(tl.int32, bitcast=True),
            sem="relaxed",
        )


@triton.jit
def _rows2d(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    G: tl.constexpr,
):
    rows = tl.arange(0, BR)
    cols = tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.int32)
    for step in range((R + BR * G - 1) // (BR * G)):
        r = (step.to(tl.int64) * G + tl.program_id(0)) * BR + rows
        x = tl.load(
            X + r[:, None] * S + cols[None, :],
            (r[:, None] < R) & (cols[None, :] < C),
            other=0,
        )
        if X.dtype.element_ty == tl.float32:
            bits = x.to(tl.int32, bitcast=True) & 2147483647
        else:
            bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        acc = tl.maximum(acc, bits)
    v = tl.max(tl.max(acc, 1), 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    bits = (maximum / 448.0).to(tl.int32, bitcast=True)
    if G == 1:
        tl.store(Output.to(tl.pointer_type(tl.int32)), bits)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)), bits, sem="relaxed"
        )


@triton.jit
def _wide2d(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    G: tl.constexpr,
    NC: tl.constexpr,
):
    rows = tl.arange(0, BR)
    cols = tl.program_id(0) % NC * BC + tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.int32)
    for step in range((R + BR * G - 1) // (BR * G)):
        r = (step.to(tl.int64) * G + tl.program_id(0) // NC) * BR + rows
        x = tl.load(
            X + r[:, None] * S + cols[None, :],
            (r[:, None] < R) & (cols[None, :] < C),
            other=0,
        )
        if X.dtype.element_ty == tl.float32:
            bits = x.to(tl.int32, bitcast=True) & 2147483647
        else:
            bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        acc = tl.maximum(acc, bits)
    v = tl.max(tl.max(acc, 1), 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    bits = (maximum / 448.0).to(tl.int32, bitcast=True)
    if G * NC == 1:
        tl.store(Output.to(tl.pointer_type(tl.int32)), bits)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)), bits, sem="relaxed"
        )


def _layout(input, output_s):
    shape = []
    strides = []
    n = 1
    for stride, size in sorted(
        (
            (s, d)
            for d, s in zip(input.shape, input.stride())
            if d > 1 and s != 0
        )
    ):
        if (
            shape
            and stride % strides[-1] == 0
            and (stride <= strides[-1] * shape[-1])
        ):
            shape[-1] += (size - 1) * (stride // strides[-1])
        else:
            shape.append(size)
            strides.append(stride)
    for size in shape:
        n *= size
    if (
        input.dtype == torch.float32
        and len(shape) >= 3
        and (strides[0] == 1)
        and (257 <= shape[0] <= 1023)
        and (shape[0] % 2 == 1)
        and (n >= 1048576)
        and (3 * shape[0] <= 2 * strides[1] <= 4 * shape[0])
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = (2048 if c <= 767 else 4096) // (1 << (c - 1).bit_length())
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        _highdim_rect.run(
            input,
            output_s,
            c,
            r,
            pitch,
            tuple(shape[2:]),
            tuple(strides[2:]),
            br,
            1 << (c - 1).bit_length(),
            grid=(tiles,),
            num_warps=1,
            warmup=False,
        )
        return output_s
    if (
        input.dtype != torch.float32
        and len(shape) >= 2
        and (strides[0] == 1)
        and (3 * shape[0] <= 2 * strides[1] <= 4 * shape[0])
        and (
            257 <= shape[0] <= 1023
            and shape[0] % 2 == 1
            and (n >= 1048576)
            and (len(shape) >= 3 or shape[0] <= 383)
            or (
                len(shape) >= 3
                and 256 <= shape[0] <= 1024
                and (shape[0] % 2 == 0)
                and (strides[1] % 2 == 1)
                and (n >= 4194304)
            )
        )
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = 4096 // (1 << (c - 1).bit_length())
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        _highdim_rect.run(
            input,
            output_s,
            c,
            r,
            pitch,
            tuple(shape[2:]),
            tuple(strides[2:]),
            br,
            1 << (c - 1).bit_length(),
            grid=(tiles,),
            num_warps=1,
            warmup=False,
        )
        return output_s
    if (
        len(shape) >= 5
        and strides[0] == 1
        and (129 <= shape[0] <= 511)
        and (shape[0] % 2 == 1 or (n >= 1048576 and strides[1] % 2 == 1))
        and (n >= 1048576)
        and (
            strides[1]
            <= (
                2 * shape[0]
                if n >= 16777216 and shape[0] % 2 == 1
                else (
                    5 * shape[0] // 4 if shape[0] <= 255 else 9 * shape[0] // 8
                )
            )
        )
        and (shape[1] >= 8192 // strides[1])
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        if (
            input.dtype != torch.float32
            and n >= 16777216
            and (pitch % 2 == 1)
            and (br == 41)
            and (input.data_ptr() % 16 == 0)
            and all((s % 8 == 0 for s in strides[2:]))
        ):
            br -= 1
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else (
                    _physical_rows_noalign
                    if input.dtype == torch.float32
                    else _physical_rows
                )
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        3 <= len(shape) <= 4
        and strides[0] == 1
        and (258 <= shape[0] <= 510)
        and (shape[0] % 2 == 0)
        and (strides[1] % 2 == 1)
        and (n >= 4194304)
        and (strides[1] <= 9 * shape[0] // 8)
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else (
                    _physical_rows_noalign
                    if input.dtype == torch.float32 and n >= 4194304
                    else _physical_rows
                )
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        3 <= len(shape) <= 4
        and strides[0] == 1
        and (129 <= shape[0] <= 255)
        and (shape[0] % 2 == 1 or (n >= 4194304 and strides[1] % 2 == 1))
        and (n >= 1048576)
        and (
            strides[1]
            <= (5 * shape[0] // 4 if n >= 4194304 else 9 * shape[0] // 8)
        )
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        if (
            input.dtype != torch.float32
            and n >= 16777216
            and (pitch % 2 == 1)
            and (br == 41)
            and (input.data_ptr() % 16 == 0)
            and all((s % 8 == 0 for s in strides[2:]))
        ):
            br -= 1
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else (
                    _physical_rows_noalign
                    if input.dtype == torch.float32 and n >= 4194304
                    else _physical_rows
                )
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        3 <= len(shape) <= 4
        and strides[0] == 1
        and (257 <= shape[0] <= 511)
        and (shape[0] % 2 == 1)
        and (1048576 <= n < 4194304)
        and (strides[1] <= 9 * shape[0] // 8)
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else _physical_rows
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        len(shape) == 2
        and strides[0] == 1
        and (257 <= shape[0] <= 511)
        and (shape[0] % 2 == 1)
        and (1048576 <= n < 4194304)
        and (strides[1] <= 9 * shape[0] // 8)
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else _physical_rows
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        (
            3 <= len(shape) <= 4
            or (
                len(shape) == 2
                and input.dtype != torch.float32
                and (n >= 16777216)
                and (shape[0] <= 319 or (shape[0] >= 703 and n >= 25165824))
            )
        )
        and strides[0] == 1
        and (257 <= shape[0] <= 769)
        and (shape[0] <= 511 or n >= 16777216)
        and (shape[0] % 2 == 1)
        and (n >= 4194304)
        and (strides[1] <= min(4096, 9 * shape[0] // 8))
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                (16384 if c >= 703 and n >= 25165824 else 8192)
                if input.dtype != torch.float32 and n >= 16777216
                else 4096
            )
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            (
                _physical_rows_noalign
                if input.dtype != torch.float32
                and len(shape) >= 3
                and (4194304 <= n < 16777216)
                and all((s % 16 == 0 for s in strides[1:]))
                else (
                    _physical_rows_noalign
                    if input.dtype == torch.float32
                    else _physical_rows
                )
            ).run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        else:
            _physical_rows_pair.run(
                input,
                output_s,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                input.data_ptr() % 4 // 2,
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
        return output_s
    if (
        len(shape) >= 3
        and strides[0] == 1
        and (256 <= shape[0] <= 4096)
        and (n >= 1048576)
    ):
        c = shape[0]
        bc = 1 << (c - 1).bit_length()
        use = (
            n >= 4194304 and (c >= 512 or c * 4 >= bc * 3) and (c & c - 1 != 0)
        )
        cap = (
            2048 if input.dtype == torch.float32 and c * 4 >= bc * 3 else 4096
        )
        packed = (
            c % 2 == 0
            and input.dtype in (torch.float16, torch.bfloat16)
            and (input.data_ptr() % 4 == 0)
            and all((s % 2 == 0 for s in strides[1:]))
        )
        use = use and (not packed)
        if use:
            br = max(1, cap // bc)
            rows = n // c
            padded_rows = (rows + br - 1) // br * br
            inner_rows = rows // shape[-1]
            outer_extra = (padded_rows - rows + inner_rows - 1) // inner_rows
            span = sum(((d - 1) * s for d, s in zip(shape, strides)))
            safe32 = (
                padded_rows <= 2147483647
                and span + outer_extra * strides[-1] + bc - c <= 2147483647
            )
            _multirow_tiles.run(
                input,
                output_s,
                tuple(shape[1:]),
                tuple(strides[1:]),
                rows,
                c,
                br,
                bc,
                safe32,
                grid=((rows + br - 1) // br,),
                num_warps=4,
                warmup=False,
            )
            return output_s
    if (
        len(shape) == 1
        and strides[0] == 2
        and (input.dtype == torch.float32)
        and (n >= 8388608)
    ):
        _stride2_words32.run(
            input,
            output_s,
            n,
            input.storage_offset() % 2,
            grid=((n + 4095) // 4096,),
            warmup=False,
        )
        return output_s
    if (
        len(shape) == 1
        and strides[0] == 2
        and (input.dtype in (torch.float16, torch.bfloat16))
        and (n >= 4194304)
    ):
        _stride2_words.run(
            input,
            output_s,
            n,
            input.storage_offset() % 2,
            grid=((n + 4095) // 4096,),
            warmup=False,
        )
        return output_s
    if len(shape) == 1 and 24 <= strides[0] <= 256 and (n >= 4194304):
        _stride_native1k.run(
            input,
            output_s,
            n,
            strides[0],
            "",
            grid=((n + 1023) // 1024,),
            num_warps=8,
            warmup=False,
        )
        return output_s
    if len(shape) == 1 and 3 <= strides[0] <= 8 and (n >= 4194304):
        _stride_native.run(
            input,
            output_s,
            n,
            strides[0],
            "",
            grid=((n + 4095) // 4096,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if not shape or (len(shape) == 1 and strides[0] == 1):
        return per_tensor_absmax_fp8(input.as_strided((n,), (1,)), output_s)
    if (
        len(shape) == 2
        and strides[0] == 1
        and (8192 < shape[0] <= 1048576)
        and (strides[1] <= 4 * shape[0] or n >= 1048576)
        and (n >= 32768)
    ):
        bc = 2048
        br = 2048 // bc
        nc = (shape[0] + bc - 1) // bc
        g = min((shape[1] + br - 1) // br, max(1, 1024 // nc))
        _wide2d.run(
            input,
            output_s,
            shape[1],
            shape[0],
            strides[1],
            br,
            bc,
            g,
            nc,
            grid=(g * nc,),
            warmup=False,
        )
        return output_s
    if (
        len(shape) == 2
        and strides[0] == 1
        and (shape[0] <= 8192)
        and (strides[1] <= 4 * shape[0] or (shape[0] >= 1025 and n >= 4194304))
        and (n >= 1048576)
    ):
        bc = triton.next_power_of_2(shape[0])
        br = max(1, min(128, 2048 // bc))
        g = min((shape[1] + br - 1) // br, 1024)
        _rows2d.run(
            input,
            output_s,
            shape[1],
            shape[0],
            strides[1],
            br,
            bc,
            g,
            grid=(g,),
            warmup=False,
        )
        return output_s
    if (
        len(shape) >= 2
        and n >= 4194304
        and (strides[0] == 1)
        and (shape[0] % 2 == 0)
        and (input.dtype in (torch.float16, torch.bfloat16))
        and (input.data_ptr() % 4 == 0)
        and all((s % 2 == 0 for s in strides[1:]))
    ):
        sh = [shape[0] // 2] + shape[1:]
        st = [1] + [s // 2 for s in strides[1:]]
        nw = n // 2
        padded = (nw + 4095) // 4096 * 4096
        span = sum(((d - 1) * s for d, s in zip(sh, st)))
        inner = nw // sh[-1]
        extra_rows = (padded - nw + inner - 1) // inner
        safe32 = (
            padded <= 2147483647 and span + extra_rows * st[-1] <= 2147483647
        )
        _multiaxis_packed.run(
            input,
            output_s,
            nw,
            tuple(reversed(sh)),
            tuple(reversed(st)),
            safe32,
            grid=((nw + 4095) // 4096,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if len(shape) >= 2 and n >= 1048576:
        padded = (n + 4095) // 4096 * 4096
        span = sum(((d - 1) * s for d, s in zip(shape, strides)))
        inner = n // shape[-1]
        extra_rows = (padded - n + inner - 1) // inner
        safe32 = (
            padded <= 2147483647
            and span + extra_rows * strides[-1] <= 2147483647
        )
        _multiaxis_fullgrid.run(
            input,
            output_s,
            n,
            tuple(reversed(shape)),
            tuple(reversed(strides)),
            safe32,
            grid=((n + 4095) // 4096,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    b = min(triton.next_power_of_2(n), 4096)
    p = min(triton.cdiv(n, b), 256)
    padded = (n + b * p - 1) // (b * p) * b * p
    span = sum(((d - 1) * s for d, s in zip(shape, strides)))
    inner = n // shape[-1]
    extra_rows = (padded - n + inner - 1) // inner
    safe32 = (
        padded <= 2147483647 and span + extra_rows * strides[-1] <= 2147483647
    )
    _reduce[p,](
        input,
        output_s,
        n,
        tuple(reversed(shape)),
        tuple(reversed(strides)),
        False,
        b,
        p,
        p == 1,
        safe32,
    )
    return output_s


def per_tensor_absmax_fp8(input, output_s):
    n = input.numel()
    if (
        input.dtype == torch.float32
        and 524288 <= n < 16777216
        and (n % 8192 != 0)
        and input.is_contiguous()
        and (input.data_ptr() % 16 == 0)
    ):
        _fp32_native_tail.run(
            input,
            output_s,
            n,
            grid=((n + 8191) // 8192,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if (
        input.dtype == torch.float32
        and input.is_contiguous()
        and (n >= 2097152)
        and (input.data_ptr() % 16 != 0)
        and (n % 8192 == (16 - input.data_ptr() % 16) // 4)
    ):
        head = (16 - input.data_ptr() % 16) // 4
        _fp32_aligned_tail.run(
            input,
            output_s,
            n,
            head,
            grid=((n - head + 8191) // 8192,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if (
        input.dtype == torch.float32
        and input.is_contiguous()
        and (n >= 16777216)
        and (n % 8192 == 0)
    ):
        _fp32_native.run(
            input, output_s, grid=(n // 8192,), num_warps=4, warmup=False
        )
        return output_s
    if (
        input.dtype == torch.float32
        and input.is_contiguous()
        and (n >= 16777216)
        and (n % 8192 != 0)
    ):
        _fp32_native_tail.run(
            input,
            output_s,
            n,
            grid=((n + 8191) // 8192,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    limit = 512
    if (
        input.dtype in (torch.float16, torch.bfloat16)
        and input.is_contiguous()
        and (input.storage_offset() % 2 == 1)
        and (n >= 2097152)
    ):
        p = min(((n - 1) // 2 + 2047) // 2048, limit)
        _packed_head.run(input, output_s, n, grid=(p,), warmup=False)
        return output_s
    if (
        input.dtype == torch.float16
        and input.is_contiguous()
        and (input.storage_offset() % 2 == 0)
        and (n >= 1048576)
        and (n % 4096 != 0)
    ):
        p = min((n // 2 + 2047) // 2048, limit)
        if n % 2:
            _packed_tail.run(input, output_s, n // 2, grid=(p,), warmup=False)
        else:
            _packed_partial.run(
                input, output_s, n // 2, grid=(p,), warmup=False
            )
        return output_s
    if (
        (
            input.dtype == torch.bfloat16
            or (
                input.dtype == torch.float16
                and n >= 4194304
                and (n % 4096 == 0)
                and (input.data_ptr() % 4 == 0)
            )
        )
        and input.is_contiguous()
        and (n % 2 == 0 or n >= 262144)
        and (input.storage_offset() % 2 == 0)
    ):
        p = min((n // 2 + 2047) // 2048, limit)
        if n % 2:
            _packed_tail.run(input, output_s, n // 2, grid=(p,), warmup=False)
        elif n >= 1048576 and n % 4096 != 0:
            _packed_partial.run(
                input, output_s, n // 2, grid=(p,), warmup=False
            )
        else:
            _packed.run(input, output_s, n // 2, grid=(p,), warmup=False)
    else:
        if not input.is_contiguous():
            return _layout(input, output_s)
        b = min(triton.next_power_of_2(n), 4096)
        p = min(triton.cdiv(n, b), 256)
        _reduce[p,](
            input,
            output_s,
            n,
            tuple(input.shape),
            tuple(input.stride()),
            input.is_contiguous(),
            b,
            p,
            p == 1,
        )
    return output_s


@triton.jit
def _packed_tail(X, Output, N: tl.constexpr):
    B: tl.constexpr = 512 if N <= 512 else 1024 if N <= 1024 else 2048
    G: tl.constexpr = (N + B - 1) // B if N <= B * 512 else 512
    pid = tl.program_id(0)
    if N >= 8388608 and G >= 64 and (G % 64 == 0):
        pid = pid % 64 * (G // 64) + pid // 64
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (pid + step.to(tl.int64) * G) * B + lane
        else:
            i = (pid + step * G) * B + lane
        if (step.to(tl.int64) * G + pid + 1) * B <= N:
            x = tl.load(X.to(tl.pointer_type(tl.int32)) + i)
        else:
            x = tl.load(X.to(tl.pointer_type(tl.int32)) + i, i < N, other=0)
        lo = x & 32767
        hi = x >> 16 & 32767
        acc = tl.maximum(acc, tl.maximum(lo, hi))
    maximum = tl.max(acc, 0)
    if pid == 0:
        last = tl.load(X + tl.full((), N, tl.int64) * 2)
        last_bits = last.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        maximum = tl.maximum(maximum, last_bits)
    maximum_bits = maximum
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum_bits << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        v = (
            maximum_bits.to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
            / 448.0
        )
    if G == 1:
        tl.store(Output, v)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)),
            v.to(tl.int32, bitcast=True),
            sem="relaxed",
        )


@triton.jit
def _packed_partial(X, Output, N: tl.constexpr):
    B: tl.constexpr = 512 if N <= 512 else 1024 if N <= 1024 else 2048
    G: tl.constexpr = (N + B - 1) // B if N <= B * 512 else 512
    pid = tl.program_id(0)
    if N >= 8388608 and G >= 64 and (G % 64 == 0):
        pid = pid % 64 * (G // 64) + pid // 64
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (pid + step.to(tl.int64) * G) * B + lane
        else:
            i = (pid + step * G) * B + lane
        if N % (G * B) == 0:
            x = tl.load(X.to(tl.pointer_type(tl.int32)) + i, i < N, other=0)
        elif (step.to(tl.int64) * G + pid + 1) * B <= N:
            x = tl.load(X.to(tl.pointer_type(tl.int32)) + i)
        else:
            x = tl.load(X.to(tl.pointer_type(tl.int32)) + i, i < N, other=0)
        lo = x & 32767
        hi = x >> 16 & 32767
        acc = tl.maximum(acc, tl.maximum(lo, hi))
    maximum_bits = tl.max(acc, 0)
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum_bits << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        v = (
            maximum_bits.to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
            / 448.0
        )
    if G == 1:
        tl.store(Output, v)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)),
            v.to(tl.int32, bitcast=True),
            sem="relaxed",
        )


@triton.jit
def _packed_head(X, Output, M: tl.constexpr):
    N: tl.constexpr = (M - 1) // 2
    B: tl.constexpr = 2048
    G: tl.constexpr = min((N + B - 1) // B, 512)
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    P = (X + 1).to(tl.pointer_type(tl.int32))
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        else:
            i = (step * G + tl.program_id(0)) * B + lane
        if (step.to(tl.int64) * G + tl.program_id(0) + 1) * B <= N:
            word = tl.load(P + i)
        else:
            word = tl.load(P + i, i < N, other=0)
        acc = tl.maximum(acc, tl.maximum(word & 32767, word >> 16 & 32767))
    maximum = tl.max(acc, 0)
    if tl.program_id(0) == 0:
        first = tl.load(X)
        first_bits = first.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        maximum = tl.maximum(maximum, first_bits)
        if M % 2 == 0:
            last = tl.load(X + tl.full((), M - 1, tl.int64))
            last_bits = last.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
            maximum = tl.maximum(maximum, last_bits)
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        v = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    if G == 1:
        tl.store(Output, v)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)),
            v.to(tl.int32, bitcast=True),
            sem="relaxed",
        )


@triton.jit
def _stride2_words(X, Output, N: tl.constexpr, ODD: tl.constexpr):
    B: tl.constexpr = 4096
    G: tl.constexpr = (N + B - 1) // B
    WORDS: tl.constexpr = N if ODD else N - 1
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    P = (X - ODD).to(tl.pointer_type(tl.int32))
    WIDE: tl.constexpr = (WORDS + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((WORDS + G * B - 1) // (G * B)):
        if WIDE:
            i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        else:
            i = (step * G + tl.program_id(0)) * B + lane
        if (step.to(tl.int64) * G + tl.program_id(0) + 1) * B <= WORDS:
            word = tl.load(P + i)
        else:
            word = tl.load(P + i, i < WORDS, other=0)
        bits = word >> (16 if ODD else 0) & 32767
        acc = tl.maximum(acc, bits)
    maximum = tl.max(acc, 0)
    if not ODD:
        if tl.program_id(0) == 0:
            last = tl.load(X + tl.full((), N - 1, tl.int64) * 2)
            maximum = tl.maximum(
                maximum, last.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
            )
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        v = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        v.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _stride2_words32(X, Output, N: tl.constexpr, ODD: tl.constexpr):
    B: tl.constexpr = 4096
    G: tl.constexpr = (N + B - 1) // B
    WORDS: tl.constexpr = N if ODD else N - 1
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    P = (X - ODD).to(tl.pointer_type(tl.int64))
    WIDE: tl.constexpr = (WORDS + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((WORDS + G * B - 1) // (G * B)):
        if WIDE:
            i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        else:
            i = (step * G + tl.program_id(0)) * B + lane
        if (step.to(tl.int64) * G + tl.program_id(0) + 1) * B <= WORDS:
            word = tl.load(P + i)
        else:
            word = tl.load(P + i, i < WORDS, other=0)
        bits = (word >> (32 if ODD else 0) & 2147483647).to(tl.int32)
        acc = tl.maximum(acc, bits)
    maximum = tl.max(acc, 0)
    if not ODD:
        if tl.program_id(0) == 0:
            last = tl.load(X + tl.full((), N - 1, tl.int64) * 2)
            maximum = tl.maximum(
                maximum, last.to(tl.int32, bitcast=True) & 2147483647
            )
    v = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        v.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _stride_native(
    X, Output, N: tl.constexpr, S: tl.constexpr, CACHE: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        x = tl.load(X + i * S, cache_modifier=CACHE)
    else:
        x = tl.load(X + i * S, i < N, other=0, cache_modifier=CACHE)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        v = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        v = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        v = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        v.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _multiaxis_fullgrid(
    X,
    Output,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    ADDR32: tl.constexpr,
):
    if ADDR32:
        i = tl.program_id(0) * 4096 + tl.arange(0, 4096)
        off = tl.full((4096,), 0, tl.int32)
    else:
        i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
        off = tl.full((4096,), 0, tl.int64)
    q = i
    for d in tl.static_range(len(SH) - 1, -1, -1):
        if d == 0:
            off += q * ST[d]
        else:
            off += q % SH[d] * ST[d]
            q = q // SH[d]
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        x = tl.load(X + off)
    else:
        x = tl.load(X + off, i < N, other=0)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        value = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _multiaxis_packed(
    X,
    Output,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    ADDR32: tl.constexpr,
):
    if ADDR32:
        i = tl.program_id(0) * 4096 + tl.arange(0, 4096)
        off = tl.full((4096,), 0, tl.int32)
    else:
        i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
        off = tl.full((4096,), 0, tl.int64)
    q = i
    for d in tl.static_range(len(SH) - 1, -1, -1):
        if d == 0:
            off += q * ST[d]
        else:
            off += q % SH[d] * ST[d]
            q = q // SH[d]
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        w = tl.load(X.to(tl.pointer_type(tl.uint32)) + off)
    else:
        w = tl.load(X.to(tl.pointer_type(tl.uint32)) + off, i < N, other=0)
    bits = tl.maximum(w & 32767, w >> 16 & 32767)
    maximum = tl.max(bits, 0).to(tl.int32)
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _multirow_tiles(
    X,
    Output,
    SH: tl.constexpr,
    ST: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    ADDR32: tl.constexpr,
):
    if ADDR32:
        row = tl.program_id(0) * BR + tl.arange(0, BR)
        off = tl.full((BR,), 0, tl.int32)
    else:
        row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
        off = tl.full((BR,), 0, tl.int64)
    col = tl.arange(0, BC)
    q = row
    for d in tl.static_range(len(SH)):
        if d == len(SH) - 1:
            off += q * ST[d]
        else:
            off += q % SH[d] * ST[d]
            q = q // SH[d]
    x = tl.load(
        X + off[:, None] + col[None, :],
        (row[:, None] < R) & (col[None, :] < C),
        other=0,
    )
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(tl.max(bits, 1), 0)
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        value = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _physical_rows(
    X,
    Output,
    C: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    BR: tl.constexpr,
    L: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    groups: tl.constexpr = (R + BR - 1) // BR
    row = tile % groups * BR
    q = tile // groups
    base = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(SH)):
        if d == len(SH) - 1:
            base += q * ST[d]
        else:
            base += q % SH[d] * ST[d]
            q = q // SH[d]
    lane = tl.arange(0, L)
    limit = tl.minimum((R - 1 - row) * P + C, BR * P).to(tl.int32)
    raw = tl.load(X + base + row * P + lane, lane < limit, other=0)
    m: tl.constexpr = ((1 << 25) + P - 1) // P
    col = lane - (lane * m >> 25) * P
    x = tl.where(col < C, raw, 0)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        value = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _physical_rows_pair(
    X,
    Output,
    C: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    BR: tl.constexpr,
    L: tl.constexpr,
    PARITY: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    groups: tl.constexpr = (R + BR - 1) // BR
    row = tile % groups * BR
    q = tile // groups
    base = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(SH)):
        if d == len(SH) - 1:
            base += q * ST[d]
        else:
            base += q % SH[d] * ST[d]
            q = q // SH[d]
    base += row * P
    head = (PARITY + base & 1).to(tl.int32)
    limit = tl.minimum((R - 1 - row) * P + C, BR * P).to(tl.int32)
    lane = tl.arange(0, L // 2)
    local = 2 * lane + head
    raw = tl.load(
        (X + base + head).to(tl.pointer_type(tl.int32)) + lane,
        local + 1 < limit,
        other=0,
    )
    m: tl.constexpr = ((1 << 25) + P - 1) // P
    col = local - (local * m >> 25) * P
    nextcol = tl.where(col + 1 == P, 0, col + 1)
    lo = tl.where(col < C, raw & 32767, 0)
    hi = tl.where(nextcol < C, raw >> 16 & 32767, 0)
    first = (
        tl.load(X + base, head == 1, other=0)
        .to(tl.uint16, bitcast=True)
        .to(tl.int32)
        & 32767
    )
    last = (
        tl.load(
            X + base + limit - 1,
            limit - head & 1 == 1 and (limit - 1) % P < C,
            other=0,
        )
        .to(tl.uint16, bitcast=True)
        .to(tl.int32)
        & 32767
    )
    maximum = tl.maximum(
        tl.max(tl.maximum(lo, hi), 0), tl.maximum(first, last)
    )
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    else:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize_on_alignment=["X"])
def _physical_rows_noalign(
    X,
    Output,
    C: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    BR: tl.constexpr,
    L: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    groups: tl.constexpr = (R + BR - 1) // BR
    row = tile % groups * BR
    q = tile // groups
    base = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(SH)):
        if d == len(SH) - 1:
            base += q * ST[d]
        else:
            base += q % SH[d] * ST[d]
            q = q // SH[d]
    lane = tl.arange(0, L)
    limit = tl.minimum((R - 1 - row) * P + C, BR * P).to(tl.int32)
    raw = tl.load(X + base + row * P + lane, lane < limit, other=0)
    m: tl.constexpr = ((1 << 25) + P - 1) // P
    col = lane - (lane * m >> 25) * P
    x = tl.where(col < C, raw, 0)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        value = (maximum << 16).to(tl.float32, bitcast=True) / 448.0
    elif X.dtype.element_ty == tl.float16:
        value = (
            maximum.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            / 448.0
        )
    else:
        value = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize_on_alignment=["X"])
def _highdim_rect(
    X,
    Output,
    C: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    groups: tl.constexpr = (R + BR - 1) // BR
    row = pid % groups * BR + tl.arange(0, BR)
    q = pid // groups
    base = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(SH)):
        if d == len(SH) - 1:
            base += q * ST[d]
        else:
            base += q % SH[d] * ST[d]
            q = q // SH[d]
    col = tl.arange(0, BC)
    x = tl.load(
        X + base + row[:, None] * P + col[None, :],
        (row[:, None] < R) & (col[None, :] < C),
        other=0,
    ).to(tl.float32)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    value = tl.max(tl.max(bits, 1), 0).to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _fp32_native(X, Output):
    i = tl.program_id(0).to(tl.int64) * 8192 + tl.arange(0, 8192)
    x = tl.load(X + i, eviction_policy="evict_first")
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(bits, 0).to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _fp32_native_tail(X, Output, N: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * 8192 + tl.arange(0, 8192)
    if (tl.program_id(0).to(tl.int64) + 1) * 8192 <= N:
        x = tl.load(X + i, eviction_policy="evict_first")
    else:
        x = tl.load(X + i, i < N, other=0, eviction_policy="evict_first")
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(bits, 0).to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _fp32_aligned_tail(X, Output, N: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, 8192)
    i = pid.to(tl.int64) * 8192 + lane
    aligned = tl.multiple_of(X.to(tl.uint64) + H * 4, 16).to(X.dtype)
    if (pid.to(tl.int64) + 1) * 8192 <= N - H:
        x = tl.load(aligned + i, eviction_policy="evict_first")
    else:
        x = tl.load(
            aligned + i, i < N - H, other=0, eviction_policy="evict_first"
        )
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(bits, 0)
    if pid == 0:
        prefix_lane = tl.arange(0, 4)
        prefix = tl.load(X + prefix_lane, prefix_lane < H, other=0)
        prefix_bits = prefix.to(tl.int32, bitcast=True) & 2147483647
        maximum = tl.maximum(maximum, tl.max(prefix_bits, 0))
    value = maximum.to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


__all__ = ["per_tensor_absmax_fp8"]
