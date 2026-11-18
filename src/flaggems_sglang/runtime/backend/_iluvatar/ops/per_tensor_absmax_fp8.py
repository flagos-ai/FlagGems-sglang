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
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
        if WIDE:
            i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        else:
            i = (step * G + tl.program_id(0)) * B + lane
        off = i
        if not CONTIG:
            q = i.to(tl.int64)
            off = tl.full((B,), 0, tl.int64)
            for d in tl.static_range(len(SH) - 1, -1, -1):
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
def _finish(P, Output, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    v = tl.load(P + i, i < N, other=0)
    tl.store(Output, tl.max(v, 0) / 448.0)


@triton.jit
def _contiguous(X, Output, N: tl.constexpr):
    if N <= 1048576:
        B: tl.constexpr = 1024
        G: tl.constexpr = 128 if N >= 196608 else triton.cdiv(N, B)
        lane = tl.arange(0, B)
        acc = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
        for step in range((N + G * B - 1) // (G * B)):
            if WIDE:
                i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
            else:
                i = (step * G + tl.program_id(0)) * B + lane
            if N % (B * G) == 0:
                x = tl.load(X + i, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x = tl.load(
                    X + i, i < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
        value = tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
                sem="relaxed",
            )
    else:
        B: tl.constexpr = 1024
        G: tl.constexpr = (N + B - 1) // B if N <= B * 256 else 256
        lane = tl.arange(0, B)
        acc0 = tl.full((B,), 0, tl.int32)
        acc1 = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + 2 * G * B - 1) // (
            2 * G * B
        ) * 2 * G * B > 2147483647
        for step in range((N + 2 * G * B - 1) // (2 * G * B)):
            if N >= 16777216 and (
                X.dtype.element_ty == tl.float16
                or (X.dtype.element_ty == tl.bfloat16 and N % 524288 != 0)
            ):
                if WIDE:
                    owner = (
                        tl.program_id(0).to(tl.int64) + step.to(tl.int64)
                    ) % G
                else:
                    owner = (tl.program_id(0) + step) % G
            else:
                owner = tl.program_id(0)
            if WIDE:
                if N > 8388608:
                    base = (step.to(tl.int64) * G + owner) * 2 * B + lane
                else:
                    base = (step.to(tl.int64) * 2 * G + owner) * B + lane
            elif N > 8388608:
                base = (step * G + owner) * 2 * B + lane
            else:
                base = (step * 2 * G + owner) * B + lane
            i0 = base + 0 * G * B
            if N % (2 * B * G) == 0:
                x0 = tl.load(X + i0, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x0 = tl.load(
                    X + i0, i0 < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            if N > 8388608:
                i1 = base + B
            else:
                i1 = base + G * B
            if N % (2 * B * G) == 0:
                x1 = tl.load(X + i1, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x1 = tl.load(
                    X + i1, i1 < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            acc0 = tl.maximum(acc0, x0.to(tl.int32, bitcast=True) & 2147483647)
            acc1 = tl.maximum(acc1, x1.to(tl.int32, bitcast=True) & 2147483647)
        value = (
            tl.max(tl.maximum(acc0, acc1), 0).to(tl.float32, bitcast=True)
            / 448.0
        )
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
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
        len(shape) == 1
        and 24 <= strides[0] <= (64 if input.dtype == torch.float32 else 256)
        and (n >= 4194304)
    ) and strides[0] & strides[0] - 1 != 0:
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
    if (
        3 <= len(shape) <= 4
        and input.dtype != torch.float32
        and (strides[0] == 1)
        and (319 <= shape[0] <= 511)
        and (shape[0] % 2 == 1)
        and (n >= 4194304)
        and (7 * shape[0] <= 4 * strides[1] <= 8 * shape[0])
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = 8
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
            grid=(tiles,),
            num_warps=4,
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
                if n >= 16777216
                and shape[0] % 2 == 1
                and (input.dtype != torch.float32 or len(shape) >= 6)
                else (
                    5 * shape[0] // 4 if shape[0] <= 255 else 9 * shape[0] // 8
                )
            )
        )
        and (shape[1] >= 8192 // strides[1])
        and (input.dtype != torch.float32 or n >= 4194304)
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        if input.dtype != torch.float32 and c >= 319 and (4 * pitch >= 7 * c):
            br = 8
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
                grid=(tiles,),
                num_warps=4,
                warmup=False,
            )
            return output_s
        br = max(
            1,
            (8192 if input.dtype != torch.float32 and n >= 16777216 else 4096)
            // pitch,
        )
        if (
            input.dtype != torch.float32
            and n >= 16777216
            and (pitch % 2 == 1)
            and (br >= 32)
            and (br % 2 == 1)
            and (input.data_ptr() % 16 == 0)
            and all((s % 8 == 0 for s in strides[2:]))
        ):
            br = br // 8 * 8
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
                num_warps=8,
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
                num_warps=8,
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
        and (input.dtype != torch.float32 or n >= 16777216)
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (8192 if input.dtype != torch.float32 and n >= 16777216 else 4096)
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
                    and len(shape) >= 3
                    and (n >= 4194304)
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
                num_warps=8,
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
                num_warps=8,
                warmup=False,
            )
        return output_s
    if (
        3 <= len(shape) <= 4
        and strides[0] == 1
        and (129 <= shape[0] <= 255)
        and (
            shape[0] % 2 == 1
            or (
                n >= 4194304
                and strides[1] % 2 == 1
                and (input.dtype != torch.float32 or n >= 16777216)
            )
        )
        and (n >= (4194304 if input.dtype == torch.float32 else 1048576))
        and (
            strides[1]
            <= (
                5 * shape[0] // 4
                if n >= (16777216 if input.dtype == torch.float32 else 4194304)
                else 9 * shape[0] // 8
            )
        )
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (8192 if input.dtype != torch.float32 and n >= 16777216 else 4096)
            // pitch,
        )
        if (
            input.dtype != torch.float32
            and n >= 16777216
            and (pitch % 2 == 1)
            and (br >= 32)
            and (br % 2 == 1)
            and (input.data_ptr() % 16 == 0)
            and all((s % 8 == 0 for s in strides[2:]))
        ):
            br = br // 8 * 8
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
                    and len(shape) >= 3
                    and (n >= 4194304)
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
                num_warps=8,
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
                num_warps=8,
                warmup=False,
            )
        return output_s
    if (
        2 <= len(shape) <= 4
        and (
            len(shape) != 2
            or input.dtype != torch.float32
            or shape[0] >= 287
            or (n >= 33554432)
        )
        and (strides[0] == 1)
        and (257 <= shape[0] <= 769)
        and (shape[0] <= 511 or n >= 16777216)
        and (shape[0] % 2 == 1)
        and (
            n
            >= (
                (2097152 if len(shape) == 2 else 4194304)
                if input.dtype == torch.float32
                else 1048576
            )
        )
        and (strides[1] <= min(4096, 9 * shape[0] // 8))
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (8192 if input.dtype != torch.float32 and n >= 16777216 else 4096)
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
                    and len(shape) >= 3
                    and (n >= 4194304)
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
                num_warps=8,
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
                num_warps=8,
                warmup=False,
            )
        return output_s
    if not shape or (len(shape) == 1 and strides[0] == 1):
        return per_tensor_absmax_fp8(input.as_strided((n,), (1,)), output_s)
    if (
        len(shape) == 2
        and strides[0] == 1
        and (17 <= shape[0] <= 512)
        and (strides[1] <= 4 * shape[0])
        and (n >= 1048576)
        and (shape[0] & shape[0] - 1 or 4194304 <= n <= 8388608)
    ):
        bc = 1 << (max(1, shape[0]) - 1).bit_length()
        br = min(128, 2048 // bc)
        g = min((shape[1] + br - 1) // br, 256)
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
    b = min(1 << (max(1, n) - 1).bit_length(), 1024)
    p = min((n + b - 1) // b, 256)
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
    )
    return output_s


def per_tensor_absmax_fp8(input, output_s):
    n = input.numel()
    if input.is_contiguous():
        p = 128 if 196608 <= n <= 1048576 else min((n + 1023) // 1024, 256)
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and input.storage_offset() % 64
            and (n >= 4194304)
        ):
            h = -input.storage_offset() % 64
            if (n - h + 524287) // 524288 * 524288 <= 2147483647 and (
                input.data_ptr() + 2 * h
            ) % 128 == 0:
                _peel_prefix.run(
                    input,
                    output_s,
                    n,
                    h,
                    grid=(256,),
                    num_warps=8,
                    warmup=False,
                )
                return output_s
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and input.storage_offset() % 8 == 0
            and (n > 1048576)
            and (n % 524288 != 0)
            and ((n + 524287) // 524288 * 524288 <= 2147483647)
        ):
            _contiguous_tail.run(
                input, output_s, n, grid=(p,), num_warps=8, warmup=False
            )
        elif (
            input.dtype == torch.float32
            and 32768 <= n <= 1048576
            and (input.data_ptr() % 128 == 0)
        ):
            _contiguous_scalar1376.run(
                input,
                output_s,
                n,
                grid=(p,),
                num_warps=(
                    (16 if input.dtype == torch.float32 else 8)
                    if n > 1048576
                    else 8 if n >= 32768 and input.data_ptr() % 128 == 0 else 4
                ),
                warmup=False,
            )
        else:
            _contiguous.run(
                input,
                output_s,
                n,
                grid=(p,),
                num_warps=(
                    (16 if input.dtype == torch.float32 else 8)
                    if n > 1048576
                    else 8 if n >= 32768 and input.data_ptr() % 128 == 0 else 4
                ),
                warmup=False,
            )
        return output_s
    return _layout(input, output_s)


@triton.jit
def _contiguous_tail(X, Output, N: tl.constexpr):
    if N <= 1048576:
        B: tl.constexpr = 1024
        G: tl.constexpr = (N + B - 1) // B if N <= B * 256 else 256
        lane = tl.arange(0, B)
        acc = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
        for step in range(N // (G * B)):
            if WIDE:
                i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
            else:
                i = (step * G + tl.program_id(0)) * B + lane
            x = tl.load(X + i, eviction_policy="evict_first").to(tl.float32)
            acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
        if N % (G * B) != 0:
            step = tl.full((), N // (G * B), tl.int64)
            if WIDE:
                i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
            else:
                i = (step * G + tl.program_id(0)) * B + lane
            if N % (B * G) == 0:
                x = tl.load(X + i, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x = tl.load(
                    X + i, i < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
        value = tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
                sem="relaxed",
            )
    else:
        B: tl.constexpr = 1024
        G: tl.constexpr = (N + B - 1) // B if N <= B * 256 else 256
        lane = tl.arange(0, B)
        acc0 = tl.full((B,), 0, tl.int32)
        acc1 = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + 2 * G * B - 1) // (
            2 * G * B
        ) * 2 * G * B > 2147483647
        for step in range(N // (2 * G * B)):
            if N >= 16777216 and (
                X.dtype.element_ty == tl.float16
                or (X.dtype.element_ty == tl.bfloat16 and N % 524288 != 0)
            ):
                if WIDE:
                    owner = (
                        tl.program_id(0).to(tl.int64) + step.to(tl.int64)
                    ) % G
                else:
                    owner = (tl.program_id(0) + step) % G
            else:
                owner = tl.program_id(0)
            if WIDE:
                if N > 8388608:
                    base = (step.to(tl.int64) * G + owner) * 2 * B + lane
                else:
                    base = (step.to(tl.int64) * 2 * G + owner) * B + lane
            elif N > 8388608:
                base = (step * G + owner) * 2 * B + lane
            else:
                base = (step * 2 * G + owner) * B + lane
            i0 = base + 0 * G * B
            x0 = tl.load(X + i0, eviction_policy="evict_first").to(tl.float32)
            if N > 8388608:
                i1 = base + B
            else:
                i1 = base + G * B
            x1 = tl.load(X + i1, eviction_policy="evict_first").to(tl.float32)
            acc0 = tl.maximum(acc0, x0.to(tl.int32, bitcast=True) & 2147483647)
            acc1 = tl.maximum(acc1, x1.to(tl.int32, bitcast=True) & 2147483647)
        if N % (2 * G * B) != 0:
            BULK: tl.constexpr = N // (2 * G * B) * (2 * G * B)
            REM: tl.constexpr = N - BULK
            tail_x = X + tl.full((), BULK, tl.int64)
            if N > 8388608:
                j0 = tl.program_id(0) * 2 * B + lane
                j1 = j0 + B
            else:
                j0 = tl.program_id(0) * B + lane
                j1 = j0 + G * B
            x0 = tl.load(
                tail_x + j0, j0 < REM, other=0, eviction_policy="evict_first"
            ).to(tl.float32)
            x1 = tl.load(
                tail_x + j1, j1 < REM, other=0, eviction_policy="evict_first"
            ).to(tl.float32)
            acc0 = tl.maximum(acc0, x0.to(tl.int32, bitcast=True) & 2147483647)
            acc1 = tl.maximum(acc1, x1.to(tl.int32, bitcast=True) & 2147483647)
        value = (
            tl.max(tl.maximum(acc0, acc1), 0).to(tl.float32, bitcast=True)
            / 448.0
        )
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
                sem="relaxed",
            )


@triton.jit
def _peel_prefix(X, Output, N: tl.constexpr, H: tl.constexpr):
    _contiguous_tail(tl.multiple_of(X + H, 128), Output, N - H)
    if tl.program_id(0) == 0:
        lane = tl.arange(0, 64)
        x = tl.load(X + lane, lane < H, other=0).to(tl.float32)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        value = tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.uint32)),
            value.to(tl.uint32, bitcast=True),
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
    col = tl.arange(0, 512)
    x = tl.load(
        X + base + row[:, None] * P + col[None, :],
        (row[:, None] < R) & (col[None, :] < C),
        other=0,
    )
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(tl.max(bits, 1), 0)
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


@triton.jit(do_not_specialize=["X", "Output"])
def _contiguous_scalar1376(X, Output, N: tl.constexpr):
    if N <= 1048576:
        B: tl.constexpr = 1024
        G: tl.constexpr = 128 if N >= 196608 else triton.cdiv(N, B)
        lane = tl.arange(0, B)
        acc = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
        for step in range((N + G * B - 1) // (G * B)):
            if WIDE:
                i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
            else:
                i = (step * G + tl.program_id(0)) * B + lane
            if N % (B * G) == 0:
                x = tl.load(X + i, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x = tl.load(
                    X + i, i < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
        value = tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
                sem="relaxed",
            )
    else:
        B: tl.constexpr = 1024
        G: tl.constexpr = (N + B - 1) // B if N <= B * 256 else 256
        lane = tl.arange(0, B)
        acc0 = tl.full((B,), 0, tl.int32)
        acc1 = tl.full((B,), 0, tl.int32)
        WIDE: tl.constexpr = (N + 2 * G * B - 1) // (
            2 * G * B
        ) * 2 * G * B > 2147483647
        for step in range((N + 2 * G * B - 1) // (2 * G * B)):
            if WIDE:
                if N > 8388608:
                    base = (
                        step.to(tl.int64) * G + tl.program_id(0)
                    ) * 2 * B + lane
                else:
                    base = (
                        step.to(tl.int64) * 2 * G + tl.program_id(0)
                    ) * B + lane
            elif N > 8388608:
                base = (step * G + tl.program_id(0)) * 2 * B + lane
            else:
                base = (step * 2 * G + tl.program_id(0)) * B + lane
            i0 = base + 0 * G * B
            if N % (2 * B * G) == 0:
                x0 = tl.load(X + i0, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x0 = tl.load(
                    X + i0, i0 < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            if N > 8388608:
                i1 = base + B
            else:
                i1 = base + G * B
            if N % (2 * B * G) == 0:
                x1 = tl.load(X + i1, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                x1 = tl.load(
                    X + i1, i1 < N, other=0, eviction_policy="evict_first"
                ).to(tl.float32)
            acc0 = tl.maximum(acc0, x0.to(tl.int32, bitcast=True) & 2147483647)
            acc1 = tl.maximum(acc1, x1.to(tl.int32, bitcast=True) & 2147483647)
        value = (
            tl.max(tl.maximum(acc0, acc1), 0).to(tl.float32, bitcast=True)
            / 448.0
        )
        if G == 1:
            tl.store(Output, value)
        else:
            tl.atomic_max(
                Output.to(tl.pointer_type(tl.uint32)),
                value.to(tl.uint32, bitcast=True),
                sem="relaxed",
            )


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


__all__ = ["per_tensor_absmax_fp8"]
