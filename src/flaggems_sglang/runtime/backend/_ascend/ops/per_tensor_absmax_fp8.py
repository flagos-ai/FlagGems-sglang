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
def _contiguous(X, Output, N: tl.constexpr):
    B: tl.constexpr = 8192 if N > 8388608 else 4096
    G: tl.constexpr = (N + B - 1) // B if N <= B * 40 else 40
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range(tl.cdiv(N, G * B)):
        if WIDE:
            i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        else:
            i = (step * G + tl.program_id(0)) * B + lane
        x = tl.load(X + i, i < N, other=0)
        if X.dtype.element_ty == tl.float32:
            bits = x.to(tl.int32, bitcast=True) & 2147483647
        else:
            bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        acc = tl.maximum(acc, bits)
    v = tl.max(acc, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    if G == 1:
        tl.store(Output, value.to(tl.int32, bitcast=True))
    else:
        tl.atomic_max(Output, value.to(tl.int32, bitcast=True), sem="relaxed")


@triton.jit
def _parallel_strided(
    X,
    Output,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    for step in range(tl.cdiv(N, B * G)):
        i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
        q = i
        off = tl.full((B,), 0, tl.int64)
        for d in tl.static_range(len(SH) - 1, -1, -1):
            off += q % SH[d] * ST[d]
            q = q // SH[d]
        x = tl.load(X + off, i < N, other=0).to(tl.float32)
        acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
    value = (tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0).to(
        tl.int32, bitcast=True
    )
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


@triton.jit
def _rows(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    K: tl.constexpr = (C + B - 1) // B
    for t in range(tl.program_id(0), R * K, G):
        row = t // K
        col = t % K * B + lane
        x = tl.load(X + row.to(tl.int64) * S + col, col < C, other=0)
        if X.dtype.element_ty == tl.float32:
            bits = x.to(tl.int32, bitcast=True) & 2147483647
        else:
            bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
        acc = tl.maximum(acc, bits)
    v = tl.max(acc, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    bits = (maximum / 448.0).to(tl.int32, bitcast=True)
    if G == 1:
        tl.store(Output, bits)
    else:
        tl.atomic_max(Output, bits, sem="relaxed")


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
        tl.store(Output, bits)
    else:
        tl.atomic_max(Output, bits, sem="relaxed")


@triton.jit
def _local_gather(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, L)
    picks = tl.arange(0, B) * S
    K: tl.constexpr = (C + B - 1) // B
    acc = tl.full((B,), 0, tl.int32)
    for step in range((R * K + G - 1) // G):
        t = step.to(tl.int64) * G + tl.program_id(0)
        row = t // K
        col = t % K * B
        q = row
        base = tl.full((), 0, tl.int64)
        for d in tl.static_range(len(SH)):
            if d == len(SH) - 1:
                base += q * ST[d]
            else:
                base += q % SH[d] * ST[d]
                q = q // SH[d]
        offset = base + col * S
        raw = tl.load(
            X + offset + lane,
            (row < R) & (lane < B * S) & (lane < (C - col - 1) * S + 1),
            other=0,
        )
        selected = tl.gather(raw, picks, axis=0).to(tl.float32)
        bits = selected.to(tl.int32, bitcast=True) & 2147483647
        acc = tl.maximum(acc, bits)
    value = (tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0).to(
        tl.int32, bitcast=True
    )
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


@triton.jit
def _row_packets(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    S: tl.constexpr,
    BR: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    G: tl.constexpr,
):
    rows = tl.arange(0, BR)
    lane = tl.arange(0, L)
    picks = tl.broadcast_to((tl.arange(0, B) * S)[None, :], (BR, B))
    acc = tl.full((BR, B), 0, tl.int32)
    for step in range((R + BR * G - 1) // (BR * G)):
        row = (step.to(tl.int64) * G + tl.program_id(0)) * BR + rows
        q = row
        base = tl.full((BR,), 0, tl.int64)
        for d in tl.static_range(len(SH)):
            if d == len(SH) - 1:
                base += q * ST[d]
            else:
                base += q % SH[d] * ST[d]
                q = q // SH[d]
        raw = tl.load(
            X + base[:, None] + lane[None, :],
            (row[:, None] < R) & (lane[None, :] < (C - 1) * S + 1),
            other=0,
        )
        x = tl.gather(raw, picks, axis=1).to(tl.float32)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        acc = tl.maximum(acc, bits)
    value = (
        tl.max(tl.max(acc, 1), 0).to(tl.float32, bitcast=True) / 448.0
    ).to(tl.int32, bitcast=True)
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


@triton.jit
def _row_filter(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    S: tl.constexpr,
    BR: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    G: tl.constexpr,
):
    rows = tl.arange(0, BR)
    lane = tl.arange(0, L)
    acc = tl.full((BR, L), 0, tl.int32)
    for step in range((R + BR * G - 1) // (BR * G)):
        row = (step.to(tl.int64) * G + tl.program_id(0)) * BR + rows
        q = row
        base = tl.full((BR,), 0, tl.int64)
        for d in tl.static_range(len(SH)):
            if d == len(SH) - 1:
                base += q * ST[d]
            else:
                base += q % SH[d] * ST[d]
                q = q // SH[d]
        raw = tl.load(
            X + base[:, None] + lane[None, :],
            (row[:, None] < R) & (lane[None, :] < (C - 1) * S + 1),
            other=0,
        )
        x = tl.where(lane[None, :] % S == 0, raw, 0).to(tl.float32)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        acc = tl.maximum(acc, bits)
    value = (
        tl.max(tl.max(acc, 1), 0).to(tl.float32, bitcast=True) / 448.0
    ).to(tl.int32, bitcast=True)
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


@triton.jit
def _flat_group(
    X,
    Output,
    R: tl.constexpr,
    C: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
    BR: tl.constexpr,
    L: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, L)
    pm: tl.constexpr = ((1 << 22) + P - 1) // P
    sm: tl.constexpr = ((1 << 16) + S - 1) // S
    col = lane - (lane * pm >> 22) * P
    rem = col - (col * sm >> 16) * S
    take = (lane < BR * P) & (col < (C - 1) * S + 1) & (rem == 0)
    acc = tl.full((L,), 0, tl.int32)
    span: tl.constexpr = (R - 1) * P + (C - 1) * S + 1
    for step in range((R + BR * G - 1) // (BR * G)):
        row = (step.to(tl.int64) * G + tl.program_id(0)) * BR
        offset = row * P + lane
        raw = tl.load(X + offset, (offset < span) & (lane < BR * P), other=0)
        x = tl.where(take, raw, 0).to(tl.float32)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        acc = tl.maximum(acc, bits)
    value = (tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0).to(
        tl.int32, bitcast=True
    )
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


def per_tensor_absmax_fp8(input, output_s):
    n = input.numel()
    if 0 < n <= 8192 and input.is_contiguous():
        _single_small.run(
            input,
            output_s,
            n,
            min(1 << (n - 1).bit_length(), 8192),
            grid=(1,),
            warmup=False,
        )
        return output_s
    linear = input.is_contiguous()
    if not linear:
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
        linear = not shape or (len(shape) == 1 and strides[0] == 1)
    output = triton.reinterpret(output_s, tl.int32)
    if linear:
        block = 8192 if n > 8388608 else 4096
        p = min((n + block - 1) // block, 64)
        if (
            n >= 1048576 or (n >= 32768 and input.dtype == torch.bfloat16)
        ) and input.dtype in (torch.float16, torch.bfloat16):
            _contiguous_native16.run(input, output, n, grid=(p,), warmup=False)
        else:
            _contiguous.run(input, output, n, grid=(p,), warmup=False)
        return output_s
    if (
        input.dtype == torch.float32
        and (not linear)
        and (len(shape) >= 5)
        and (strides[0] == 1)
        and (287 <= shape[0] <= 447)
        and (
            n >= 16777216 or (n >= 4194304 and 7 * shape[0] <= 4 * strides[1])
        )
        and (3 * shape[0] <= 2 * strides[1] <= 4 * shape[0])
    ):
        c = shape[0]
        r = shape[1]
        bc = 1 << (c - 1).bit_length()
        br = 8192 // bc
        tiles = (r + br - 1) // br * (n // (c * r))
        g = min(tiles, 40)
        _highdim_rect.run(
            input,
            output,
            c,
            r,
            strides[1],
            tuple(shape[2:]),
            tuple(strides[2:]),
            br,
            bc,
            tiles,
            g,
            grid=(g,),
            warmup=False,
        )
        return output_s
    if (
        not linear
        and len(shape) == 2
        and (strides[0] == 1)
        and (17 <= shape[0] <= 255)
        and (n >= 4194304)
        and (
            strides[1]
            <= min(512, 2 * shape[0] if shape[0] <= 63 else 5 * shape[0] // 4)
        )
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                8192
                if input.dtype != torch.float32
                and n >= 16777216
                and (r >= 8192 // pitch)
                else 4096
            )
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        g = min(tiles, 40)
        _multi_flat.run(
            input,
            output,
            c,
            r,
            pitch,
            (),
            (),
            br,
            length,
            tiles,
            g,
            grid=(g,),
            warmup=False,
        )
        return output_s
    if (
        not linear
        and len(shape) >= 3
        and (strides[0] == 1)
        and (17 <= shape[0] <= 511)
        and (n >= 65536)
        and (strides[1] <= 512)
        and (
            n < 1048576
            and shape[0] <= 255
            or shape[0] <= 31
            or (
                n >= 4194304
                and (
                    len(shape) == 3
                    or (
                        len(shape) >= 5 and n >= 16777216 and (shape[0] <= 255)
                    )
                )
                and (
                    strides[1]
                    <= (
                        2 * shape[0]
                        if len(shape) >= 5
                        and (
                            input.dtype == torch.float32
                            and shape[0] <= 127
                            or (
                                input.dtype != torch.float32
                                and (
                                    shape[0] <= 95
                                    or (n >= 33554432 and shape[0] <= 191)
                                )
                            )
                        )
                        else 5 * shape[0] // 4
                    )
                )
            )
        )
    ):
        c = shape[0]
        r = shape[1]
        pitch = strides[1]
        br = max(
            1,
            (
                8192
                if input.dtype != torch.float32
                and n >= 16777216
                and (r >= 8192 // pitch)
                else 4096
            )
            // pitch,
        )
        bc = 1 << (c - 1).bit_length()
        use = r >= br and pitch <= 2 * c
        if c <= 31:
            cutoff = (
                65536
                if n < 1048576
                else (
                    1048576
                    if len(shape) == 3 and 2 * pitch <= 3 * bc
                    else 2097152 if c <= 24 else 4194304
                )
            )
        elif n < 1048576:
            cutoff = 65536
        else:
            cutoff = (
                4194304
                if input.dtype == torch.float32 or c <= 127
                else 16777216
            )
            use = (
                use
                and (c <= 255 or input.dtype == torch.float32)
                and (input.dtype != torch.float32 or r >= 1024)
            )
        if use and n >= cutoff:
            length = 1 << (br * pitch - 1).bit_length()
            tiles = (r + br - 1) // br * (n // (c * r))
            g = min(tiles, 40)
            _multi_flat.run(
                input,
                output,
                c,
                r,
                pitch,
                tuple(shape[2:]),
                tuple(strides[2:]),
                br,
                length,
                tiles,
                g,
                grid=(g,),
                warmup=False,
            )
            return output_s
    if (
        not linear
        and len(shape) >= 3
        and (strides[0] == 1)
        and (17 <= shape[0] <= 2048)
        and (n >= 1048576)
    ):
        c = shape[0]
        r = n // c
        length = 1 << (c - 1).bit_length()
        br = min(128, 4096 // length)
        g = min((r + br - 1) // br, 40)
        _row_filter.run(
            input,
            output,
            r,
            c,
            tuple(shape[1:]),
            tuple(strides[1:]),
            1,
            br,
            length,
            length,
            g,
            grid=(g,),
            warmup=False,
        )
        return output_s
    if (
        not linear
        and len(shape) >= 2
        and (1 < strides[0] <= 8)
        and (17 <= shape[0] <= 64)
        and (n >= 131072)
    ):
        c = shape[0]
        r = n // c
        b = 1 << (max(1, c) - 1).bit_length()
        br = min(8, 512 // b)
        length = 1 << (max(1, b * strides[0]) - 1).bit_length()
        if len(shape) == 2 and n >= (
            4194304 if input.dtype == torch.float32 else 524288
        ):
            br = min(
                32,
                2048 // b,
                (8192 if input.dtype == torch.float32 else 16384) // length,
            )
            if input.dtype == torch.bfloat16 and n >= (
                4194304 if c <= 32 else 16777216
            ):
                br = min(64, 4096 // b, 16384 // length)
        cutoff = (
            16777216
            if c > 32
            else 524288 if input.dtype == torch.bfloat16 else 4194304
        )
        if len(shape) == 2 and n >= cutoff and (length <= 256):
            br = min(
                br, (4096 if input.dtype == torch.float32 else 8192) // length
            )
            pitch = strides[1]
            if (
                c <= 32
                and strides[0] <= 5
                and (
                    n
                    >= (
                        2097152
                        if input.dtype == torch.bfloat16
                        and (c <= 24 or strides[0] >= 3)
                        else 4194304
                    )
                )
                and ((c - 1) * strides[0] + 1 <= pitch <= c * (strides[0] + 1))
            ):
                cap = 4096 if input.dtype == torch.float32 else 8192
                br = 1 << (cap // pitch).bit_length() - 1
                group = 1 << (max(1, br * pitch) - 1).bit_length()
                g = min((r + br - 1) // br, 40)
                _flat_group.run(
                    input,
                    output,
                    r,
                    c,
                    pitch,
                    strides[0],
                    br,
                    group,
                    g,
                    grid=(g,),
                    warmup=False,
                )
            else:
                g = min((r + br - 1) // br, 40)
                _row_filter.run(
                    input,
                    output,
                    r,
                    c,
                    tuple(shape[1:]),
                    tuple(strides[1:]),
                    strides[0],
                    br,
                    b,
                    length,
                    g,
                    grid=(g,),
                    warmup=False,
                )
        else:
            g = min((r + br - 1) // br, 40)
            _row_packets.run(
                input,
                output,
                r,
                c,
                tuple(shape[1:]),
                tuple(strides[1:]),
                strides[0],
                br,
                b,
                length,
                g,
                grid=(g,),
                warmup=False,
            )
        return output_s
    if (
        not linear
        and 1 < strides[0] <= 8
        and (shape[0] >= 17)
        and (n >= 65536)
    ):
        c = shape[0]
        r = n // c
        b = (
            1 << (max(1, c) - 1).bit_length()
            if c <= 64
            else (
                1024
                if n >= 8388608 and strides[0] == 3 and (c >= 1024)
                else 512
            )
        )
        length = 1 << (max(1, b * strides[0]) - 1).bit_length()
        g = min(r * ((c + b - 1) // b), 40)
        _local_gather.run(
            input,
            output,
            r,
            c,
            tuple(shape[1:]),
            tuple(strides[1:]),
            strides[0],
            b,
            length,
            g,
            grid=(g,),
            warmup=False,
        )
        return output_s
    if (
        len(shape) == 2
        and strides[0] == 1
        and (shape[0] <= 512)
        and (strides[1] <= 8 * shape[0])
        and (n >= 65536)
        and (n >= 262144 or (shape[0] <= 64 and strides[1] <= 4 * shape[0]))
    ):
        bc = 1 << (max(1, shape[0]) - 1).bit_length()
        br = min(128, 2048 // bc)
        g = min((shape[1] + br - 1) // br, 40)
        _rows2d.run(
            input,
            output,
            shape[1],
            shape[0],
            strides[1],
            br,
            bc,
            g,
            grid=(g,),
            warmup=False,
        )
    elif (
        len(shape) == 2
        and strides[0] == 1
        and (512 < shape[0] <= 2048)
        and (strides[1] <= 4 * shape[0])
        and (n >= 4194304)
    ):
        bc = 1 << (max(1, shape[0]) - 1).bit_length()
        br = max(1, 8192 // bc)
        g = min((shape[1] + br - 1) // br, 40)
        _rows2d.run(
            input,
            output,
            shape[1],
            shape[0],
            strides[1],
            br,
            bc,
            g,
            grid=(g,),
            warmup=False,
        )
    elif (
        len(shape) == 2
        and strides[0] == 1
        and (17 <= shape[0] <= 2147479552)
        and (shape[1] * ((shape[0] + 4095) // 4096) < 2147483607)
    ):
        b = min(1 << (max(1, shape[0]) - 1).bit_length(), 4096)
        g = min(shape[1] * ((shape[0] + b - 1) // b), 40)
        _rows.run(
            input,
            output,
            shape[1],
            shape[0],
            strides[1],
            b,
            g,
            grid=(g,),
            warmup=False,
        )
    else:
        b = min(1 << (max(1, n) - 1).bit_length(), 1024)
        g = min((n + b - 1) // b, 40)
        _parallel_strided.run(
            input,
            output,
            n,
            tuple(reversed(shape)),
            tuple(reversed(strides)),
            b,
            g,
            grid=(g,),
            warmup=False,
        )
    return output_s


@triton.jit
def _multi_flat(
    X,
    Output,
    C: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    BR: tl.constexpr,
    L: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, L)
    groups: tl.constexpr = (R + BR - 1) // BR
    acc = tl.full((L,), 0, tl.int32)
    for step in range((T + G - 1) // G):
        if (T + G - 1) // G * G > 2147483647:
            tile = step.to(tl.int64) * G + tl.program_id(0)
        else:
            tile = step * G + tl.program_id(0)
        row = (tile % groups).to(tl.int64) * BR
        q = (tile // groups).to(tl.int64)
        base = tl.full((), 0, tl.int64)
        for d in tl.static_range(len(SH)):
            if d == len(SH) - 1:
                base += q * ST[d]
            else:
                base += q % SH[d] * ST[d]
                q = q // SH[d]
        local = row * P + lane
        raw = tl.load(
            X + base + local,
            (tile < T) & (lane < BR * P) & (local < (R - 1) * P + C),
            other=0,
        )
        m: tl.constexpr = ((1 << 22) + P - 1) // P
        col = lane - (lane * m >> 22) * P
        x = tl.where(col < C, raw, 0).to(tl.float32)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        acc = tl.maximum(acc, bits)
    value = (tl.max(acc, 0).to(tl.float32, bitcast=True) / 448.0).to(
        tl.int32, bitcast=True
    )
    if G == 1:
        tl.store(Output, value)
    else:
        tl.atomic_max(Output, value, sem="relaxed")


@triton.jit
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
    T: tl.constexpr,
    G: tl.constexpr,
):
    groups: tl.constexpr = (R + BR - 1) // BR
    rows = tl.arange(0, BR)
    cols = tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.int32)
    for step in range((T + G - 1) // G):
        tile = step.to(tl.int64) * G + tl.program_id(0)
        row = tile % groups * BR + rows
        q = tile // groups
        base = tl.full((), 0, tl.int64)
        for d in tl.static_range(len(SH)):
            if d == len(SH) - 1:
                base += q * ST[d]
            else:
                base += q % SH[d] * ST[d]
                q = q // SH[d]
        x = tl.load(
            X + base + row[:, None] * P + cols[None, :],
            (tile < T) & (row[:, None] < R) & (cols[None, :] < C),
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
        tl.store(Output, bits)
    else:
        tl.atomic_max(Output, bits, sem="relaxed")


@triton.jit
def _contiguous_native16(X, Output, N: tl.constexpr):
    B: tl.constexpr = (
        8192
        if N > 8388608 or (X.dtype.element_ty != tl.float32 and N >= 1048576)
        else 4096
    )
    G: tl.constexpr = (N + B - 1) // B if N <= B * 64 else 64
    lane = tl.arange(0, B)
    acc = tl.full(
        (B,), 0, tl.int32 if X.dtype.element_ty == tl.float32 else tl.int16
    )
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    if X.dtype.element_ty == tl.bfloat16 and N >= 262144 and (N <= 2097152):
        for tile in range(tl.program_id(0), tl.cdiv(N, B), G):
            if WIDE:
                i = tile.to(tl.int64) * B + lane
            else:
                i = tile * B + lane
            x = tl.load(X + i, i < N, other=0)
            if X.dtype.element_ty == tl.float32:
                bits = x.to(tl.int32, bitcast=True) & 2147483647
            else:
                bits = x.to(tl.int16, bitcast=True) & tl.full(
                    (), 32767, tl.int16
                )
            acc = tl.maximum(acc, bits)
    else:
        for step in range(tl.cdiv(N, G * B)):
            if WIDE:
                i = (step.to(tl.int64) * G + tl.program_id(0)) * B + lane
            else:
                i = (step * G + tl.program_id(0)) * B + lane
            x = tl.load(X + i, i < N, other=0)
            if X.dtype.element_ty == tl.float32:
                bits = x.to(tl.int32, bitcast=True) & 2147483647
            else:
                bits = x.to(tl.int16, bitcast=True) & tl.full(
                    (), 32767, tl.int16
                )
            acc = tl.maximum(acc, bits)
    v = tl.reduce(acc, 0, _native_max_combine).to(tl.int32)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    if G == 1:
        tl.store(Output, value.to(tl.int32, bitcast=True))
    else:
        tl.atomic_max(Output, value.to(tl.int32, bitcast=True), sem="relaxed")


@triton.jit
def _native_max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _single_small(X, Output, N: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    acc = tl.full(
        (B,), 0, tl.int32 if X.dtype.element_ty == tl.float32 else tl.int16
    )
    for start in range(0, N, B):
        x = tl.load(X + start + lane, start + lane < N, other=0)
        if X.dtype.element_ty == tl.float32:
            bits = x.to(tl.int32, bitcast=True) & 2147483647
        else:
            bits = x.to(tl.int16, bitcast=True) & tl.full((), 32767, tl.int16)
        acc = tl.maximum(acc, bits)
    v = tl.reduce(acc, 0, _native_max_combine).to(tl.int32)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    tl.store(Output, maximum / 448.0)


__all__ = ["per_tensor_absmax_fp8"]
