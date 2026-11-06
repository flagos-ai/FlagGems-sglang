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


@triton.jit(do_not_specialize=["P"])
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
    CACHE: tl.constexpr = "",
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
        x = tl.load(X + off, i < N, other=0, cache_modifier=CACHE).to(
            tl.float32
        )
        acc = tl.maximum(acc, x.to(tl.int32, bitcast=True) & 2147483647)
    v = tl.max(acc, 0).to(tl.float32, bitcast=True)
    if FINAL:
        v = v / 448.0
    if FINAL:
        tl.store(P, v)
    else:
        bits = (v / 448.0).to(tl.uint32, bitcast=True)
        tl.atomic_max(P.to(tl.pointer_type(tl.uint32)), bits, sem="relaxed")


@triton.jit(do_not_specialize=["Output"])
def _contiguous(X, Output, N: tl.constexpr):
    B: tl.constexpr = 1024 if N <= 1024 else 2048 if N <= 2048 else 4096
    G: tl.constexpr = (N + B - 1) // B if N <= B * 512 else 512
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int32)
    WIDE: tl.constexpr = (N + G * B - 1) // (G * B) * G * B > 2147483647
    for step in range((N + G * B - 1) // (G * B)):
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
        tl.store(Output, value)
    else:
        tl.atomic_max(
            Output.to(tl.pointer_type(tl.int32)),
            value.to(tl.int32, bitcast=True),
            sem="relaxed",
        )


@triton.jit(do_not_specialize=["Output"])
def _native4k(X, Output):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    x = tl.load(X + i, eviction_policy="evict_first")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["Output"])
def _small(X, Output, N: tl.constexpr):
    B: tl.constexpr = 1024 if N <= 1024 else 2048 if N <= 2048 else 4096
    i = tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0)
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.store(Output, value)


@triton.jit(do_not_specialize=["Output"])
def _small1024(X, Output):
    _small(X, Output, 1024)


@triton.jit(do_not_specialize=["Output"])
def _small2048(X, Output):
    _small(X, Output, 2048)


@triton.jit(do_not_specialize=["Output"])
def _small4096(X, Output):
    _small(X, Output, 4096)


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


@triton.jit(do_not_specialize=["Output"])
def _native4k_hip(X, Output):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    x = tl.load(X + i, cache_modifier=".cg")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
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
        torch.version.hip is not None
        and len(shape) >= 3
        and (strides[0] == 1)
        and (shape[0] <= 4096)
        and (n // shape[0] <= 2147483647)
    ):
        c = shape[0]
        r = shape[1]
        bc = 1 << (c - 1).bit_length()
        tiles = n // c
        _hip_row_cg1540.run(
            input,
            output_s,
            c,
            r,
            strides[1],
            tuple(shape[2:]),
            tuple(strides[2:]),
            1,
            bc,
            grid=(tiles,),
            num_warps=1,
            warmup=False,
        )
        return output_s
    if (
        torch.version.hip is None
        and input.dtype == torch.float32
        and (
            len(shape) >= 5
            or (len(shape) >= 3 and n < 8388608 and (shape[0] >= 513))
        )
        and (strides[0] == 1)
        and (128 <= shape[0] <= 767)
        and (shape[0] != 512 or n >= 16777216)
        and (n >= 1048576)
        and (strides[1] <= 2 * shape[0])
    ):
        c = shape[0]
        r = shape[1]
        bc = 1 << (c - 1).bit_length()
        br = 4096 // bc
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        _highdim_rect.run(
            input,
            output_s,
            c,
            r,
            strides[1],
            tuple(shape[2:]),
            tuple(strides[2:]),
            br,
            bc,
            grid=(tiles,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if torch.version.hip is None and (
        3 <= len(shape) <= 4
        and strides[0] == 1
        and (
            257 <= shape[0] <= 511
            and (
                len(shape) == 4 and n >= 33554432
                if input.dtype == torch.float32
                else shape[0] <= 319
            )
        )
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
            (8192 if input.dtype != torch.float32 and n >= 16777216 else 4096)
            // pitch,
        )
        length = 1 << (br * pitch - 1).bit_length()
        tiles = (r + br - 1) // br
        for d in shape[2:]:
            tiles *= d
        if input.dtype == torch.float32 or n < 16777216:
            _physical_rows.run(
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
        torch.version.hip is None
        and len(shape) >= 3
        and (strides[0] == 1)
        and (256 <= shape[0] <= 4096)
        and (n >= 1048576)
    ):
        c = shape[0]
        bc = 1 << (c - 1).bit_length()
        use = input.dtype != torch.float32 or (
            c >= 512 and c * 4 >= bc * 3 and (n >= 4194304)
        )
        cap = 4096 if input.dtype != torch.float32 and c * 4 < bc * 3 else 2048
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
        and (input.dtype in (torch.float16, torch.bfloat16))
        and (n >= 1048576)
        and (torch.version.hip is None)
    ):
        _stride2_native.run(
            input,
            output_s,
            n,
            grid=((n + 4095) // 4096,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if (
        len(shape) == 1
        and strides[0] == 2
        and (input.dtype == torch.float32)
        and (n >= 1048576)
        and (torch.version.hip is None)
    ):
        _stride2_native32.run(
            input,
            output_s,
            n,
            grid=((n + 4095) // 4096,),
            num_warps=4,
            warmup=False,
        )
        return output_s
    if (
        torch.version.hip is None
        and input.dtype == torch.float32
        and (len(shape) == 1)
        and (24 <= strides[0] <= 256)
        and (n >= 4194304)
    ):
        _stride_native1k.run(
            input,
            output_s,
            n,
            strides[0],
            "",
            grid=((n + 1023) // 1024,),
            num_warps=16,
            warmup=False,
        )
        return output_s
    if (
        len(shape) == 1
        and 3 <= strides[0] <= 8
        and (n >= 4194304)
        and (torch.version.hip is None)
    ):
        hip = torch.version.hip is not None
        _stride_native.run(
            input,
            output_s,
            n,
            strides[0],
            ".cg" if hip else "",
            grid=((n + 4095) // 4096,),
            num_warps=1 if hip else 4,
            warmup=False,
        )
        return output_s
    if not shape or (len(shape) == 1 and strides[0] == 1):
        return per_tensor_absmax_fp8(input.as_strided((n,), (1,)), output_s)
    if (
        len(shape) == 2
        and strides[0] == 1
        and (shape[0] <= 8192)
        and (strides[1] <= 4 * shape[0])
        and (n >= 1048576)
    ):
        bc = 1 << (max(1, shape[0]) - 1).bit_length()
        br = max(1, min(128, 2048 // bc))
        g = min((shape[1] + br - 1) // br, 1024)
        (_rows2d_cg1557 if torch.version.hip is not None else _rows2d).run(
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
    if len(shape) >= 2 and n >= 1048576 and (torch.version.hip is None):
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
    b = min(1 << (max(1, n) - 1).bit_length(), 4096)
    p = min((n + b - 1) // b, 256)
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
        ".cg" if torch.version.hip is not None else "",
    )
    return output_s


def per_tensor_absmax_fp8(input, output_s):
    n = input.numel()
    if (
        torch.version.hip is None
        and input.dtype != torch.float32
        and (8388608 < n <= 16777216)
        and (n % 8192 == 0)
        and input.is_contiguous()
        and (input.data_ptr() % 16 == 0)
    ):
        _vector8192_1861.run(
            input, output_s, grid=(n // 8192,), num_warps=32, warmup=False
        )
        return output_s
    if (
        torch.version.hip is None
        and input.dtype == torch.float16
        and (n > 67108864)
        and (n % 4096 == 0)
        and input.is_contiguous()
        and (input.data_ptr() % 16 == 0)
    ):
        _vector128_1847.run(
            input, output_s, grid=(n // 4096,), num_warps=16, warmup=False
        )
        return output_s
    if (
        torch.version.hip is None
        and input.dtype == torch.float32
        and (n >= 4194304)
        and (n % 4096 != 0)
        and input.is_contiguous()
    ):
        _scalar_tail1436.run(
            input,
            output_s,
            n,
            grid=((n + 4095) // 4096,),
            num_warps=32,
            warmup=False,
        )
        return output_s
    if (
        torch.version.hip is not None
        and n >= 8192
        and (n % 4096 != 0)
        and input.is_contiguous()
    ):
        b = 2048 if n <= 2097152 and input.dtype != torch.float32 else 4096
        (
            _hip_branch1698
            if 2097152 < n <= 8388608
            and input.dtype != torch.float32
            and (input.data_ptr() % 16 == 0)
            else _hip_overlap1681
        ).run(
            input,
            output_s,
            n,
            grid=((n + b - 1) // b,),
            num_warps=1,
            warmup=False,
        )
        return output_s
    if input.is_contiguous():
        if n == 1024:
            _small1024.run(input, output_s, grid=(1,), warmup=False)
        elif n == 2048:
            _small2048.run(input, output_s, grid=(1,), warmup=False)
        elif n == 4096:
            _small4096.run(input, output_s, grid=(1,), warmup=False)
        elif n <= 4096:
            _small.run(input, output_s, n, grid=(1,), warmup=False)
        elif n % 4096 == 0:
            if torch.version.hip is not None:
                if n <= 2097152 and input.dtype != torch.float32:
                    _native2k_hip1569.run(
                        input,
                        output_s,
                        grid=(n // 2048,),
                        num_warps=1,
                        warmup=False,
                    )
                else:
                    _native4k_hip.run(
                        input,
                        output_s,
                        grid=(n // 4096,),
                        num_warps=1,
                        warmup=False,
                    )
            elif n > 8388608 or (
                n >= 1048576 and input.dtype == torch.float32
            ):
                _native4k_scalar.run(
                    input,
                    output_s,
                    grid=(n // 4096,),
                    num_warps=32 if input.dtype == torch.float32 else 16,
                    warmup=False,
                )
            else:
                _native4k.run(
                    input,
                    output_s,
                    grid=(n // 4096,),
                    num_warps=16 if n <= 8388608 else 8,
                    warmup=False,
                )
        elif torch.version.hip is None and (
            n >= 8388608 or (input.dtype == torch.float32 and n >= 2097152)
        ):
            if n > 8388608 and n <= 67108864 and (input.data_ptr() % 16 == 0):
                _scalar_tail1436.run(
                    input,
                    output_s,
                    n,
                    grid=((n + 4095) // 4096,),
                    num_warps=16,
                    warmup=False,
                )
            else:
                _native4k_tail.run(
                    input,
                    output_s,
                    n,
                    grid=((n + 4095) // 4096,),
                    num_warps=16 if n <= 8388608 else 8,
                    warmup=False,
                )
        else:
            p = min((n + 4095) // 4096, 512)
            _contiguous.run(input, output_s, n, grid=(p,), warmup=False)
    else:
        return _layout(input, output_s)
    return output_s


@triton.jit(do_not_specialize=["Output"])
def _native4k_tail(X, Output, N: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        x = tl.load(X + i, eviction_policy="evict_first")
    else:
        x = tl.load(X + i, i < N, other=0, eviction_policy="evict_first")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _stride2_native(X, Output, N: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        x = tl.load(X + i * 2)
    else:
        x = tl.load(X + i * 2, i < N, other=0)
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    maximum = tl.max(bits, 0)
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
def _stride2_native32(X, Output, N: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    if (tl.program_id(0).to(tl.int64) + 1) * 4096 <= N:
        x = tl.load(X + i * 2)
    else:
        x = tl.load(X + i * 2, i < N, other=0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    maximum = tl.max(bits, 0)
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


@triton.jit(do_not_specialize=["X", "Output"])
def _native4k_scalar(X, Output):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 4096)
    x = tl.load(X + i, eviction_policy="evict_first")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["X", "Output"])
def _scalar_tail1436(X, Output, N: tl.constexpr):
    start = tl.minimum(tl.program_id(0).to(tl.int64) * 4096, N - 4096)
    i = start + tl.arange(0, 4096)
    x = tl.load(X + i, eviction_policy="evict_first")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize_on_alignment=["X"])
def _hip_row_cg1540(
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
        cache_modifier=".cg",
    ).to(tl.float32)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    value = tl.max(tl.max(bits, 1), 0).to(tl.float32, bitcast=True) / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit
def _rows2d_cg1557(
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
            cache_modifier=".cg",
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


@triton.jit(do_not_specialize=["Output"])
def _native2k_hip1569(X, Output):
    i = tl.program_id(0).to(tl.int64) * 2048 + tl.arange(0, 2048)
    x = tl.load(X + i, cache_modifier=".cg")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["Output"])
def _hip_overlap1681(X, Output, N: tl.constexpr):
    B: tl.constexpr = (
        2048 if N <= 2097152 and X.dtype.element_ty != tl.float32 else 4096
    )
    start = tl.minimum(tl.program_id(0).to(tl.int64) * B, N - B)
    i = start + tl.arange(0, B)
    x = tl.load(X + i, cache_modifier=".cg")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["Output"])
def _hip_branch1698(X, Output, N: tl.constexpr):
    B: tl.constexpr = (
        2048 if N <= 2097152 and X.dtype.element_ty != tl.float32 else 4096
    )
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    if (pid + 1) * B <= N:
        x = tl.load(X + pid * B + lane, cache_modifier=".cg")
    else:
        x = tl.load(X + (N - B) + lane, cache_modifier=".cg")
    if X.dtype.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32) & 32767
    v = tl.max(bits, 0)
    if X.dtype.element_ty == tl.bfloat16:
        maximum = (v << 16).to(tl.float32, bitcast=True)
    elif X.dtype.element_ty == tl.float16:
        maximum = v.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        maximum = v.to(tl.float32, bitcast=True)
    value = maximum / 448.0
    tl.atomic_max(
        Output.to(tl.pointer_type(tl.int32)),
        value.to(tl.int32, bitcast=True),
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["X", "Output"])
def _vector128_1847(X, Output):
    i = tl.program_id(0).to(tl.int64) * 4096 + tl.arange(0, 512) * 8
    a, b, c, d = tl.inline_asm_elementwise(
        (
            "{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 "
            "policy, 1.0; ld.global.L1::evict_first.L2::cache_hint.v4.u32 {$0"
            ", $1, $2, $3}, [$4], policy; }"
        ),
        constraints="=r,=r,=r,=r,l",
        args=[(X + i).to(tl.uint64)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    lo = tl.maximum(
        tl.maximum(a & 32767, b & 32767), tl.maximum(c & 32767, d & 32767)
    )
    hi = tl.maximum(
        tl.maximum(a >> 16 & 32767, b >> 16 & 32767),
        tl.maximum(c >> 16 & 32767, d >> 16 & 32767),
    )
    maximum = tl.max(tl.maximum(lo, hi), 0)
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
def _vector8192_1861(X, Output):
    i = tl.program_id(0).to(tl.int64) * 8192 + tl.arange(0, 1024) * 8
    a, b, c, d = tl.inline_asm_elementwise(
        (
            "{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 "
            "policy, 1.0; ld.global.L1::evict_first.L2::cache_hint.v4.u32 {$0"
            ", $1, $2, $3}, [$4], policy; }"
        ),
        constraints="=r,=r,=r,=r,l",
        args=[(X + i).to(tl.uint64)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    lo = tl.maximum(
        tl.maximum(a & 32767, b & 32767), tl.maximum(c & 32767, d & 32767)
    )
    hi = tl.maximum(
        tl.maximum(a >> 16 & 32767, b >> 16 & 32767),
        tl.maximum(c >> 16 & 32767, d >> 16 & 32767),
    )
    maximum = tl.max(tl.maximum(lo, hi), 0)
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


__all__ = ["per_tensor_absmax_fp8"]
