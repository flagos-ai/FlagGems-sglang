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


@triton.jit(do_not_specialize=["P", "Output"])
def _finish_pair_int1614(P, Output):
    i = tl.arange(0, 16)
    x = tl.load(P + i, i < 12, other=0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit
def _is_multiple(index, STEP: tl.constexpr):
    if STEP % 2 == 0 or index.dtype == tl.int64:
        return index % STEP == 0
    else:
        INV0: tl.constexpr = 1
        INV1: tl.constexpr = INV0 * (2 - STEP * INV0) & 4294967295
        INV2: tl.constexpr = INV1 * (2 - STEP * INV1) & 4294967295
        INV3: tl.constexpr = INV2 * (2 - STEP * INV2) & 4294967295
        INV4: tl.constexpr = INV3 * (2 - STEP * INV3) & 4294967295
        INV5: tl.constexpr = INV4 * (2 - STEP * INV4) & 4294967295
        product = index.to(tl.uint32) * tl.full((), INV5, tl.uint32)
        return product <= 4294967295 // STEP


@triton.jit
def _maxsum(a, b, c, d):
    return (tl.maximum(a, c), b + d)


@triton.jit(do_not_specialize=["X", "P"])
def _contiguous(X, P, N: tl.constexpr):
    B: tl.constexpr = (
        16384
        if X.dtype.element_ty != tl.float32 and N > 393216 and (N <= 524288)
        else (
            16384
            if X.dtype.element_ty != tl.float32
            and N > 786432
            and (N <= 2097152)
            else (
                16384
                if N >= 32768 and N <= 49152
                else triton.next_power_of_2(
                    min(
                        N,
                        (
                            16384
                            if X.dtype.element_ty == tl.float32 and N >= 262144
                            else 32768
                        ),
                    )
                )
            )
        )
    )
    G: tl.constexpr = 1 if N <= 262144 else 12
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    U: tl.constexpr = (
        1
        if N < 262144
        or (
            G == 12
            and N
            <= 2 * G * (16384 if X.dtype.element_ty == tl.float32 else 32768)
        )
        else 3
    )
    FIRST: tl.constexpr = U if U == 3 and N > 16777216 else 1
    GAP: tl.constexpr = 1 if U == 3 and N > 16777216 else G
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, U * G * B)
        increment = 1
    else:
        begin = pid * FIRST * B
        end = N
        increment = U * G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * FIRST * B + index.to(tl.int64) * (U * G * B)
        else:
            start = index
        if U == 3:
            base0 = tl.minimum(start, N - B)
            base1 = tl.minimum(start + GAP * B, N - B)
            base2 = tl.minimum(start + 2 * GAP * B, N - B)
            raw0 = tl.load(X + base0 + lane, offset_state_policy="continuous")
            raw1 = tl.load(X + base1 + lane, offset_state_policy="continuous")
            raw2 = tl.load(X + base2 + lane, offset_state_policy="continuous")
            x = tl.abs(raw0.to(tl.float32))
            y = tl.abs(raw1.to(tl.float32))
            z = tl.abs(raw2.to(tl.float32))
            if N > 16777216:
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
                seen = seen + x + (y + z)
            else:
                seen = seen + x + (y + z)
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
        else:
            for j in tl.static_range(U):
                i = start + j * G * B + lane
                if N % (U * G * B) == 0:
                    x = tl.load(X + i).to(tl.float32)
                elif N >= B:
                    base = tl.minimum(start + j * G * B, N - B)
                    x = tl.load(
                        X + base + lane, offset_state_policy="continuous"
                    ).to(tl.float32)
                else:
                    x = tl.load(X + i, i < N, other=0).to(tl.float32)
                magnitude = tl.abs(x)
                seen = seen + magnitude
                acc = tl.maximum(acc, magnitude)
    maximum, total = tl.reduce((acc, seen), 0, _maxsum)
    is_nan = total != total
    v = tl.where(is_nan, float("nan"), maximum)
    if G == 1:
        v = v / 448.0
    tl.store(P + tl.program_id(0), v)


@triton.jit
def _strided(
    X,
    Output,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, G * B)
        increment = 1
    else:
        begin = pid * B
        end = N
        increment = G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * B + index.to(tl.int64) * (G * B)
        else:
            start = index
        i = start + lane
        q = i
        off = tl.full((B,), 0, tl.int64)
        for d in tl.static_range(len(SH) - 1, -1, -1):
            off += q % SH[d] * ST[d]
            q = q // SH[d]
        if len(SH) > 1:
            x = tl.load(
                X + off, i < N, other=0, offset_state_policy="unknown"
            ).to(tl.float32)
        else:
            x = tl.load(X + off, i < N, other=0).to(tl.float32)
        magnitude = tl.abs(x)
        acc = tl.maximum(acc, magnitude)
        seen = seen + magnitude
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(Output + tl.program_id(0), value)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish(P, Output):
    i = tl.arange(0, 16)
    x = tl.load(P + i, i < 12, other=0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit
def _stepped(
    X,
    Output,
    S: tl.constexpr,
    STEP: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    if S > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(S, G * B)
        increment = 1
    else:
        begin = tl.program_id(0) * B
        end = S
        increment = G * B
    for index in range(begin, end, increment):
        if S > 1073741824:
            start = tl.program_id(0).to(tl.int64) * B + index.to(tl.int64) * (
                G * B
            )
        else:
            start = index
        i = tl.minimum(start, S - B) + lane
        raw = tl.load(X + i, offset_state_policy="continuous").to(tl.float32)
        value = tl.abs(tl.where(_is_multiple(i, STEP), raw, 0.0))
        acc = tl.maximum(acc, value)
        seen = seen + value
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(Output + tl.program_id(0), value)


@triton.jit
def _linear(
    X, P, N: tl.constexpr, STEP: tl.constexpr, B: tl.constexpr, G: tl.constexpr
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, G * B)
        increment = 1
    else:
        begin = pid * B
        end = N
        increment = G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * B + index.to(tl.int64) * (G * B)
        else:
            start = index
        base = tl.minimum(start, N - B)
        offsets = ((base + lane) % N).to(tl.int64) * STEP
        value = tl.abs(
            tl.load(X + offsets, offset_state_policy="unknown").to(tl.float32)
        )
        acc = tl.maximum(acc, value)
        seen = seen + value
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _batch_load(
    X,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
    tile,
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    T: tl.constexpr = triton.cdiv(C, B)
    lane = tl.arange(0, BR * B)
    row = tl.minimum(tile // T * BR, R - BR) + lane // B
    col = tl.minimum(tile % T * B, C - B) + lane % B
    if len(SH) == 2 and (R - 1) * ST[1] + C <= 2147483647:
        base_row = tl.minimum(tile // T * BR, R - BR).to(tl.int64)
        X = X + base_row * ST[1]
        off = lane // B * ST[1]
    else:
        q = row
        off = tl.full((BR * B,), 0, tl.int64)
        for d in tl.static_range(1, len(SH)):
            off += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
    if BR == 1:
        raw = tl.load(X + off + col, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off + col, offset_state_policy="locally_continuous")
    return raw


@triton.jit
def _row_batch(
    X,
    P,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
    G: tl.constexpr,
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    T: tl.constexpr = triton.cdiv(C, B)
    TOTAL: tl.constexpr = triton.cdiv(R, BR) * T
    U: tl.constexpr = (
        4
        if len(SH) >= 2
        and X.dtype.element_ty != tl.float32
        and (C >= 256)
        and (B < 4096)
        and (2 * B < C)
        and (C >= 512 or N <= 1048576)
        else (
            4
            if len(SH) == 2
            and X.dtype.element_ty != tl.float32
            and (N >= 524288)
            and (C >= 256)
            and (B < C)
            and (C <= B * 5 // 4)
            and (TOTAL >= 8 * G)
            and (G == 1 or TOTAL >= 128 * G)
            else (
                3
                if (C >= 64 or (C < 8 and N >= 524288)) and TOTAL >= 6 * G
                else 1
            )
        )
    )
    acc = tl.full((BR * B,), 0, tl.float32)
    seen = tl.full((BR * B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    for tile in range(pid * U, TOTAL, G * U):
        raw0 = _batch_load(X, SH, ST, N, B, BR, tl.minimum(tile, TOTAL - 1))
        if U >= 3:
            raw1 = _batch_load(
                X, SH, ST, N, B, BR, tl.minimum(tile + 1, TOTAL - 1)
            )
            raw2 = _batch_load(
                X, SH, ST, N, B, BR, tl.minimum(tile + 2, TOTAL - 1)
            )
            if U == 4:
                raw3 = _batch_load(
                    X, SH, ST, N, B, BR, tl.minimum(tile + 3, TOTAL - 1)
                )
        x = tl.abs(raw0.to(tl.float32))
        if U >= 3:
            y = tl.abs(raw1.to(tl.float32))
            z = tl.abs(raw2.to(tl.float32))
            if U == 4:
                w = tl.abs(raw3.to(tl.float32))
                acc = tl.maximum(
                    acc, tl.maximum(tl.maximum(x, y), tl.maximum(z, w))
                )
                seen = seen + (x + y + (z + w))
            else:
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
                seen = seen + x + (y + z)
        else:
            acc = tl.maximum(acc, x)
            seen = seen + x
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _split_load(
    X,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
    tile,
    W: tl.constexpr,
):
    C: tl.constexpr = W
    R: tl.constexpr = N // SH[0]
    T: tl.constexpr = triton.cdiv(C, B)
    lane = tl.arange(0, BR * B)
    row = tl.minimum(tile // T * BR, R - BR) + lane // B
    col = tl.minimum(tile % T * B, C - B) + lane % B
    if len(SH) == 2 and (R - 1) * ST[1] + C <= 2147483647:
        base_row = tl.minimum(tile // T * BR, R - BR).to(tl.int64)
        X = X + base_row * ST[1]
        off = lane // B * ST[1]
    elif (
        X.dtype.element_ty == tl.float32
        and len(SH) > 2
        and (SH[1] % BR == 0)
        and ((SH[1] - 1) * ST[1] + C <= 2147483647)
    ):
        base_row = tl.minimum(tile // T * BR, R - BR)
        q = base_row // SH[1]
        outer = tl.full((), 0, tl.int64)
        for d in tl.static_range(2, len(SH)):
            outer += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
        X = X + outer
        off = (base_row % SH[1] + lane // B) * ST[1]
    else:
        q = row
        off = tl.full((BR * B,), 0, tl.int64)
        for d in tl.static_range(1, len(SH)):
            off += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
    if BR == 1:
        raw = tl.load(X + off + col, offset_state_policy="continuous")
    else:
        raw = tl.load(X + off + col, offset_state_policy="locally_continuous")
    return raw


@triton.jit
def _split_row_batch(
    X,
    P,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
    G: tl.constexpr,
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    T: tl.constexpr = C // B
    TOTAL: tl.constexpr = triton.cdiv(R, BR) * T
    U: tl.constexpr = (
        4
        if len(SH) >= 2
        and X.dtype.element_ty != tl.float32
        and (C >= 256)
        and (B < 4096)
        and (2 * B < C)
        and (C >= 512 or N <= 1048576)
        else (
            4
            if len(SH) == 2
            and X.dtype.element_ty != tl.float32
            and (N >= 524288)
            and (C >= 256)
            and (B < C)
            and (C <= B * 5 // 4)
            and (TOTAL >= 8 * G)
            and (G == 1 or TOTAL >= 128 * G)
            else (
                3
                if (C >= 64 or (C < 8 and N >= 524288)) and TOTAL >= 6 * G
                else 1
            )
        )
    )
    acc = tl.full((BR * B,), 0, tl.float32)
    seen = tl.full((BR * B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    for tile in range(pid * U, TOTAL, G * U):
        raw0 = _split_load(
            X, SH, ST, N, B, BR, tl.minimum(tile, TOTAL - 1), T * B
        )
        if U >= 3:
            raw1 = _split_load(
                X, SH, ST, N, B, BR, tl.minimum(tile + 1, TOTAL - 1), T * B
            )
            raw2 = _split_load(
                X, SH, ST, N, B, BR, tl.minimum(tile + 2, TOTAL - 1), T * B
            )
            if U == 4:
                raw3 = _split_load(
                    X, SH, ST, N, B, BR, tl.minimum(tile + 3, TOTAL - 1), T * B
                )
        x = tl.abs(raw0.to(tl.float32))
        if U >= 3:
            y = tl.abs(raw1.to(tl.float32))
            z = tl.abs(raw2.to(tl.float32))
            if U == 4:
                w = tl.abs(raw3.to(tl.float32))
                acc = tl.maximum(
                    acc, tl.maximum(tl.maximum(x, y), tl.maximum(z, w))
                )
                seen = seen + (x + y + (z + w))
            else:
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
                seen = seen + x + (y + z)
        else:
            acc = tl.maximum(acc, x)
            seen = seen + x
    TB: tl.constexpr = max(
        triton.next_power_of_2(C % B),
        triton.next_power_of_2(triton.cdiv(16384, R)),
        (
            32
            if X.dtype.element_ty == tl.float32
            and len(SH) > 2
            and (SH[1] % 1024 != 0 or (SH[1] - 1) * ST[1] + C > 2147483647)
            else 16
        ),
    )
    TR: tl.constexpr = 16384 // TB
    for rowtile in range(pid, triton.cdiv(R, TR), G):
        raw = _split_load(X + C - TB, SH, ST, N, TB, TR, rowtile, TB)
        magnitude = tl.abs(raw.to(tl.float32))
        acc = tl.maximum(acc, magnitude)
        seen = seen + magnitude
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _flat_gather(
    X,
    Output,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, G * B)
        increment = 1
    else:
        begin = pid * B
        end = N
        increment = G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * B + index.to(tl.int64) * (G * B)
        else:
            start = index
        i = tl.minimum(start, N - B) + lane
        q = i
        off = tl.full((B,), 0, tl.int64)
        for d in tl.static_range(len(SH) - 1, -1, -1):
            off += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
        x = tl.load(X + off, offset_state_policy="unknown").to(tl.float32)
        magnitude = tl.abs(x)
        acc = tl.maximum(acc, magnitude)
        seen = seen + magnitude
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(Output + tl.program_id(0), value)


@triton.jit
def _span_packets(X, P, S: tl.constexpr, STEP: tl.constexpr, G: tl.constexpr):
    B: tl.constexpr = 16384 if X.dtype.element_ty == tl.float32 else 32768
    U: tl.constexpr = 3
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    for start in range(tl.program_id(0) * U * B, S, U * G * B):
        base0 = tl.minimum(start, S - B)
        raw0 = tl.load(X + base0 + lane, offset_state_policy="continuous")
        if U >= 2:
            base1 = tl.minimum(start + B, S - B)
            raw1 = tl.load(X + base1 + lane, offset_state_policy="continuous")
        if U >= 3:
            base2 = tl.minimum(start + 2 * B, S - B)
            raw2 = tl.load(X + base2 + lane, offset_state_policy="continuous")
        x = tl.abs(
            tl.where(
                _is_multiple(base0 + lane, STEP), raw0.to(tl.float32), 0.0
            )
        )
        if U >= 2:
            y = tl.abs(
                tl.where(
                    _is_multiple(base1 + lane, STEP), raw1.to(tl.float32), 0.0
                )
            )
        if U >= 3:
            z = tl.abs(
                tl.where(
                    _is_multiple(base2 + lane, STEP), raw2.to(tl.float32), 0.0
                )
            )
        if U == 3:
            acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
            seen = seen + x + (y + z)
        elif U == 2:
            acc = tl.maximum(acc, tl.maximum(x, y))
            seen = seen + (x + y)
        else:
            acc = tl.maximum(acc, x)
            seen = seen + x
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _row_packets(
    X, P, SH: tl.constexpr, ST: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    STEP: tl.constexpr = ST[0]
    S: tl.constexpr = (C - 1) * STEP + 1
    U: tl.constexpr = 3 if STEP >= 8 else 1
    T: tl.constexpr = triton.cdiv(S, U * B)
    G: tl.constexpr = 12
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    for tile in range(pid, R * T, G):
        q = tile // T
        offset = tl.full((), 0, tl.int64)
        for d in tl.static_range(1, len(SH)):
            offset += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
        start = tile % T * U * B
        base0 = tl.minimum(start, S - B)
        raw0 = tl.load(
            X + offset + base0 + lane, offset_state_policy="continuous"
        )
        if U >= 2:
            base1 = tl.minimum(start + B, S - B)
            raw1 = tl.load(
                X + offset + base1 + lane, offset_state_policy="continuous"
            )
        if U >= 3:
            base2 = tl.minimum(start + 2 * B, S - B)
            raw2 = tl.load(
                X + offset + base2 + lane, offset_state_policy="continuous"
            )
        x = tl.abs(
            tl.where(
                _is_multiple(base0 + lane, STEP), raw0.to(tl.float32), 0.0
            )
        )
        if U >= 2:
            y = tl.abs(
                tl.where(
                    _is_multiple(base1 + lane, STEP), raw1.to(tl.float32), 0.0
                )
            )
        if U >= 3:
            z = tl.abs(
                tl.where(
                    _is_multiple(base2 + lane, STEP), raw2.to(tl.float32), 0.0
                )
            )
        if U == 3:
            acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
            seen = seen + x + (y + z)
        elif U == 2:
            acc = tl.maximum(acc, tl.maximum(x, y))
            seen = seen + (x + y)
        else:
            acc = tl.maximum(acc, x)
            seen = seen + x
    total = tl.sum(seen, 0)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _static_phase(X, P, N: tl.constexpr, STEP: tl.constexpr, G: tl.constexpr):
    B: tl.constexpr = (
        2048
        if STEP % 2 == 1
        else (
            32768 if STEP == 32 and X.dtype.element_ty != tl.float32 else 16384
        )
    )
    U: tl.constexpr = 3 if B == 32768 else 4
    S: tl.constexpr = (N - 1) * STEP + 1
    K: tl.constexpr = B // STEP * STEP
    END: tl.constexpr = (S - B) // STEP * STEP
    lane = tl.arange(0, B)
    valid = lane % STEP == 0
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    for start in range(tl.program_id(0) * U * K, S, G * U * K):
        base0 = tl.minimum(start, END)
        base1 = tl.minimum(start + K, END)
        base2 = tl.minimum(start + 2 * K, END)
        raw0 = tl.load(X + base0 + lane, offset_state_policy="continuous")
        raw1 = tl.load(X + base1 + lane, offset_state_policy="continuous")
        raw2 = tl.load(X + base2 + lane, offset_state_policy="continuous")
        if U == 4:
            base3 = tl.minimum(start + 3 * K, END)
            raw3 = tl.load(X + base3 + lane, offset_state_policy="continuous")
        x = tl.abs(tl.where(valid, raw0.to(tl.float32), 0.0))
        y = tl.abs(tl.where(valid, raw1.to(tl.float32), 0.0))
        z = tl.abs(tl.where(valid, raw2.to(tl.float32), 0.0))
        if U == 4:
            w = tl.abs(tl.where(valid, raw3.to(tl.float32), 0.0))
            acc = tl.maximum(
                acc, tl.maximum(tl.maximum(x, y), tl.maximum(z, w))
            )
            seen = seen + (x + y + (z + w))
        else:
            acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
            seen = seen + x + (y + z)
    last = tl.abs(tl.load(X + (N - 1) * STEP).to(tl.float32))
    prior = tl.abs(tl.load(X + (N - 2) * STEP).to(tl.float32))
    acc = tl.maximum(acc, tl.maximum(last, prior))
    total = tl.sum(seen, 0) + (last + prior)
    value = tl.where(total != total, float("nan"), tl.max(acc, 0))
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _static_rows(
    X, P, SH: tl.constexpr, ST: tl.constexpr, N: tl.constexpr, G: tl.constexpr
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    STEP: tl.constexpr = ST[0]
    S: tl.constexpr = (C - 1) * STEP + 1
    if STEP % 2 == 1:
        U: tl.constexpr = 4
        B: tl.constexpr = 2048
    elif C < 16384:
        U: tl.constexpr = 4
        T4: tl.constexpr = triton.cdiv(S, 4 * (4096 // STEP * STEP))
        T8: tl.constexpr = triton.cdiv(S, 4 * (8192 // STEP * STEP))
        B: tl.constexpr = (
            2048
            if STEP == 3
            else 4096 if STEP == 6 and 4 * T4 <= 7 * T8 else 8192
        )
    else:
        LARGE: tl.constexpr = (
            STEP == 12 and X.dtype.element_ty != tl.float32 and (C >= 65536)
        )
        U: tl.constexpr = (
            3
            if LARGE
            or (
                STEP == 12
                and X.dtype.element_ty == tl.float32
                and (C <= 32768)
            )
            else 4
        )
        T8: tl.constexpr = triton.cdiv(S, 4 * (8192 // STEP * STEP))
        T16: tl.constexpr = triton.cdiv(S, 4 * (16384 // STEP * STEP))
        B: tl.constexpr = (
            32768
            if LARGE
            else 8192 if STEP != 12 and 4 * T8 <= 7 * T16 else 16384
        )
    K: tl.constexpr = B // STEP * STEP
    END: tl.constexpr = (S - B) // STEP * STEP
    T: tl.constexpr = triton.cdiv(S, U * K)
    lane = tl.arange(0, B)
    valid = lane % STEP == 0
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    for tile in range(pid, R * T, G):
        q = tile // T
        offset = tl.full((), 0, tl.int64)
        for d in tl.static_range(1, len(SH)):
            offset += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
        start = (tile % T * (U * K)).to(tl.int32)
        base0 = tl.minimum(start, END)
        raw0 = tl.load(
            X + offset + base0 + lane, offset_state_policy="continuous"
        )
        if U >= 2:
            base1 = tl.minimum(start + K, END)
            raw1 = tl.load(
                X + offset + base1 + lane, offset_state_policy="continuous"
            )
        if U >= 3:
            base2 = tl.minimum(start + 2 * K, END)
            raw2 = tl.load(
                X + offset + base2 + lane, offset_state_policy="continuous"
            )
        if U == 4:
            base3 = tl.minimum(start + 3 * K, END)
            raw3 = tl.load(
                X + offset + base3 + lane, offset_state_policy="continuous"
            )
        x = tl.abs(tl.where(valid, raw0.to(tl.float32), 0.0))
        if U >= 2:
            y = tl.abs(tl.where(valid, raw1.to(tl.float32), 0.0))
        if U >= 3:
            z = tl.abs(tl.where(valid, raw2.to(tl.float32), 0.0))
        if U == 4:
            w = tl.abs(tl.where(valid, raw3.to(tl.float32), 0.0))
            acc = tl.maximum(
                acc, tl.maximum(tl.maximum(x, y), tl.maximum(z, w))
            )
            seen = seen + (x + y + (z + w))
        elif U == 3:
            acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
            seen = seen + x + (y + z)
        elif U == 2:
            acc = tl.maximum(acc, tl.maximum(x, y))
            seen = seen + (x + y)
        else:
            acc = tl.maximum(acc, x)
            seen = seen + x
    tail_max = tl.full((), 0, tl.float32)
    tail_sum = tl.full((), 0, tl.float32)
    if G == 1 or R < 128:
        for row in range(pid, R, G):
            q = row
            offset = tl.full((), 0, tl.int64)
            for d in tl.static_range(1, len(SH)):
                offset += (q % SH[d]).to(tl.int64) * ST[d]
                q = q // SH[d]
            last = tl.abs(tl.load(X + offset + (C - 1) * STEP).to(tl.float32))
            prior = tl.abs(tl.load(X + offset + (C - 2) * STEP).to(tl.float32))
            tail_max = tl.maximum(tail_max, tl.maximum(last, prior))
            tail_sum = tail_sum + (last + prior)
    total = tl.sum(seen, 0) + tail_sum
    value = tl.where(
        total != total, float("nan"), tl.maximum(tl.max(acc, 0), tail_max)
    )
    if G == 1:
        value = value / 448.0
    tl.store(P + tl.program_id(0), value)


@triton.jit
def _tail_finish(
    X,
    P,
    Output,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
):
    C: tl.constexpr = SH[0]
    R: tl.constexpr = N // C
    lane = tl.arange(0, B)
    acc = tl.load(P + lane % 12, offset_state_policy="unknown")
    seen = acc
    for row in range(0, 2 * R, B):
        i = tl.minimum(row, 2 * R - B) + lane
        q = i // 2
        offset = tl.full((B,), 0, tl.int64)
        for d in tl.static_range(1, len(SH)):
            offset += (q % SH[d]).to(tl.int64) * ST[d]
            q = q // SH[d]
        value = tl.abs(
            tl.load(
                X + offset + (C - 2 + i % 2) * ST[0],
                offset_state_policy="unknown",
            ).to(tl.float32)
        )
        acc = tl.maximum(acc, value)
        seen = seen + value
    total = tl.sum(seen, 0)
    result = tl.where(total != total, float("nan"), tl.max(acc, 0))
    tl.store(Output, result / 448.0)


def per_tensor_absmax_fp8(input, output_s):
    n = input.numel()
    if (
        327680 < n < 393216
        and input.dtype != torch.float32
        and input.is_contiguous()
    ):
        partial = torch.empty((12,), device=input.device, dtype=torch.float32)
        _direct12_1724.run(
            input,
            partial,
            n,
            grid=(12,),
            warmup=False,
            buffer_size_limit=2048,
            isCloseMemoryAsync=False,
        )
        _finish_pair_int1614.run(
            partial,
            output_s,
            grid=(1,),
            warmup=False,
            isClusterOneCoreActOnly=True,
        )
        return output_s
    if (
        131072 < n <= 163840
        and input.dtype == torch.float32
        and input.is_contiguous()
    ):
        partial = torch.empty((10,), device=input.device, dtype=torch.float32)
        _fp32_ten1671.run(
            input,
            partial,
            n,
            grid=(10,),
            warmup=False,
            buffer_size_limit=2048,
            isCloseMemoryAsync=False,
        )
        _finish10_1658.run(
            partial,
            output_s,
            grid=(1,),
            warmup=False,
            isClusterOneCoreActOnly=True,
        )
        return output_s
    if (
        262144 < n <= 327680
        and input.dtype != torch.float32
        and input.is_contiguous()
    ):
        partial = torch.empty((10,), device=input.device, dtype=torch.float32)
        _direct10_1658.run(
            input,
            partial,
            n,
            grid=(10,),
            warmup=False,
            buffer_size_limit=2048,
            isCloseMemoryAsync=False,
        )
        _finish10_1658.run(
            partial,
            output_s,
            grid=(1,),
            warmup=False,
            isClusterOneCoreActOnly=True,
        )
        return output_s
    if (
        131072 < n <= 196608
        and input.dtype != torch.float32
        and input.is_contiguous()
    ):
        partial = torch.empty((6,), device=input.device, dtype=torch.float32)
        _low_direct6_1634.run(
            input,
            partial,
            n,
            grid=(6,),
            warmup=False,
            buffer_size_limit=2048,
            isCloseMemoryAsync=False,
        )
        _finish6_1634.run(
            partial,
            output_s,
            grid=(1,),
            warmup=False,
            isClusterOneCoreActOnly=True,
        )
        return output_s
    if input.is_contiguous() and (
        8192 <= n <= 98304
        and (input.dtype != torch.float32 or n >= 16384)
        or (input.dtype == torch.float16 and 98304 < n <= 131072)
    ):
        (
            _small_split1394
            if input.dtype == torch.float32 and 65536 < n <= 81920
            else _small_static1371
        ).run(
            input, output_s, n, grid=(1,), warmup=False, buffer_size_limit=2048
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
    if not linear:
        b = min(1 << (max(1, n) - 1).bit_length(), 16384)
        sh = tuple(reversed(shape))
        st = tuple(reversed(strides))
        if (
            len(shape) >= 2
            and strides[0] == 1
            and (
                sum(((d - 1) * s for d, s in zip(shape, strides))) < 2147483647
            )
        ):
            b = min(1 << shape[0].bit_length() - 1, 4096)
            if (
                (
                    len(shape) == 2
                    or n <= 1572864
                    or n
                    >= (8388608 if input.dtype == torch.float32 else 16777216)
                )
                and shape[0] >= 256
                and (n >= 524288)
                and (n <= 1572864 or n >= 8388608)
            ):
                smaller = max(
                    32,
                    b
                    // (
                        8
                        if input.dtype == torch.float32 and shape[0] >= 2048
                        else 4
                    ),
                )
                old_work = (shape[0] + b - 1) // b * b
                new_work = (shape[0] + smaller - 1) // smaller * smaller
                if new_work * 8 <= old_work * 7:
                    b = smaller
            br = min(
                1 << (n // shape[0]).bit_length() - 1,
                (8192 if shape[0] < 8 and n >= 524288 else 16384) // b,
            )
            work = (
                (n // shape[0] + br - 1)
                // br
                * ((shape[0] + b - 1) // b)
                * br
                * b
            )
            budget = 1310720 if input.dtype == torch.float32 else 1572864
            extra = (
                (len(shape) == 2 or shape[0] >= 512)
                and shape[0] >= 64
                and (work <= budget)
            )
            g = 1 if (n <= 1048576 or extra) and n // shape[0] <= 32768 else 12
            row_kernel = _row_batch
            full_b = min(1 << shape[0].bit_length() - 1, 4096)
            large_split = n >= (
                16777216
                if input.dtype == torch.float32 or shape[0] < 512
                else 33554432
            )
            small_split = (
                len(shape) == 2 and shape[0] < 512 and (524288 <= n <= 1048576)
            )
            rows = n // shape[0]
            if (
                256 <= shape[0] <= 4128
                and rows >= 512
                and (0 < shape[0] % full_b <= 32)
            ):
                split = large_split or small_split
                if (
                    1048576 < n <= budget
                    and rows <= 32768
                    and (
                        len(shape) == 2
                        or (
                            g != 1
                            and (
                                input.dtype == torch.float32
                                or shape[0] >= 512
                                or n <= 1310720
                            )
                        )
                    )
                ):
                    main_rows = 16384 // full_b
                    tail_min = (
                        32
                        if input.dtype == torch.float32
                        and len(shape) > 2
                        and (
                            shape[1] % 1024 != 0
                            or (shape[1] - 1) * strides[1] + shape[0]
                            > 2147483647
                        )
                        else 16
                    )
                    tail_b = max(
                        tail_min,
                        1 << (shape[0] % full_b - 1).bit_length(),
                        1 << ((16384 + rows - 1) // rows - 1).bit_length(),
                    )
                    tail_rows = 16384 // tail_b
                    split_work = (rows + main_rows - 1) // main_rows * (
                        shape[0] // full_b
                    ) * 16384 + (rows + tail_rows - 1) // tail_rows * 16384
                    if split_work <= budget:
                        split = True
                        g = 1
                if split:
                    b = full_b
                    br = 16384 // b
                    row_kernel = _split_row_batch
            partial = (
                output_s
                if g == 1
                else torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
            )
            row_kernel.run(
                input,
                partial,
                tuple(shape),
                tuple(strides),
                n,
                b,
                br,
                g,
                grid=(g,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            if g != 1:
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif (
            len(shape) > 1
            and shape[0] >= 2048
            and (n >= 16384)
            and (n // shape[0] <= 32)
            and (strides[0] in (3, 5, 7))
            and (n // shape[0] * ((shape[0] - 1) * strides[0] + 1) <= 524288)
        ):
            _static_rows.run(
                input,
                output_s,
                tuple(shape),
                tuple(strides),
                n,
                1,
                grid=(1,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
        elif (
            len(shape) > 1
            and (
                4096 <= shape[0] < 16384
                and n >= 524288
                and (strides[0] in (3, 5, 6, 7, 9, 11, 12))
                or (
                    shape[0] >= 16384
                    and n >= 1048576
                    and (
                        strides[0]
                        in (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 20, 24)
                        or (
                            strides[0] == 32
                            and (
                                input.dtype != torch.float32
                                or shape[0] >= 65536
                            )
                        )
                    )
                )
            )
            and ((shape[0] - 1) * strides[0] + 1 <= 1073741824)
        ):
            partial = torch.empty(
                (12,), device=input.device, dtype=torch.float32
            )
            _static_rows.run(
                input,
                partial,
                tuple(shape),
                tuple(strides),
                n,
                12,
                grid=(12,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            rows = n // shape[0]
            if rows < 128:
                _finish.run(partial, output_s, grid=(1,), warmup=False)
            else:
                bt = min(1024, 1 << (2 * rows).bit_length() - 1)
                _tail_finish.run(
                    input,
                    partial,
                    output_s,
                    tuple(shape),
                    tuple(strides),
                    n,
                    bt,
                    grid=(1,),
                    warmup=False,
                    buffer_size_limit=2048,
                    isCloseMemoryAsync=False,
                )
        elif (
            len(shape) > 1
            and shape[0] >= 4096
            and (strides[0] in (2, 3, 4, 8, 16))
            and (n >= 524288)
            and ((shape[0] - 1) * strides[0] + 1 <= 1073741824)
        ):
            span = (shape[0] - 1) * strides[0] + 1
            b = min(
                1 << span.bit_length() - 1,
                16384 if input.dtype == torch.float32 else 32768,
            )
            partial = torch.empty(
                (12,), device=input.device, dtype=torch.float32
            )
            _row_packets.run(
                input,
                partial,
                tuple(shape),
                tuple(strides),
                n,
                b,
                grid=(12,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif len(shape) > 1:
            b = min(1 << n.bit_length() - 1, 1024)
            g = 1 if n <= 49152 else 12
            partial = (
                output_s
                if g == 1
                else torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
            )
            _flat_gather.run(
                input,
                partial,
                n,
                sh,
                st,
                b,
                g,
                grid=(g,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            if g != 1:
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif (
            len(shape) == 1
            and (n - 1) * strides[0] + 1 <= 1073741824
            and (
                n >= 524288
                and strides[0] in (3, 5, 7, 9, 11, 13)
                or (
                    n >= 1048576
                    and (
                        strides[0]
                        in (2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 20, 24, 32)
                        or (
                            input.dtype == torch.float32
                            and strides[0] in (4, 8, 16)
                        )
                    )
                )
                or (
                    strides[0] in (3, 5, 7, 9, 11, 13)
                    and n >= 8192
                    and ((n - 1) * strides[0] + 1 <= 786432)
                )
                or (
                    strides[0] in (6, 12)
                    and 32768 <= n <= (131072 if strides[0] == 6 else 98304)
                )
            )
        ):
            g = 1 if n < 524288 else 12
            partial = (
                output_s
                if g == 1
                else torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
            )
            _static_phase.run(
                input,
                partial,
                n,
                strides[0],
                g,
                grid=(g,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            if g != 1:
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif len(shape) == 1 and (
            strides[0] in (2, 4, 8, 16, 32)
            and (
                n >= 32768
                and (n - 1) * strides[0] + 1 <= 1572864
                or (
                    strides[0] >= 4
                    and n >= 524288
                    and ((n - 1) * strides[0] + 1 <= 1073741824)
                )
            )
            or (
                strides[0] in (3, 5)
                and n >= (4194304 if strides[0] == 3 else 2097152)
                and ((n - 1) * strides[0] + 1 <= 1073741824)
            )
        ):
            span = (n - 1) * strides[0] + 1
            g = 1 if span <= 1572864 else 12
            partial = (
                output_s
                if g == 1
                else torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
            )
            _span_packets.run(
                input,
                partial,
                span,
                strides[0],
                g,
                grid=(g,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
            if g != 1:
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif len(shape) == 1 and (
            (n - 1) * strides[0] > 2147483647
            or n < 4096
            or (
                strides[0] >= 3
                and (strides[0] % 2 == 0 or strides[0] >= 7 or n <= 32768)
            )
        ):
            if n <= 32768:
                _linear.run(
                    input,
                    output_s,
                    n,
                    strides[0],
                    min(1024, 1 << n.bit_length() - 1),
                    1,
                    grid=(1,),
                    warmup=False,
                    buffer_size_limit=2048,
                    isCloseMemoryAsync=False,
                )
            else:
                partial = torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
                _linear.run(
                    input,
                    partial,
                    n,
                    strides[0],
                    1024,
                    12,
                    grid=(12,),
                    warmup=False,
                    buffer_size_limit=2048,
                    isCloseMemoryAsync=False,
                )
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif (
            len(shape) == 1
            and strides[0] % 2 == 1
            and (
                3 <= strides[0] <= 17
                or (strides[0] <= 65 and (n - 1) * strides[0] + 1 <= 8388608)
            )
        ):
            span = (n - 1) * strides[0] + 1
            b = 1 << min(span, 16384).bit_length() - 1
            g = 1 if span <= 262144 else 12
            if g == 1:
                _stepped.run(
                    input,
                    output_s,
                    span,
                    strides[0],
                    b,
                    g,
                    grid=(g,),
                    warmup=False,
                    buffer_size_limit=2048,
                )
            else:
                partial = torch.empty(
                    (12,), device=input.device, dtype=torch.float32
                )
                _stepped.run(
                    input,
                    partial,
                    span,
                    strides[0],
                    b,
                    g,
                    grid=(g,),
                    warmup=False,
                    buffer_size_limit=2048,
                )
                _finish.run(partial, output_s, grid=(1,), warmup=False)
        elif n <= (65536 if strides[0] % 2 == 1 else 262144):
            _strided.run(
                input,
                output_s,
                n,
                sh,
                st,
                b,
                1,
                grid=(1,),
                warmup=False,
                buffer_size_limit=2048,
            )
        else:
            partial = torch.empty(
                (12,), device=input.device, dtype=torch.float32
            )
            _strided.run(
                input,
                partial,
                n,
                sh,
                st,
                b,
                12,
                grid=(12,),
                warmup=False,
                buffer_size_limit=2048,
            )
            _finish.run(partial, output_s, grid=(1,), warmup=False)
    elif n <= 262144:
        if n > 98304:
            if n > 131072:
                partial = torch.empty(
                    (8,), device=input.device, dtype=torch.float32
                )
                if input.dtype != torch.float32 and 196608 < n <= 262144:
                    _low_direct_full8.run(
                        input,
                        partial,
                        n,
                        grid=(8,),
                        warmup=False,
                        buffer_size_limit=2048,
                        isCloseMemoryAsync=False,
                    )
                else:
                    _lowstage8.run(
                        input,
                        partial,
                        n,
                        grid=(8,),
                        warmup=False,
                        buffer_size_limit=2048,
                        isCloseMemoryAsync=False,
                    )
                _finish_low8.run(
                    partial,
                    output_s,
                    grid=(1,),
                    warmup=False,
                    isClusterOneCoreActOnly=True,
                )
            else:
                partial = torch.empty(
                    (4,), device=input.device, dtype=torch.float32
                )
                _low_direct4.run(
                    input,
                    partial,
                    n,
                    grid=(4,),
                    warmup=False,
                    buffer_size_limit=2048,
                    isCloseMemoryAsync=False,
                )
                _finish_low.run(
                    partial,
                    output_s,
                    grid=(1,),
                    warmup=False,
                    isClusterOneCoreActOnly=True,
                )
        elif n >= 262144:
            _contiguous.run(
                input,
                output_s,
                n,
                grid=(1,),
                warmup=False,
                buffer_size_limit=2048,
                isCloseMemoryAsync=False,
            )
        else:
            _contiguous.run(
                input,
                output_s,
                n,
                grid=(1,),
                warmup=False,
                buffer_size_limit=2048,
            )
    else:
        partial = torch.empty((12,), device=input.device, dtype=torch.float32)
        if (
            input.dtype != torch.float32
            and 262144 < n <= 786432
            and (n % 393216 == 0)
        ):
            _mid_direct12.run(
                input,
                partial,
                n,
                grid=(12,),
                warmup=False,
                isCloseMemoryAsync=False,
                buffer_size_limit=2048,
            )
        else:
            _contiguous.run(
                input,
                partial,
                n,
                grid=(12,),
                warmup=False,
                isCloseMemoryAsync=False,
                buffer_size_limit=2048,
            )
        if 393216 < n <= 2097152 and input.dtype == torch.bfloat16:
            _finish_pair_int1614.run(
                partial,
                output_s,
                grid=(1,),
                warmup=False,
                isClusterOneCoreActOnly=True,
            )
        else:
            (_finish_pair if n <= 6291456 else _finish).run(
                partial, output_s, grid=(1,), warmup=False
            )
    return output_s


@triton.jit(do_not_specialize=["X", "P"])
def _lowstage(X, P, N: tl.constexpr):
    B: tl.constexpr = triton.next_power_of_2(
        min(
            N,
            (
                16384
                if X.dtype.element_ty == tl.float32 and N >= 262144
                else 32768
            ),
        )
    )
    G: tl.constexpr = 4
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    U: tl.constexpr = 1
    FIRST: tl.constexpr = U if U == 3 and N > 16777216 else 1
    GAP: tl.constexpr = 1 if U == 3 and N > 16777216 else G
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, U * G * B)
        increment = 1
    else:
        begin = pid * FIRST * B
        end = N
        increment = U * G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * FIRST * B + index.to(tl.int64) * (U * G * B)
        else:
            start = index
        if U == 3:
            base0 = tl.minimum(start, N - B)
            base1 = tl.minimum(start + GAP * B, N - B)
            base2 = tl.minimum(start + 2 * GAP * B, N - B)
            raw0 = tl.load(X + base0 + lane, offset_state_policy="continuous")
            raw1 = tl.load(X + base1 + lane, offset_state_policy="continuous")
            raw2 = tl.load(X + base2 + lane, offset_state_policy="continuous")
            x = tl.abs(raw0.to(tl.float32))
            y = tl.abs(raw1.to(tl.float32))
            z = tl.abs(raw2.to(tl.float32))
            if N > 16777216:
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
                seen = seen + x + (y + z)
            else:
                seen = seen + x + (y + z)
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
        else:
            for j in tl.static_range(U):
                i = start + j * G * B + lane
                if N % (U * G * B) == 0:
                    x = tl.load(X + i).to(tl.float32)
                elif N >= B:
                    base = tl.minimum(start + j * G * B, N - B)
                    x = tl.load(
                        X + base + lane, offset_state_policy="continuous"
                    ).to(tl.float32)
                else:
                    x = tl.load(X + i, i < N, other=0).to(tl.float32)
                magnitude = tl.abs(x)
                seen = seen + magnitude
                acc = tl.maximum(acc, magnitude)
    maximum, total = tl.reduce((acc, seen), 0, _maxsum)
    is_nan = total != total
    v = tl.where(is_nan, float("nan"), maximum)
    if G == 1:
        v = v / 448.0
    tl.store(P + tl.program_id(0), v)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish_low(P, Output):
    i = tl.arange(0, 4)
    x = tl.load(P + i)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit(do_not_specialize=["X", "P"])
def _lowstage8(X, P, N: tl.constexpr):
    B: tl.constexpr = 16384 if X.dtype.element_ty == tl.float32 else 32768
    G: tl.constexpr = 8
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    U: tl.constexpr = 1
    FIRST: tl.constexpr = U if U == 3 and N > 16777216 else 1
    GAP: tl.constexpr = 1 if U == 3 and N > 16777216 else G
    pid = tl.program_id(0)
    if N > 1073741824:
        pid = pid.to(tl.int64)
    if N > 1073741824:
        begin = tl.full((), 0, tl.int32)
        end = triton.cdiv(N, U * G * B)
        increment = 1
    else:
        begin = pid * FIRST * B
        end = N
        increment = U * G * B
    for index in range(begin, end, increment):
        if N > 1073741824:
            start = pid * FIRST * B + index.to(tl.int64) * (U * G * B)
        else:
            start = index
        if U == 3:
            base0 = tl.minimum(start, N - B)
            base1 = tl.minimum(start + GAP * B, N - B)
            base2 = tl.minimum(start + 2 * GAP * B, N - B)
            raw0 = tl.load(X + base0 + lane, offset_state_policy="continuous")
            raw1 = tl.load(X + base1 + lane, offset_state_policy="continuous")
            raw2 = tl.load(X + base2 + lane, offset_state_policy="continuous")
            x = tl.abs(raw0.to(tl.float32))
            y = tl.abs(raw1.to(tl.float32))
            z = tl.abs(raw2.to(tl.float32))
            if N > 16777216:
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
                seen = seen + x + (y + z)
            else:
                seen = seen + x + (y + z)
                acc = tl.maximum(tl.maximum(acc, x), tl.maximum(y, z))
        else:
            for j in tl.static_range(U):
                i = start + j * G * B + lane
                if N % (U * G * B) == 0:
                    x = tl.load(X + i).to(tl.float32)
                elif N >= B:
                    base = tl.minimum(start + j * G * B, N - B)
                    x = tl.load(
                        X + base + lane, offset_state_policy="continuous"
                    ).to(tl.float32)
                else:
                    x = tl.load(X + i, i < N, other=0).to(tl.float32)
                magnitude = tl.abs(x)
                seen = seen + magnitude
                acc = tl.maximum(acc, magnitude)
    maximum, total = tl.reduce((acc, seen), 0, _maxsum)
    is_nan = total != total
    v = tl.where(is_nan, float("nan"), maximum)
    if G == 1:
        v = v / 448.0
    tl.store(P + tl.program_id(0), v)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish_low8(P, Output):
    i = tl.arange(0, 8)
    x = tl.load(P + i)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit(do_not_specialize=["X", "P"])
def _low_direct_full8(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (8 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


@triton.jit(do_not_specialize=["X", "P"])
def _mid_direct12(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    x = tl.abs(tl.load(X + pid * B + lane).to(tl.float32))
    if N > 393216:
        y = tl.abs(tl.load(X + (pid + 12) * B + lane).to(tl.float32))
        maximum = tl.max(tl.maximum(x, y), 0)
        total = tl.sum(x + y, 0)
    else:
        maximum = tl.max(x, 0)
        total = tl.sum(x, 0)
    tl.store(P + pid, tl.where(total != total, float("nan"), maximum))


@triton.jit(do_not_specialize=["X", "P"])
def _low_direct4(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (4 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


@triton.jit(do_not_specialize=["X", "Output"])
def _small_static1371(X, Output, N: tl.constexpr):
    B: tl.constexpr = (
        triton.next_power_of_2(N + 1) // 2
        if N < 32768
        else 16384 if N <= 49152 else 32768
    )
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    for j in tl.static_range((N + B - 1) // B):
        start = min(j * B, N - B)
        x = tl.abs(
            tl.load(X + start + lane, offset_state_policy="continuous").to(
                tl.float32
            )
        )
        acc = tl.maximum(acc, x)
        seen = seen + x
    maximum, total = tl.reduce((acc, seen), 0, _maxsum)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(Output, value / 448.0)


@triton.jit(do_not_specialize=["X", "Output"])
def _small_split1394(X, Output, N: tl.constexpr):
    B: tl.constexpr = (
        triton.next_power_of_2(N + 1) // 2 if N < 32768 else 16384
    )
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    seen = tl.full((B,), 0, tl.float32)
    for j in tl.static_range((N + B - 1) // B):
        start = min(j * B, N - B)
        x = tl.abs(
            tl.load(X + start + lane, offset_state_policy="continuous").to(
                tl.float32
            )
        )
        acc = tl.maximum(acc, x)
        seen = seen + x
    maximum, total = tl.reduce((acc, seen), 0, _maxsum)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(Output, value / 448.0)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish_pair(P, Output):
    i = tl.arange(0, 16)
    x = tl.load(P + i, i < 12, other=0)
    maximum, total = tl.reduce((x, x), 0, _maxsum)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(Output, value / 448.0)


@triton.jit(do_not_specialize=["X", "P"])
def _low_direct6_1634(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (6 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish6_1634(P, Output):
    i = tl.arange(0, 8)
    x = tl.load(P + i, i < 6, other=0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit(do_not_specialize=["X", "P"])
def _direct10_1658(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (10 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


@triton.jit(do_not_specialize=["P", "Output"])
def _finish10_1658(P, Output):
    i = tl.arange(0, 16)
    x = tl.load(P + i, i < 10, other=0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    tl.store(Output, tl.max(bits, 0).to(tl.float32, bitcast=True) / 448.0)


@triton.jit(do_not_specialize=["X", "P"])
def _fp32_ten1671(X, P, N: tl.constexpr):
    B: tl.constexpr = 16384
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (10 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


@triton.jit(do_not_specialize=["X", "P"])
def _direct12_1724(X, P, N: tl.constexpr):
    B: tl.constexpr = 32768
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if N % (12 * B) == 0:
        x = tl.load(X + pid * B + lane).to(tl.float32)
    else:
        base = tl.minimum(pid * B, N - B)
        x = tl.load(X + base + lane, offset_state_policy="continuous").to(
            tl.float32
        )
    magnitude = tl.abs(x)
    maximum = tl.max(magnitude, 0)
    total = tl.sum(magnitude, 0)
    value = tl.where(total != total, float("nan"), maximum)
    tl.store(P + pid, value)


__all__ = ["per_tensor_absmax_fp8"]
