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
import triton as tr
import triton.language as tl


@tr.jit
def _encode_short(v):
    bits = v.to(tl.int32, bitcast=True)
    mag = bits & 2147483647
    normal = tl.minimum(mag, 1138753536) + 67633151 + (mag >> 20 & 1) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.int32, bitcast=True)
    code = tl.where(mag < 1015021568, subnormal, normal)
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _copy(
    X,
    output_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    XS: tl.constexpr,
    XC: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i // D * XS + i % D * XC, i < N * D, 0)
    tl.store(output_ptr + i, value, i < N * D)


@tr.jit
def _pack_bytes2(X, R, Q, M: tl.constexpr, BR: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    R = R.to(tl.pointer_type(tl.bfloat16))
    Q = Q.to(tl.pointer_type(tl.uint8))
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    XS: tl.constexpr = M[4]
    XC: tl.constexpr = M[5]
    RS: tl.constexpr = M[6]
    RC: tl.constexpr = M[7]
    NG: tl.constexpr = D // G
    QS: tl.constexpr = M[8]
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G)
    RB: tl.constexpr = tr.next_power_of_2(DR)
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    valid = (
        (groups[:, None] < N * NG)
        & (c[None, :] < G)
        & (tl.arange(0, BG)[:, None] < BR * NG)
    )
    if N % BR == 0 and BG == BR * NG and (BC == G) and (RB == DR):
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC
        ).to(tl.float32)
        a = tl.max(tl.abs(x), 1)
        scale = a / 448.0
        v = x / scale[:, None]
        quant = _encode_short(v)
        tl.store(
            Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :], quant
        )
        scale_bits = scale.to(tl.int32, bitcast=True)
        tl.store(
            Q + rows * QS + D + tiles * 4,
            scale_bits.to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 1,
            (scale_bits >> 8).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 2,
            (scale_bits >> 16).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 3,
            (scale_bits >> 24).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, 2 * RB)
        mask = (rr[:, None] < N) & (rc[None, :] < 2 * DR)
        rope = (
            tl.load(R + rr[:, None] * RS + rc[None, :] // 2 * RC, mask, 0)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
        )
        tl.store(
            Q + rr[:, None] * QS + D + NG * 4 + rc[None, :],
            (rope >> rc[None, :] % 2 * 8).to(tl.uint8),
            mask,
        )
    else:
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
            valid,
            0,
        ).to(tl.float32)
        a = tl.max(tl.abs(x), 1)
        scale = a / 448.0
        v = x / scale[:, None]
        quant = _encode_short(v)
        tl.store(
            Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :],
            quant,
            valid,
        )
        scale_bits = scale.to(tl.int32, bitcast=True)
        tl.store(
            Q + rows * QS + D + tiles * 4,
            scale_bits.to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 1,
            (scale_bits >> 8).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 2,
            (scale_bits >> 16).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        tl.store(
            Q + rows * QS + D + tiles * 4 + 3,
            (scale_bits >> 24).to(tl.uint8),
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, 2 * RB)
        mask = (rr[:, None] < N) & (rc[None, :] < 2 * DR)
        rope = (
            tl.load(R + rr[:, None] * RS + rc[None, :] // 2 * RC, mask, 0)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
        )
        tl.store(
            Q + rr[:, None] * QS + D + NG * 4 + rc[None, :],
            (rope >> rc[None, :] % 2 * 8).to(tl.uint8),
            mask,
        )


@tr.jit
def _pack_addr(X, R, Q, S, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    R = R.to(tl.pointer_type(tl.int16))
    Q = Q.to(tl.pointer_type(tl.uint8))
    S = S.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int16))
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    XS: tl.constexpr = M[4]
    XC: tl.constexpr = M[5]
    RS: tl.constexpr = M[6]
    RC: tl.constexpr = M[7]
    NG: tl.constexpr = D // G
    QS: tl.constexpr = M[8]
    SS: tl.constexpr = QS // 4
    OS: tl.constexpr = QS // 2
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G)
    RB: tl.constexpr = tr.next_power_of_2(DR)
    S = S + D // 4
    output_ptr = output_ptr + (D + NG * 4) // 2
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    valid = (
        (groups[:, None] < N * NG)
        & (c[None, :] < G)
        & (tl.arange(0, BG)[:, None] < BR * NG)
    )
    if N % BR == 0 and BG == BR * NG and (BC == G) and (RB == DR):
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC
        ).to(tl.float32)
        a = tl.max(tl.abs(x), 1)
        scale = a / 448.0
        v = x / scale[:, None]
        quant = _encode_short(v)
        tl.store(
            Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :], quant
        )
        tl.store(S + rows * SS + tiles, scale)
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, RB)
        mask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(R + rr[:, None] * RS + rc[None, :] * RC)
        tl.store(output_ptr + rr[:, None] * OS + rc[None, :], rope)
    else:
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
            valid,
            0,
        ).to(tl.float32)
        a = tl.max(tl.abs(x), 1)
        scale = a / 448.0
        v = x / scale[:, None]
        quant = _encode_short(v)
        tl.store(
            Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :],
            quant,
            valid,
        )
        tl.store(
            S + rows * SS + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, RB)
        mask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(R + rr[:, None] * RS + rc[None, :] * RC, mask, 0)
        tl.store(output_ptr + rr[:, None] * OS + rc[None, :], rope, mask)


@tr.jit
def _loop_addr(X, R, Q, S, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    R = R.to(tl.pointer_type(tl.int16))
    Q = Q.to(tl.pointer_type(tl.uint8))
    S = S.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int16))
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    XS: tl.constexpr = M[4]
    XC: tl.constexpr = M[5]
    RS: tl.constexpr = M[6]
    RC: tl.constexpr = M[7]
    NG: tl.constexpr = D // G
    QS: tl.constexpr = M[8]
    SS: tl.constexpr = QS // 4
    OS: tl.constexpr = QS // 2
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G)
    RB: tl.constexpr = tr.next_power_of_2(DR)
    S = S + D // 4
    output_ptr = output_ptr + (D + NG * 4) // 2
    for block in range(tl.program_id(0), tr.cdiv(N, BR), tl.num_programs(0)):
        groups = block * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        valid = (
            (groups[:, None] < N * NG)
            & (c[None, :] < G)
            & (tl.arange(0, BG)[:, None] < BR * NG)
        )
        if N % BR == 0 and BG == BR * NG and (BC == G) and (RB == DR):
            x = tl.load(
                X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC
            ).to(tl.float32)
            a = tl.max(tl.abs(x), 1)
            scale = a / 448.0
            v = x / scale[:, None]
            quant = _encode_short(v)
            tl.store(
                Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :], quant
            )
            tl.store(S + rows * SS + tiles, scale)
            rr = block * BR + tl.arange(0, BR)
            rc = tl.arange(0, RB)
            mask = (rr[:, None] < N) & (rc[None, :] < DR)
            rope = tl.load(R + rr[:, None] * RS + rc[None, :] * RC)
            tl.store(output_ptr + rr[:, None] * OS + rc[None, :], rope)
        else:
            x = tl.load(
                X
                + rows[:, None] * XS
                + (tiles[:, None] * G + c[None, :]) * XC,
                valid,
                0,
            ).to(tl.float32)
            a = tl.max(tl.abs(x), 1)
            scale = a / 448.0
            v = x / scale[:, None]
            quant = _encode_short(v)
            tl.store(
                Q + rows[:, None] * QS + tiles[:, None] * G + c[None, :],
                quant,
                valid,
            )
            tl.store(
                S + rows * SS + tiles,
                scale,
                (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
            )
            rr = block * BR + tl.arange(0, BR)
            rc = tl.arange(0, RB)
            mask = (rr[:, None] < N) & (rc[None, :] < DR)
            rope = tl.load(R + rr[:, None] * RS + rc[None, :] * RC, mask, 0)
            tl.store(output_ptr + rr[:, None] * OS + rc[None, :], rope, mask)


@tr.jit
def _quant_contiguous_addr(
    X,
    Q,
    S,
    N: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    XS: tl.constexpr,
    XC: tl.constexpr,
    BR: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.bfloat16))
    Q = Q.to(tl.pointer_type(tl.uint8))
    S = S.to(tl.pointer_type(tl.float32))
    NG: tl.constexpr = D // G
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G)
    groups = tl.program_id(0) * (BR * NG) + tl.arange(0, BG)
    rows = groups // NG
    c = tl.arange(0, BC)
    valid = (
        (groups[:, None] < N * NG)
        & (tl.arange(0, BG)[:, None] < BR * NG)
        & (c[None, :] < G)
    )
    x = tl.load(
        X + rows[:, None] * XS + (groups[:, None] % NG * G + c[None, :]) * XC,
        valid,
        0,
    ).to(tl.float32)
    scale = tl.max(tl.abs(x), 1) / 448.0
    quant = _encode_short(x / scale[:, None])
    tl.store(Q + groups[:, None] * G + c[None, :], quant, valid)
    tl.store(
        S + groups, scale, (groups < N * NG) & (tl.arange(0, BG) < BR * NG)
    )


@tr.jit
def _assemble_typed_addr(
    Q,
    S,
    R,
    output_ptr,
    SO,
    RO,
    N: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    G: tl.constexpr,
    RS: tl.constexpr,
    RC: tl.constexpr,
    OS: tl.constexpr,
    BR: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    S = S.to(tl.pointer_type(tl.float32))
    R = R.to(tl.pointer_type(tl.bfloat16))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    SO = SO.to(tl.pointer_type(tl.float32))
    RO = RO.to(tl.pointer_type(tl.bfloat16))
    NG: tl.constexpr = D // G
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, tr.next_power_of_2(D))
    valid = (rows[:, None] < N) & (c[None, :] < D)
    value = tl.load(Q + rows[:, None] * D + c[None, :], valid, 0)
    tl.store(output_ptr + rows[:, None] * OS + c[None, :], value, valid)
    g = tl.arange(0, tr.next_power_of_2(NG))
    valid_s = (rows[:, None] < N) & (g[None, :] < NG)
    scale = tl.load(S + rows[:, None] * NG + g[None, :], valid_s, 0)
    tl.store(
        SO + rows[:, None] * (OS // 4) + D // 4 + g[None, :], scale, valid_s
    )
    r = tl.arange(0, tr.next_power_of_2(DR))
    valid_r = (rows[:, None] < N) & (r[None, :] < DR)
    rope = tl.load(R + rows[:, None] * RS + r[None, :] * RC, valid_r, 0)
    tl.store(
        RO + rows[:, None] * (OS // 2) + (D + NG * 4) // 2 + r[None, :],
        rope,
        valid_r,
    )


@tr.jit
def _assemble_words_addr(
    Q, S, R, output_ptr, M: tl.constexpr, BR: tl.constexpr
):
    Q = Q.to(tl.pointer_type(tl.int32))
    S = S.to(tl.pointer_type(tl.float32))
    R = R.to(tl.pointer_type(tl.int32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int32))
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    RS: tl.constexpr = M[4]
    OS: tl.constexpr = M[5]
    NG: tl.constexpr = D // G
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, tr.next_power_of_2(D // 4))
    valid = (rows[:, None] < N) & (c[None, :] < D // 4)
    value = tl.load(Q + rows[:, None] * (D // 4) + c[None, :], valid, 0)
    tl.store(output_ptr + rows[:, None] * (OS // 4) + c[None, :], value, valid)
    g = tl.arange(0, tr.next_power_of_2(NG))
    valid_s = (rows[:, None] < N) & (g[None, :] < NG)
    scale = tl.load(S + rows[:, None] * NG + g[None, :], valid_s, 0).to(
        tl.int32, bitcast=True
    )
    tl.store(
        output_ptr + rows[:, None] * (OS // 4) + D // 4 + g[None, :],
        scale,
        valid_s,
    )
    r = tl.arange(0, tr.next_power_of_2(DR // 2))
    valid_r = (rows[:, None] < N) & (r[None, :] < DR // 2)
    rope = tl.load(R + rows[:, None] * RS + r[None, :], valid_r, 0)
    tl.store(
        output_ptr + rows[:, None] * (OS // 4) + D // 4 + NG + r[None, :],
        rope,
        valid_r,
    )


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    dr = k_rope.shape[1]
    ng = d // group_size
    width = d + ng * 4 + dr * 2
    if n < 4096:
        physical = width
        if n < 1024:
            output = k_nope.new_empty((n, width), dtype=torch.float8_e4m3fn)
        else:
            output = torch.empty(
                (n, width), dtype=torch.uint8, device=k_nope.device
            )
    else:
        physical = tr.cdiv(width, 128) * 128
        output = torch.empty_strided(
            (n, width), (physical, 1), dtype=torch.uint8, device=k_nope.device
        )
    if n:
        if k_nope.stride(1) != 1:
            x = torch.empty((n, d), dtype=k_nope.dtype, device=k_nope.device)
            _copy[tr.cdiv(n * d, 256),](k_nope, x, n, d, *k_nope.stride(), 256)
            k_nope = x
        if k_rope.stride(1) != 1:
            r = torch.empty((n, dr), dtype=k_rope.dtype, device=k_rope.device)
            _copy[tr.cdiv(n * dr, 256),](
                k_rope, r, n, dr, *k_rope.stride(), 256
            )
            k_rope = r
        if n < 1024:
            _pack_bytes2.run(
                k_nope.data_ptr(),
                k_rope.data_ptr(),
                output.data_ptr(),
                (
                    n,
                    d,
                    dr,
                    group_size,
                    *k_nope.stride(),
                    *k_rope.stride(),
                    physical,
                ),
                4,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                grid=(tr.cdiv(n, 4),),
                warmup=False,
            )
        elif n >= 4096 and tr.cdiv(n, 8) <= 65535:
            q = k_nope.new_empty((n * (d + ng * 4),), dtype=torch.uint8)
            _quant_contiguous_addr.run(
                k_nope.data_ptr(),
                q.data_ptr(),
                q.data_ptr() + n * d,
                n,
                d,
                group_size,
                *k_nope.stride(),
                8,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                grid=(tr.cdiv(n, 8),),
                warmup=False,
            )
            if (
                d % 4 == 0
                and dr % 2 == 0
                and (width % 4 == 0)
                and (k_rope.stride(0) % 2 == 0)
                and (k_rope.data_ptr() % 4 == 0)
            ):
                _assemble_words_addr.run(
                    q.data_ptr(),
                    q.data_ptr() + n * d,
                    k_rope.data_ptr(),
                    output.data_ptr(),
                    (n, d, dr, group_size, k_rope.stride(0) // 2, physical),
                    16,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                    grid=(tr.cdiv(n, 16),),
                    warmup=False,
                )
            else:
                _assemble_typed_addr.run(
                    q.data_ptr(),
                    q.data_ptr() + n * d,
                    k_rope.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    n,
                    d,
                    dr,
                    group_size,
                    *k_rope.stride(),
                    physical,
                    8,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                    grid=(tr.cdiv(n, 8),),
                    warmup=False,
                )
        else:
            meta = (
                n,
                d,
                dr,
                group_size,
                *k_nope.stride(),
                *k_rope.stride(),
                physical,
            )
            if n >= 4096:
                _loop_addr.run(
                    k_nope.data_ptr(),
                    k_rope.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    meta,
                    8,
                    num_warps=4,
                    num_stages=2,
                    enable_fp_fusion=False,
                    grid=(min(tr.cdiv(n, 8), 40),),
                    warmup=False,
                )
            else:
                _pack_addr.run(
                    k_nope.data_ptr(),
                    k_rope.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    output.data_ptr(),
                    meta,
                    8,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                    grid=(tr.cdiv(n, 8),),
                    warmup=False,
                )
    if n < 1024:
        return output
    return output.view(torch.float8_e4m3fn)


__all__ = ["quantize_k_cache_fast"]
