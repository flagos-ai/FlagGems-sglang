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
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 2147483647
    normal = mag + 67633151 + (mag >> 20 & 1) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True)
    code = tl.where(mag < 1015021568, subnormal, tl.minimum(normal, 1150))
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _quant(x, s):
    factor = tl.where(s < 7.888609052210118e-31, 4294967296.0, 1.0)
    return _encode(x * factor / (s * factor))


@tr.jit
def _quant_fast(x, s, factor, inv, inv_lo):
    numerator = x * factor
    return _encode(tl.fma(numerator, inv, numerator * inv_lo))


@tr.jit
def _pack(X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    if M[0] >= 1024:
        P: tl.constexpr = 1
        N: tl.constexpr = M[0]
        D: tl.constexpr = M[1]
        DR: tl.constexpr = M[2]
        G: tl.constexpr = M[3]
        XS: tl.constexpr = M[4]
        XC: tl.constexpr = M[5]
        RS: tl.constexpr = M[6]
        RC: tl.constexpr = M[7]
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        BG: tl.constexpr = tr.next_power_of_2(BR * NG)
        BC: tl.constexpr = tr.next_power_of_2(G // P)
        groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        mask = (
            (groups[:, None] < N * NG)
            & (c[None, :] < G // P)
            & (tl.arange(0, BG)[:, None] < BR * NG)
        )
        if P == 4:
            offsets = (
                rows[:, None] * (XS // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :]
            )
            raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
            lo = raw.to(tl.int32)
            hi = (raw >> 32).to(tl.int32)
            a = (lo << 16).to(tl.float32, bitcast=True)
            b = (lo & -65536).to(tl.float32, bitcast=True)
            e = (hi << 16).to(tl.float32, bitcast=True)
            f = (hi & -65536).to(tl.float32, bitcast=True)
            mag = tl.maximum(
                a.to(tl.int32, bitcast=True) & 2147483647,
                b.to(tl.int32, bitcast=True) & 2147483647,
            )
            mag = tl.maximum(
                mag,
                tl.maximum(
                    e.to(tl.int32, bitcast=True) & 2147483647,
                    f.to(tl.int32, bitcast=True) & 2147483647,
                ),
            )
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
            denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
            root = tl.rsqrt(denominator)
            estimate = root * root
            inv = tl.where(scale == float("inf"), 0.0, estimate)
            inv_lo = inv * tl.fma(-inv, denominator, 1.0)
            word = (
                _quant_fast(
                    a,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                | _quant_fast(
                    b,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 8
            )
            word = (
                word
                | _quant_fast(
                    e,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 16
                | _quant_fast(
                    f,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 24
            )
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                mask,
            )
        else:
            x = tl.load(
                X
                + rows[:, None] * XS
                + (tiles[:, None] * G + c[None, :]) * XC,
                mask,
                0,
            ).to(tl.float32)
            mag = x.to(tl.int32, bitcast=True) & 2147483647
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
            denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
            root = tl.rsqrt(denominator)
            estimate = root * root
            inv = tl.where(scale == float("inf"), 0.0, estimate)
            inv_lo = inv * tl.fma(-inv, denominator, 1.0)
            tl.store(
                output_ptr
                + rows[:, None] * WIDTH
                + tiles[:, None] * G
                + c[None, :],
                _quant_fast(
                    x,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ),
                mask,
            )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, tr.next_power_of_2(DR))
        rmask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.int16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rmask,
            0,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.int16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            rmask,
        )
    elif M[0] < 512:
        output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
        N: tl.constexpr = M[0]
        D: tl.constexpr = M[1]
        DR: tl.constexpr = M[2]
        G: tl.constexpr = M[3]
        XS: tl.constexpr = M[4]
        XC: tl.constexpr = M[5]
        RS: tl.constexpr = M[6]
        RC: tl.constexpr = M[7]
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        cc = tl.arange(0, D)
        x = tl.load(
            X + rr[:, None] * XS + cc[None, :] * XC, rr[:, None] < N, 0
        ).to(tl.float32)
        tiles = tl.reshape(x, (BR, NG, G))
        bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
        scale = bits.to(tl.float32, bitcast=True) / 448.0
        quant = tl.reshape(_quant(tiles, scale[:, :, None]), (BR, D))
        rc = tl.arange(0, DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.uint16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rr[:, None] < N,
            0,
        )
        if False:
            oc = tl.arange(0, tr.next_power_of_2(WIDTH))
            oi = tl.broadcast_to(oc[None, :], (BR, tr.next_power_of_2(WIDTH)))
            q = tl.gather(quant, oi % D, 1)
            s = tl.gather(
                scale.to(tl.uint32, bitcast=True), (oi - D) // 4 % NG, 1
            )
            sb = (s >> oi % 4 * 8).to(tl.uint8)
            r = tl.gather(rope, (oi - END) // 2 % DR, 1)
            rb = (r >> oi % 2 * 8).to(tl.uint8)
            output = tl.where(oi < D, q, tl.where(oi < END, sb, rb))
            tl.store(
                output_ptr + rr[:, None] * WIDTH + oc[None, :],
                output,
                (rr[:, None] < N) & (oc[None, :] < WIDTH),
            )
        else:
            tl.store(
                output_ptr + rr[:, None] * WIDTH + cc[None, :],
                quant,
                rr[:, None] < N,
            )
            sg = tl.arange(0, NG)
            tl.store(
                (output_ptr + D).to(tl.pointer_type(tl.float32))
                + rr[:, None] * (WIDTH // 4)
                + sg[None, :],
                scale,
                rr[:, None] < N,
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.uint16))
                + rr[:, None] * (WIDTH // 2)
                + rc[None, :],
                rope,
                rr[:, None] < N,
            )
    else:
        P: tl.constexpr = 1
        N: tl.constexpr = M[0]
        D: tl.constexpr = M[1]
        DR: tl.constexpr = M[2]
        G: tl.constexpr = M[3]
        XS: tl.constexpr = M[4]
        XC: tl.constexpr = M[5]
        RS: tl.constexpr = M[6]
        RC: tl.constexpr = M[7]
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        BG: tl.constexpr = tr.next_power_of_2(BR * NG)
        BC: tl.constexpr = tr.next_power_of_2(G // P)
        groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        mask = (
            (groups[:, None] < N * NG)
            & (c[None, :] < G // P)
            & (tl.arange(0, BG)[:, None] < BR * NG)
        )
        if P == 4:
            offsets = (
                rows[:, None] * (XS // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :]
            )
            raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
            lo = raw.to(tl.int32)
            hi = (raw >> 32).to(tl.int32)
            a = (lo << 16).to(tl.float32, bitcast=True)
            b = (lo & -65536).to(tl.float32, bitcast=True)
            e = (hi << 16).to(tl.float32, bitcast=True)
            f = (hi & -65536).to(tl.float32, bitcast=True)
            mag = tl.maximum(
                a.to(tl.int32, bitcast=True) & 2147483647,
                b.to(tl.int32, bitcast=True) & 2147483647,
            )
            mag = tl.maximum(
                mag,
                tl.maximum(
                    e.to(tl.int32, bitcast=True) & 2147483647,
                    f.to(tl.int32, bitcast=True) & 2147483647,
                ),
            )
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            word = (
                _quant(a, scale[:, None]).to(tl.uint32)
                | _quant(b, scale[:, None]).to(tl.uint32) << 8
            )
            word = (
                word
                | _quant(e, scale[:, None]).to(tl.uint32) << 16
                | _quant(f, scale[:, None]).to(tl.uint32) << 24
            )
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                mask,
            )
        else:
            x = tl.load(
                X
                + rows[:, None] * XS
                + (tiles[:, None] * G + c[None, :]) * XC,
                mask,
                0,
            ).to(tl.float32)
            mag = x.to(tl.int32, bitcast=True) & 2147483647
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            tl.store(
                output_ptr
                + rows[:, None] * WIDTH
                + tiles[:, None] * G
                + c[None, :],
                _quant(x, scale[:, None]),
                mask,
            )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, tr.next_power_of_2(DR))
        rmask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.int16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rmask,
            0,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.int16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            rmask,
        )


def _baseline(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    dr = k_rope.shape[1]
    width = d + d // group_size * 4 + dr * 2
    output = torch.empty((n, width), dtype=torch.uint8, device=k_nope.device)
    if n:
        br = 1 if n <= 16 else 4
        meta = (n, d, dr, group_size, *k_nope.stride(), *k_rope.stride())
        _pack.run(
            k_nope,
            k_rope,
            output,
            meta,
            br,
            grid=(tr.cdiv(n, br),),
            warmup=False,
        )
    return output.view(torch.float8_e4m3fn)


@tr.jit
def _pack_contiguous(X, R, output_ptr, N: tl.constexpr):
    BR: tl.constexpr = 1 if N <= 16 else 4
    if N >= 1024:
        P: tl.constexpr = 1
        D: tl.constexpr = 512
        DR: tl.constexpr = 64
        G: tl.constexpr = 128
        XS: tl.constexpr = 512
        XC: tl.constexpr = 1
        RS: tl.constexpr = 64
        RC: tl.constexpr = 1
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        BG: tl.constexpr = tr.next_power_of_2(BR * NG)
        BC: tl.constexpr = tr.next_power_of_2(G // P)
        groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        mask = (
            (groups[:, None] < N * NG)
            & (c[None, :] < G // P)
            & (tl.arange(0, BG)[:, None] < BR * NG)
        )
        if P == 4:
            offsets = (
                rows[:, None] * (XS // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :]
            )
            raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
            lo = raw.to(tl.int32)
            hi = (raw >> 32).to(tl.int32)
            a = (lo << 16).to(tl.float32, bitcast=True)
            b = (lo & -65536).to(tl.float32, bitcast=True)
            e = (hi << 16).to(tl.float32, bitcast=True)
            f = (hi & -65536).to(tl.float32, bitcast=True)
            mag = tl.maximum(
                a.to(tl.int32, bitcast=True) & 2147483647,
                b.to(tl.int32, bitcast=True) & 2147483647,
            )
            mag = tl.maximum(
                mag,
                tl.maximum(
                    e.to(tl.int32, bitcast=True) & 2147483647,
                    f.to(tl.int32, bitcast=True) & 2147483647,
                ),
            )
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
            denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
            root = tl.rsqrt(denominator)
            estimate = root * root
            inv = tl.where(scale == float("inf"), 0.0, estimate)
            inv_lo = inv * tl.fma(-inv, denominator, 1.0)
            word = (
                _quant_fast(
                    a,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                | _quant_fast(
                    b,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 8
            )
            word = (
                word
                | _quant_fast(
                    e,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 16
                | _quant_fast(
                    f,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ).to(tl.uint32)
                << 24
            )
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                mask,
            )
        else:
            x = tl.load(
                X
                + rows[:, None] * XS
                + (tiles[:, None] * G + c[None, :]) * XC,
                mask,
                0,
            ).to(tl.float32)
            mag = x.to(tl.int32, bitcast=True) & 2147483647
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
            denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
            root = tl.rsqrt(denominator)
            estimate = root * root
            inv = tl.where(scale == float("inf"), 0.0, estimate)
            inv_lo = inv * tl.fma(-inv, denominator, 1.0)
            tl.store(
                output_ptr
                + rows[:, None] * WIDTH
                + tiles[:, None] * G
                + c[None, :],
                _quant_fast(
                    x,
                    denominator[:, None],
                    factor[:, None],
                    inv[:, None],
                    inv_lo[:, None],
                ),
                mask,
            )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, tr.next_power_of_2(DR))
        rmask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.int16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rmask,
            0,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.int16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            rmask,
        )
    elif N < 512:
        output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
        D: tl.constexpr = 512
        DR: tl.constexpr = 64
        G: tl.constexpr = 128
        XS: tl.constexpr = 512
        XC: tl.constexpr = 1
        RS: tl.constexpr = 64
        RC: tl.constexpr = 1
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        cc = tl.arange(0, D)
        x = tl.load(
            X + rr[:, None] * XS + cc[None, :] * XC, rr[:, None] < N, 0
        ).to(tl.float32)
        tiles = tl.reshape(x, (BR, NG, G))
        bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
        scale = bits.to(tl.float32, bitcast=True) / 448.0
        quant = tl.reshape(_quant(tiles, scale[:, :, None]), (BR, D))
        rc = tl.arange(0, DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.uint16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rr[:, None] < N,
            0,
        )
        if False:
            oc = tl.arange(0, tr.next_power_of_2(WIDTH))
            oi = tl.broadcast_to(oc[None, :], (BR, tr.next_power_of_2(WIDTH)))
            q = tl.gather(quant, oi % D, 1)
            s = tl.gather(
                scale.to(tl.uint32, bitcast=True), (oi - D) // 4 % NG, 1
            )
            sb = (s >> oi % 4 * 8).to(tl.uint8)
            r = tl.gather(rope, (oi - END) // 2 % DR, 1)
            rb = (r >> oi % 2 * 8).to(tl.uint8)
            output = tl.where(oi < D, q, tl.where(oi < END, sb, rb))
            tl.store(
                output_ptr + rr[:, None] * WIDTH + oc[None, :],
                output,
                (rr[:, None] < N) & (oc[None, :] < WIDTH),
            )
        else:
            tl.store(
                output_ptr + rr[:, None] * WIDTH + cc[None, :],
                quant,
                rr[:, None] < N,
            )
            sg = tl.arange(0, NG)
            tl.store(
                (output_ptr + D).to(tl.pointer_type(tl.float32))
                + rr[:, None] * (WIDTH // 4)
                + sg[None, :],
                scale,
                rr[:, None] < N,
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.uint16))
                + rr[:, None] * (WIDTH // 2)
                + rc[None, :],
                rope,
                rr[:, None] < N,
            )
    else:
        P: tl.constexpr = 1
        D: tl.constexpr = 512
        DR: tl.constexpr = 64
        G: tl.constexpr = 128
        XS: tl.constexpr = 512
        XC: tl.constexpr = 1
        RS: tl.constexpr = 64
        RC: tl.constexpr = 1
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        BG: tl.constexpr = tr.next_power_of_2(BR * NG)
        BC: tl.constexpr = tr.next_power_of_2(G // P)
        groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        mask = (
            (groups[:, None] < N * NG)
            & (c[None, :] < G // P)
            & (tl.arange(0, BG)[:, None] < BR * NG)
        )
        if P == 4:
            offsets = (
                rows[:, None] * (XS // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :]
            )
            raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
            lo = raw.to(tl.int32)
            hi = (raw >> 32).to(tl.int32)
            a = (lo << 16).to(tl.float32, bitcast=True)
            b = (lo & -65536).to(tl.float32, bitcast=True)
            e = (hi << 16).to(tl.float32, bitcast=True)
            f = (hi & -65536).to(tl.float32, bitcast=True)
            mag = tl.maximum(
                a.to(tl.int32, bitcast=True) & 2147483647,
                b.to(tl.int32, bitcast=True) & 2147483647,
            )
            mag = tl.maximum(
                mag,
                tl.maximum(
                    e.to(tl.int32, bitcast=True) & 2147483647,
                    f.to(tl.int32, bitcast=True) & 2147483647,
                ),
            )
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            word = (
                _quant(a, scale[:, None]).to(tl.uint32)
                | _quant(b, scale[:, None]).to(tl.uint32) << 8
            )
            word = (
                word
                | _quant(e, scale[:, None]).to(tl.uint32) << 16
                | _quant(f, scale[:, None]).to(tl.uint32) << 24
            )
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                mask,
            )
        else:
            x = tl.load(
                X
                + rows[:, None] * XS
                + (tiles[:, None] * G + c[None, :]) * XC,
                mask,
                0,
            ).to(tl.float32)
            mag = x.to(tl.int32, bitcast=True) & 2147483647
            scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
            tl.store(
                output_ptr
                + rows[:, None] * WIDTH
                + tiles[:, None] * G
                + c[None, :],
                _quant(x, scale[:, None]),
                mask,
            )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        rr = tl.program_id(0) * BR + tl.arange(0, BR)
        rc = tl.arange(0, tr.next_power_of_2(DR))
        rmask = (rr[:, None] < N) & (rc[None, :] < DR)
        rope = tl.load(
            R.to(tl.pointer_type(tl.int16))
            + rr[:, None] * RS
            + rc[None, :] * RC,
            rmask,
            0,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.int16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            rmask,
        )


@tr.jit
def _pack_contiguous_small1(X, R, output_ptr):
    BR: tl.constexpr = 1
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = 128
    XS: tl.constexpr = 512
    XC: tl.constexpr = 1
    RS: tl.constexpr = 64
    RC: tl.constexpr = 1
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.arange(0, D)
    x = tl.load(X + rr[:, None] * XS + cc[None, :] * XC, True, 0).to(
        tl.float32
    )
    tiles = tl.reshape(x, (BR, NG, G))
    bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
    scale = bits.to(tl.float32, bitcast=True) / 448.0
    quant = tl.reshape(_quant(tiles, scale[:, :, None]), (BR, D))
    rc = tl.arange(0, DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * RS + rc[None, :] * RC,
        True,
        0,
    )
    if False:
        oc = tl.arange(0, tr.next_power_of_2(WIDTH))
        oi = tl.broadcast_to(oc[None, :], (BR, tr.next_power_of_2(WIDTH)))
        q = tl.gather(quant, oi % D, 1)
        s = tl.gather(scale.to(tl.uint32, bitcast=True), (oi - D) // 4 % NG, 1)
        sb = (s >> oi % 4 * 8).to(tl.uint8)
        r = tl.gather(rope, (oi - END) // 2 % DR, 1)
        rb = (r >> oi % 2 * 8).to(tl.uint8)
        output = tl.where(oi < D, q, tl.where(oi < END, sb, rb))
        tl.store(
            output_ptr + rr[:, None] * WIDTH + oc[None, :],
            output,
            True & (oc[None, :] < WIDTH),
        )
    else:
        tl.store(output_ptr + rr[:, None] * WIDTH + cc[None, :], quant, True)
        sg = tl.arange(0, NG)
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rr[:, None] * (WIDTH // 4)
            + sg[None, :],
            scale,
            True,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.uint16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            True,
        )


@tr.jit
def _pack_contiguous_small4(X, R, output_ptr):
    BR: tl.constexpr = 4
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = 128
    XS: tl.constexpr = 512
    XC: tl.constexpr = 1
    RS: tl.constexpr = 64
    RC: tl.constexpr = 1
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.arange(0, D)
    x = tl.load(X + rr[:, None] * XS + cc[None, :] * XC, True, 0).to(
        tl.float32
    )
    tiles = tl.reshape(x, (BR, NG, G))
    bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
    scale = bits.to(tl.float32, bitcast=True) / 448.0
    quant = tl.reshape(_quant(tiles, scale[:, :, None]), (BR, D))
    rc = tl.arange(0, DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * RS + rc[None, :] * RC,
        True,
        0,
    )
    if False:
        oc = tl.arange(0, tr.next_power_of_2(WIDTH))
        oi = tl.broadcast_to(oc[None, :], (BR, tr.next_power_of_2(WIDTH)))
        q = tl.gather(quant, oi % D, 1)
        s = tl.gather(scale.to(tl.uint32, bitcast=True), (oi - D) // 4 % NG, 1)
        sb = (s >> oi % 4 * 8).to(tl.uint8)
        r = tl.gather(rope, (oi - END) // 2 % DR, 1)
        rb = (r >> oi % 2 * 8).to(tl.uint8)
        output = tl.where(oi < D, q, tl.where(oi < END, sb, rb))
        tl.store(
            output_ptr + rr[:, None] * WIDTH + oc[None, :],
            output,
            True & (oc[None, :] < WIDTH),
        )
    else:
        tl.store(output_ptr + rr[:, None] * WIDTH + cc[None, :], quant, True)
        sg = tl.arange(0, NG)
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rr[:, None] * (WIDTH // 4)
            + sg[None, :],
            scale,
            True,
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.uint16))
            + rr[:, None] * (WIDTH // 2)
            + rc[None, :],
            rope,
            True,
        )


@tr.jit
def _pack_contiguous_large4(X, R, output_ptr):
    BR: tl.constexpr = 4
    P: tl.constexpr = 1
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = 128
    XS: tl.constexpr = 512
    XC: tl.constexpr = 1
    RS: tl.constexpr = 64
    RC: tl.constexpr = 1
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G // P)
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    mask = True & (c[None, :] < G // P) & (tl.arange(0, BG)[:, None] < BR * NG)
    if P == 4:
        offsets = (
            rows[:, None] * (XS // 4) + tiles[:, None] * (G // 4) + c[None, :]
        )
        raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
        lo = raw.to(tl.int32)
        hi = (raw >> 32).to(tl.int32)
        a = (lo << 16).to(tl.float32, bitcast=True)
        b = (lo & -65536).to(tl.float32, bitcast=True)
        e = (hi << 16).to(tl.float32, bitcast=True)
        f = (hi & -65536).to(tl.float32, bitcast=True)
        mag = tl.maximum(
            a.to(tl.int32, bitcast=True) & 2147483647,
            b.to(tl.int32, bitcast=True) & 2147483647,
        )
        mag = tl.maximum(
            mag,
            tl.maximum(
                e.to(tl.int32, bitcast=True) & 2147483647,
                f.to(tl.int32, bitcast=True) & 2147483647,
            ),
        )
        scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
        factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
        denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
        root = tl.rsqrt(denominator)
        estimate = root * root
        inv = tl.where(scale == float("inf"), 0.0, estimate)
        inv_lo = inv * tl.fma(-inv, denominator, 1.0)
        word = (
            _quant_fast(
                a,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ).to(tl.uint32)
            | _quant_fast(
                b,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ).to(tl.uint32)
            << 8
        )
        word = (
            word
            | _quant_fast(
                e,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ).to(tl.uint32)
            << 16
            | _quant_fast(
                f,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ).to(tl.uint32)
            << 24
        )
        tl.store(
            output_ptr.to(tl.pointer_type(tl.uint32))
            + rows[:, None] * (WIDTH // 4)
            + tiles[:, None] * (G // 4)
            + c[None, :],
            word,
            mask,
        )
    else:
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
            mask,
            0,
        ).to(tl.float32)
        mag = x.to(tl.int32, bitcast=True) & 2147483647
        scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
        factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
        denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
        root = tl.rsqrt(denominator)
        estimate = root * root
        inv = tl.where(scale == float("inf"), 0.0, estimate)
        inv_lo = inv * tl.fma(-inv, denominator, 1.0)
        tl.store(
            output_ptr
            + rows[:, None] * WIDTH
            + tiles[:, None] * G
            + c[None, :],
            _quant_fast(
                x,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ),
            mask,
        )
    tl.store(
        (output_ptr + D).to(tl.pointer_type(tl.float32))
        + rows * (WIDTH // 4)
        + tiles,
        scale,
        True & (tl.arange(0, BG) < BR * NG),
    )
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    rc = tl.arange(0, tr.next_power_of_2(DR))
    rmask = True & (rc[None, :] < DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.int16)) + rr[:, None] * RS + rc[None, :] * RC,
        rmask,
        0,
    )
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.int16))
        + rr[:, None] * (WIDTH // 2)
        + rc[None, :],
        rope,
        rmask,
    )


@tr.jit
def _split376(X, R, output_ptr):
    quant_blocks = tl.num_programs(0) * 4 // 5
    if tl.program_id(0) < quant_blocks:
        BR: tl.constexpr = 4
        D: tl.constexpr = 512
        DR: tl.constexpr = 64
        G: tl.constexpr = 128
        NG: tl.constexpr = D // G
        END: tl.constexpr = D + NG * 4
        WIDTH: tl.constexpr = END + DR * 2
        BG: tl.constexpr = tr.next_power_of_2(BR * NG)
        BC: tl.constexpr = 32
        groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
        rows = groups // NG
        tiles = groups % NG
        c = tl.arange(0, BC)
        offsets = rows[:, None] * 512 + tiles[:, None] * 128 + c[None, :]
        x0 = tl.load(X + offsets).to(tl.float32)
        x1 = tl.load(X + offsets + 32).to(tl.float32)
        x2 = tl.load(X + offsets + 64).to(tl.float32)
        x3 = tl.load(X + offsets + 96).to(tl.float32)
        mag = tl.maximum(
            tl.maximum(
                x0.to(tl.uint32, bitcast=True) & 2147483647,
                x1.to(tl.uint32, bitcast=True) & 2147483647,
            ),
            tl.maximum(
                x2.to(tl.uint32, bitcast=True) & 2147483647,
                x3.to(tl.uint32, bitcast=True) & 2147483647,
            ),
        )
        scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
        factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
        denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
        root = tl.rsqrt(denominator)
        estimate = root * root
        inv = tl.where(scale == float("inf"), 0.0, estimate)
        inv_lo = inv * tl.fma(-inv, denominator, 1.0)
        tl.store(
            output_ptr
            + rows[:, None] * 656
            + tiles[:, None] * 128
            + c[None, :]
            + 0,
            _quant_fast(
                x0,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ),
        )
        tl.store(
            output_ptr
            + rows[:, None] * 656
            + tiles[:, None] * 128
            + c[None, :]
            + 32,
            _quant_fast(
                x1,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ),
        )
        tl.store(
            output_ptr
            + rows[:, None] * 656
            + tiles[:, None] * 128
            + c[None, :]
            + 64,
            _quant_fast(
                x2,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ),
        )
        tl.store(
            output_ptr
            + rows[:, None] * 656
            + tiles[:, None] * 128
            + c[None, :]
            + 96,
            _quant_fast(
                x3,
                denominator[:, None],
                factor[:, None],
                inv[:, None],
                inv_lo[:, None],
            ),
        )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            True & (tl.arange(0, BG) < BR * NG),
        )
    else:
        rr = (tl.program_id(0) - quant_blocks) * 16 + tl.arange(0, 16)
        rc = tl.arange(0, 64)
        rope = tl.load(
            R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * 64 + rc[None, :]
        )
        tl.store(
            (output_ptr + 528).to(tl.pointer_type(tl.uint16))
            + rr[:, None] * 328
            + rc[None, :],
            rope,
        )


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    if (
        group_size != 128
        or not k_nope.is_contiguous()
        or (not k_rope.is_contiguous())
    ):
        return _baseline(k_nope, k_rope, group_size)
    n = k_nope.shape[0]
    output = torch.empty((n, 656), dtype=torch.uint8, device=k_nope.device)
    if n:
        if n % 4 != 0 and n > 16 or 512 <= n < 1024:
            _pack_contiguous.run(
                k_nope,
                k_rope,
                output,
                n,
                grid=(tr.cdiv(n, 1 if n <= 16 else 4),),
                warmup=False,
            )
        elif n >= 1024:
            if n % 16 == 0:
                _split376.run(
                    k_nope,
                    k_rope,
                    output,
                    grid=(n // 4 + n // 16,),
                    warmup=False,
                )
            else:
                _pack_contiguous_large4.run(
                    k_nope, k_rope, output, grid=(n // 4,), warmup=False
                )
        elif n <= 16:
            _pack_contiguous_small1.run(
                k_nope, k_rope, output, grid=(n,), warmup=False
            )
        else:
            _pack_contiguous_small4.run(
                k_nope, k_rope, output, grid=(n // 4,), warmup=False
            )
    return output.view(torch.float8_e4m3fn)


__all__ = ["quantize_k_cache_fast"]
