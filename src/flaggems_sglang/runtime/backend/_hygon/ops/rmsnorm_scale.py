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
def _offset(row, SH: tl.constexpr, ST: tl.constexpr):
    base = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SH) - 1, -1, -1):
        base += (row % SH[j]).to(tl.int64) * ST[j]
        row //= SH[j]
    return base


@triton.jit
def _row(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    SSH: tl.constexpr,
    SST: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + _offset(row, XSH, XST) + c * XST[-1], c < D, other=0).to(
        tl.float32
    )
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    s = tl.load(
        S + _offset(row // REP, SSH, SST) + c * SST[-1], c < D, other=0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c, z * s, c < D)


@triton.jit
def _lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D)


def _flat_rows(t):
    if t.ndim <= 2:
        return True
    row_stride = None
    rows = 1
    for size, stride in zip(reversed(t.shape[:-1]), reversed(t.stride()[:-1])):
        if size == 1:
            continue
        if row_stride is None:
            row_stride = stride
        elif stride != row_stride * rows:
            return False
        rows *= size
    return True


def rmsnorm_scale(x, weight, scale, eps):
    xshape = x.shape
    sshape = scale.shape
    if not xshape or not sshape:
        return None
    if not x.dtype == weight.dtype == scale.dtype == torch.bfloat16:
        return None
    if not x.device == weight.device == scale.device:
        return None
    d = xshape[-1]
    if (
        not 0 < d <= 8192
        or weight.shape != (d,)
        or (not weight.is_contiguous())
        or (sshape[-1] != d)
    ):
        return None
    m, sm = (xshape.numel() // d, sshape.numel() // d)
    if sm == 0 or m % sm != 0:
        return None
    xc, sc = (x.is_contiguous(), scale.is_contiguous())
    if not xc and (not _flat_rows(x)) or (not sc and (not _flat_rows(scale))):
        return None
    y = (
        torch.empty_like(x)
        if xc
        else torch.empty_like(x, memory_format=torch.contiguous_format)
    )
    if m:
        if xc and sc:
            xp, wp, sp, yp = (
                x.data_ptr(),
                weight.data_ptr(),
                scale.data_ptr(),
                y.data_ptr(),
            )
            input_alignment = xp | wp | sp
            aligned = (input_alignment | yp) & 15 == 0
            if not aligned:
                xp, wp, sp, yp = (x, weight, scale, y)
        else:
            aligned = False
            xp, wp, sp, yp = (x, weight, scale, y)
        b = 1 << max(6, (d - 1).bit_length())
        if (
            xc
            and sc
            and (d == 4096)
            and (m < 8192)
            and (input_alignment & 3 == 0)
        ):
            fn = (
                (_address_hint_pk2 if aligned else _pk2)
                if m >= 1024
                else _address_hint_pk1 if aligned else _pk1
            )
            fn.run(
                xp,
                wp,
                sp,
                yp,
                d,
                m // sm,
                float(eps),
                b,
                "",
                "",
                grid=(m,),
                warmup=False,
                num_warps=2 if m >= 1024 else 8,
                enable_fp_fusion=False,
            )
        elif xc and sc:
            (
                (
                    _address_hint_lean_delayed_scale
                    if aligned
                    else _lean_delayed_scale
                )
                if m >= 8192
                else _address_hint_lean if aligned else _lean
            ).run(
                xp,
                wp,
                sp,
                yp,
                (d, m // sm, float(eps), b),
                grid=(m,),
                warmup=False,
                num_warps=8 if d >= 4096 and m >= 8192 else 4,
                enable_fp_fusion=False,
            )
        else:
            _row[m,](
                x,
                weight,
                scale,
                y,
                d,
                m // sm,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                tuple(scale.shape[:-1]),
                tuple(scale.stride()),
                float(eps),
                b,
                num_warps=4 if b <= 2048 else 8,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return y


@triton.jit
def _lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D)


@triton.jit
def _asm_round(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = tl.inline_asm_elementwise(
        "v_bfe_u32 $0, $1, 16, 1\nv_add3_u32 $0, $1, $0, $2\nv_and_b32 $0, 0xffff0000, $0",
        constraints="=&v,v,s",
        args=[bits, tl.full((), 32767, tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    y = rounded.to(tl.float32, bitcast=True)
    return tl.where(x != x, float("nan"), y)


@triton.jit
def _pk1(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    CACHE: tl.constexpr,
    STORE: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B // 2)
    xp = X.to(tl.pointer_type(tl.uint32))
    wp = W.to(tl.pointer_type(tl.uint32))
    sp = S.to(tl.pointer_type(tl.uint32))
    yp = Y.to(tl.pointer_type(tl.uint32))
    c0 = c + 0 * (B // 2)
    a0 = tl.load(
        xp + row * (D // 2) + c0, c0 < D // 2, other=0, cache_modifier=CACHE
    )
    lo0 = (a0 << 16).to(tl.float32, bitcast=True)
    hi0 = (a0 & 4294901760).to(tl.float32, bitcast=True)
    inv = tl.rsqrt(tl.sum(lo0 * lo0 + hi0 * hi0, 0) / D + EPS)
    w0 = tl.load(wp + c0, c0 < D // 2, other=0)
    s0 = tl.load(
        sp + row // REP * (D // 2) + c0,
        c0 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    wl0 = (w0 << 16).to(tl.float32, bitcast=True)
    wh0 = (w0 & 4294901760).to(tl.float32, bitcast=True)
    sl0 = (s0 << 16).to(tl.float32, bitcast=True)
    sh0 = (s0 & 4294901760).to(tl.float32, bitcast=True)
    if 1 != 0:
        nl0 = _asm_round(lo0 * inv)
    else:
        nl0 = (lo0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh0 = _asm_round(hi0 * inv)
    else:
        nh0 = (hi0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl0 = _asm_round(nl0 * wl0)
    else:
        zl0 = (nl0 * wl0).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh0 = _asm_round(nh0 * wh0)
    else:
        zh0 = (nh0 * wh0).to(tl.bfloat16).to(tl.float32)
    if 1 == 2:
        yl0 = _asm_round(zl0 * sl0).to(tl.uint32, bitcast=True) >> 16
    else:
        yl0 = (
            (zl0 * sl0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh0 = _asm_round(zh0 * sh0).to(tl.uint32, bitcast=True) >> 16
    else:
        yh0 = (
            (zh0 * sh0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c0,
        yl0 | yh0 << 16,
        c0 < D // 2,
        cache_modifier=STORE,
    )


@triton.jit
def _pk2(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    CACHE: tl.constexpr,
    STORE: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B // 4)
    xp = X.to(tl.pointer_type(tl.uint32))
    wp = W.to(tl.pointer_type(tl.uint32))
    sp = S.to(tl.pointer_type(tl.uint32))
    yp = Y.to(tl.pointer_type(tl.uint32))
    c0 = c + 0 * (B // 4)
    a0 = tl.load(
        xp + row * (D // 2) + c0, c0 < D // 2, other=0, cache_modifier=CACHE
    )
    lo0 = (a0 << 16).to(tl.float32, bitcast=True)
    hi0 = (a0 & 4294901760).to(tl.float32, bitcast=True)
    c1 = c + 1 * (B // 4)
    a1 = tl.load(
        xp + row * (D // 2) + c1, c1 < D // 2, other=0, cache_modifier=CACHE
    )
    lo1 = (a1 << 16).to(tl.float32, bitcast=True)
    hi1 = (a1 & 4294901760).to(tl.float32, bitcast=True)
    w0 = tl.load(wp + c0, c0 < D // 2, other=0)
    wl0 = (w0 << 16).to(tl.float32, bitcast=True)
    wh0 = (w0 & 4294901760).to(tl.float32, bitcast=True)
    w1 = tl.load(wp + c1, c1 < D // 2, other=0)
    wl1 = (w1 << 16).to(tl.float32, bitcast=True)
    wh1 = (w1 & 4294901760).to(tl.float32, bitcast=True)
    inv = tl.rsqrt(
        tl.sum(lo0 * lo0 + hi0 * hi0 + lo1 * lo1 + hi1 * hi1, 0) / D + EPS
    )
    if 1 != 0:
        nl0 = _asm_round(lo0 * inv)
    else:
        nl0 = (lo0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh0 = _asm_round(hi0 * inv)
    else:
        nh0 = (hi0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl0 = _asm_round(nl0 * wl0)
    else:
        zl0 = (nl0 * wl0).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh0 = _asm_round(nh0 * wh0)
    else:
        zh0 = (nh0 * wh0).to(tl.bfloat16).to(tl.float32)
    s0 = tl.load(
        sp + row // REP * (D // 2) + c0,
        c0 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    sl0 = (s0 << 16).to(tl.float32, bitcast=True)
    sh0 = (s0 & 4294901760).to(tl.float32, bitcast=True)
    if 1 == 2:
        yl0 = _asm_round(zl0 * sl0).to(tl.uint32, bitcast=True) >> 16
    else:
        yl0 = (
            (zl0 * sl0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh0 = _asm_round(zh0 * sh0).to(tl.uint32, bitcast=True) >> 16
    else:
        yh0 = (
            (zh0 * sh0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c0,
        yl0 | yh0 << 16,
        c0 < D // 2,
        cache_modifier=STORE,
    )
    if 1 != 0:
        nl1 = _asm_round(lo1 * inv)
    else:
        nl1 = (lo1 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh1 = _asm_round(hi1 * inv)
    else:
        nh1 = (hi1 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl1 = _asm_round(nl1 * wl1)
    else:
        zl1 = (nl1 * wl1).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh1 = _asm_round(nh1 * wh1)
    else:
        zh1 = (nh1 * wh1).to(tl.bfloat16).to(tl.float32)
    s1 = tl.load(
        sp + row // REP * (D // 2) + c1,
        c1 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    sl1 = (s1 << 16).to(tl.float32, bitcast=True)
    sh1 = (s1 & 4294901760).to(tl.float32, bitcast=True)
    if 1 == 2:
        yl1 = _asm_round(zl1 * sl1).to(tl.uint32, bitcast=True) >> 16
    else:
        yl1 = (
            (zl1 * sl1)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh1 = _asm_round(zh1 * sh1).to(tl.uint32, bitcast=True) >> 16
    else:
        yh1 = (
            (zh1 * sh1)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c1,
        yl1 | yh1 << 16,
        c1 < D // 2,
        cache_modifier=STORE,
    )


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_pk1(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    CACHE: tl.constexpr,
    STORE: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    row = tl.program_id(0)
    c = tl.arange(0, B // 2)
    xp = X.to(tl.pointer_type(tl.uint32))
    wp = W.to(tl.pointer_type(tl.uint32))
    sp = S.to(tl.pointer_type(tl.uint32))
    yp = Y.to(tl.pointer_type(tl.uint32))
    c0 = c + 0 * (B // 2)
    a0 = tl.load(
        xp + row * (D // 2) + c0, c0 < D // 2, other=0, cache_modifier=CACHE
    )
    lo0 = (a0 << 16).to(tl.float32, bitcast=True)
    hi0 = (a0 & 4294901760).to(tl.float32, bitcast=True)
    inv = tl.rsqrt(tl.sum(lo0 * lo0 + hi0 * hi0, 0) / D + EPS)
    w0 = tl.load(wp + c0, c0 < D // 2, other=0)
    s0 = tl.load(
        sp + row // REP * (D // 2) + c0,
        c0 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    wl0 = (w0 << 16).to(tl.float32, bitcast=True)
    wh0 = (w0 & 4294901760).to(tl.float32, bitcast=True)
    sl0 = (s0 << 16).to(tl.float32, bitcast=True)
    sh0 = (s0 & 4294901760).to(tl.float32, bitcast=True)
    if 1 != 0:
        nl0 = _asm_round(lo0 * inv)
    else:
        nl0 = (lo0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh0 = _asm_round(hi0 * inv)
    else:
        nh0 = (hi0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl0 = _asm_round(nl0 * wl0)
    else:
        zl0 = (nl0 * wl0).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh0 = _asm_round(nh0 * wh0)
    else:
        zh0 = (nh0 * wh0).to(tl.bfloat16).to(tl.float32)
    if 1 == 2:
        yl0 = _asm_round(zl0 * sl0).to(tl.uint32, bitcast=True) >> 16
    else:
        yl0 = (
            (zl0 * sl0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh0 = _asm_round(zh0 * sh0).to(tl.uint32, bitcast=True) >> 16
    else:
        yh0 = (
            (zh0 * sh0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c0,
        yl0 | yh0 << 16,
        c0 < D // 2,
        cache_modifier=STORE,
    )


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_pk2(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    CACHE: tl.constexpr,
    STORE: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    row = tl.program_id(0)
    c = tl.arange(0, B // 4)
    xp = X.to(tl.pointer_type(tl.uint32))
    wp = W.to(tl.pointer_type(tl.uint32))
    sp = S.to(tl.pointer_type(tl.uint32))
    yp = Y.to(tl.pointer_type(tl.uint32))
    c0 = c + 0 * (B // 4)
    a0 = tl.load(
        xp + row * (D // 2) + c0, c0 < D // 2, other=0, cache_modifier=CACHE
    )
    lo0 = (a0 << 16).to(tl.float32, bitcast=True)
    hi0 = (a0 & 4294901760).to(tl.float32, bitcast=True)
    c1 = c + 1 * (B // 4)
    a1 = tl.load(
        xp + row * (D // 2) + c1, c1 < D // 2, other=0, cache_modifier=CACHE
    )
    lo1 = (a1 << 16).to(tl.float32, bitcast=True)
    hi1 = (a1 & 4294901760).to(tl.float32, bitcast=True)
    w0 = tl.load(wp + c0, c0 < D // 2, other=0)
    wl0 = (w0 << 16).to(tl.float32, bitcast=True)
    wh0 = (w0 & 4294901760).to(tl.float32, bitcast=True)
    w1 = tl.load(wp + c1, c1 < D // 2, other=0)
    wl1 = (w1 << 16).to(tl.float32, bitcast=True)
    wh1 = (w1 & 4294901760).to(tl.float32, bitcast=True)
    inv = tl.rsqrt(
        tl.sum(lo0 * lo0 + hi0 * hi0 + lo1 * lo1 + hi1 * hi1, 0) / D + EPS
    )
    if 1 != 0:
        nl0 = _asm_round(lo0 * inv)
    else:
        nl0 = (lo0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh0 = _asm_round(hi0 * inv)
    else:
        nh0 = (hi0 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl0 = _asm_round(nl0 * wl0)
    else:
        zl0 = (nl0 * wl0).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh0 = _asm_round(nh0 * wh0)
    else:
        zh0 = (nh0 * wh0).to(tl.bfloat16).to(tl.float32)
    s0 = tl.load(
        sp + row // REP * (D // 2) + c0,
        c0 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    sl0 = (s0 << 16).to(tl.float32, bitcast=True)
    sh0 = (s0 & 4294901760).to(tl.float32, bitcast=True)
    if 1 == 2:
        yl0 = _asm_round(zl0 * sl0).to(tl.uint32, bitcast=True) >> 16
    else:
        yl0 = (
            (zl0 * sl0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh0 = _asm_round(zh0 * sh0).to(tl.uint32, bitcast=True) >> 16
    else:
        yh0 = (
            (zh0 * sh0)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c0,
        yl0 | yh0 << 16,
        c0 < D // 2,
        cache_modifier=STORE,
    )
    if 1 != 0:
        nl1 = _asm_round(lo1 * inv)
    else:
        nl1 = (lo1 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        nh1 = _asm_round(hi1 * inv)
    else:
        nh1 = (hi1 * inv).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zl1 = _asm_round(nl1 * wl1)
    else:
        zl1 = (nl1 * wl1).to(tl.bfloat16).to(tl.float32)
    if 1 != 0:
        zh1 = _asm_round(nh1 * wh1)
    else:
        zh1 = (nh1 * wh1).to(tl.bfloat16).to(tl.float32)
    s1 = tl.load(
        sp + row // REP * (D // 2) + c1,
        c1 < D // 2,
        other=0,
        cache_modifier=CACHE,
    )
    sl1 = (s1 << 16).to(tl.float32, bitcast=True)
    sh1 = (s1 & 4294901760).to(tl.float32, bitcast=True)
    if 1 == 2:
        yl1 = _asm_round(zl1 * sl1).to(tl.uint32, bitcast=True) >> 16
    else:
        yl1 = (
            (zl1 * sl1)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    if 1 == 2:
        yh1 = _asm_round(zh1 * sh1).to(tl.uint32, bitcast=True) >> 16
    else:
        yh1 = (
            (zh1 * sh1)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
    tl.store(
        yp + row * (D // 2) + c1,
        yl1 | yh1 << 16,
        c1 < D // 2,
        cache_modifier=STORE,
    )


__all__ = ["rmsnorm_scale"]
