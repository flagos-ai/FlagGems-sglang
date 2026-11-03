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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _offset(row, SH: tl.constexpr, ST: tl.constexpr):
    base = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SH) - 1, -1, -1):
        base += (row % SH[j]).to(tl.int64) * ST[j]
        row //= SH[j]
    return base


@triton.jit
def _mul_round(a, b):
    return tl.fma(a, b, -0.0).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _row(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    XSH: tl.constexpr = P[2]
    XST: tl.constexpr = P[3]
    SSH: tl.constexpr = P[4]
    SST: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    B: tl.constexpr = P[7]
    row = tl.program_id(0)
    c = tl.arange(0, B)
    k = c % D
    x = tl.load(X + _offset(row, XSH, XST) + k * XST[-1]).to(tl.float32)
    x = tl.where(c < D, x, 0.0)
    w = tl.load(W + k).to(tl.float32)
    s = tl.load(S + _offset(row // REP, SSH, SST) + k * SST[-1]).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = _mul_round(x, inv)
    z = _mul_round(n, w)
    tl.store(Y + row * D + c, _mul_round(z, s), c < D)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _tile(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    B: tl.constexpr = P[4]
    R: tl.constexpr = P[5]
    CONT: tl.constexpr = P[6]
    c = tl.arange(0, B)
    r = tl.arange(0, R)
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(M, R), tl.num_programs(0), num_stages=1
    ):
        row = block * R + r
        if CONT:
            x = tl.load(
                X + row[:, None] * D + c[None, :],
                offset_state_policy="continuous",
            ).to(tl.float32)
            if REP == 1:
                s = tl.load(
                    S + row[:, None] * D + c[None, :],
                    offset_state_policy="continuous",
                ).to(tl.float32)
            else:
                s = tl.load(S + row[:, None] // REP * D + c[None, :]).to(
                    tl.float32
                )
        else:
            x = tl.load(
                X + row[:, None] * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
            s = tl.load(
                S + row[:, None] // REP * D + c[None, :],
                (row[:, None] < M) & (c[None, :] < D),
                other=0,
            ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        n = _mul_round(x, inv[:, None])
        z = _mul_round(n, w[None, :])
        if CONT:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                _mul_round(z, s),
                offset_state_policy="continuous",
            )
        else:
            tl.store(
                Y + row[:, None] * D + c[None, :],
                _mul_round(z, s),
                (row[:, None] < M) & (c[None, :] < D),
            )


@triton.jit
def _lean(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    CONT: tl.constexpr = P[1]
    D: tl.constexpr = P[2]
    REP: tl.constexpr = P[3]
    EPS: tl.constexpr = P[4]
    B: tl.constexpr = P[5]
    col = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        if CONT:
            x = tl.load(
                X + row * D + col, offset_state_policy="continuous"
            ).to(tl.float32)
            w = tl.load(W + col, offset_state_policy="continuous").to(
                tl.float32
            )
            s = tl.load(
                S + row // REP * D + col, offset_state_policy="continuous"
            ).to(tl.float32)
        else:
            x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
            w = tl.load(W + col, col < D, other=0).to(tl.float32)
            s = tl.load(S + row // REP * D + col, col < D, other=0).to(
                tl.float32
            )
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = _mul_round(x, inv)
        z = _mul_round(n, w)
        if CONT:
            tl.store(
                Y + row * D + col,
                _mul_round(z, s),
                offset_state_policy="continuous",
            )
        else:
            tl.store(Y + row * D + col, _mul_round(z, s), col < D)


@triton.jit
def _materialized(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    REP: tl.constexpr = P[2]
    EPS: tl.constexpr = P[3]
    c = tl.arange(0, D)
    w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        p = Y + row * D + c
        x = tl.load(X + row * D + c, offset_state_policy="continuous").to(
            tl.float32
        )
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        tl.store(
            p,
            _mul_round(x, inv),
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        )
        tl.debug_barrier()
        n = tl.load(
            p,
            volatile=True,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        ).to(tl.float32)
        tl.store(
            p,
            _mul_round(n, w),
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        )
        tl.debug_barrier()
        z = tl.load(
            p,
            volatile=True,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        ).to(tl.float32)
        s = tl.load(
            S + row // REP * D + c, offset_state_policy="continuous"
        ).to(tl.float32)
        tl.store(p, _mul_round(z, s), offset_state_policy="continuous")


@triton.jit
def _small_sync(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    BARRIER: tl.constexpr,
):
    c = tl.arange(0, B)
    w = tl.load(W + c, c < D, other=0, offset_state_policy="continuous").to(
        tl.float32
    )
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        p = Y + row * D + c
        x = tl.load(
            X + row * D + c, c < D, other=0, offset_state_policy="continuous"
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        tl.store(
            p,
            _mul_round(x, inv),
            c < D,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        )
        if BARRIER:
            tl.debug_barrier()
        n = tl.load(
            p,
            c < D,
            other=0,
            volatile=True,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        ).to(tl.float32)
        tl.store(
            p,
            _mul_round(n, w),
            c < D,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        )
        if BARRIER:
            tl.debug_barrier()
        z = tl.load(
            p,
            c < D,
            other=0,
            volatile=True,
            offset_state_policy="continuous",
            mem_sync_mode="sync",
        ).to(tl.float32)
        s = tl.load(
            S + row // REP * D + c,
            c < D,
            other=0,
            offset_state_policy="continuous",
        ).to(tl.float32)
        tl.store(p, _mul_round(z, s), c < D, offset_state_policy="continuous")


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


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _native_lean(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    CONT: tl.constexpr = P[1]
    D: tl.constexpr = P[2]
    REP: tl.constexpr = P[3]
    EPS: tl.constexpr = P[4]
    B: tl.constexpr = P[5]
    col = tl.arange(0, B)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0), num_stages=1):
        if CONT:
            x = tl.load(
                X + row * D + col, offset_state_policy="continuous"
            ).to(tl.float32)
            w = tl.load(W + col, offset_state_policy="continuous").to(
                tl.float32
            )
            s = tl.load(
                S + row // REP * D + col, offset_state_policy="continuous"
            ).to(tl.float32)
        else:
            x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
            w = tl.load(W + col, col < D, other=0).to(tl.float32)
            s = tl.load(S + row // REP * D + col, col < D, other=0).to(
                tl.float32
            )
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        n = _mul_round(x, inv)
        z = _mul_round(n, w)
        if CONT:
            tl.store(
                Y + row * D + col,
                _mul_round(z, s),
                offset_state_policy="continuous",
            )
        else:
            tl.store(Y + row * D + col, _mul_round(z, s), col < D)


@triton.jit
def _barrier_round(a, b):
    v = a * b
    v = tl.inline_asm_elementwise(
        "",
        constraints="=r,0",
        args=[v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _split_stats(
    X,
    T,
    M: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    NP: tl.constexpr,
):
    c = tl.arange(0, D // 2)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0) * (M // R) // NP,
        (tl.program_id(0) + 1) * (M // R) // NP,
        num_stages=1,
    ):
        row = block * R + r
        offsets = row[:, None] * D + c[None, :]
        x0 = tl.load(X + offsets, offset_state_policy="continuous").to(
            tl.float32
        )
        x1 = tl.load(
            X + offsets + D // 2, offset_state_policy="continuous"
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, 1) / D + EPS)
        tl.store(
            T + offsets, x0 * inv[:, None], offset_state_policy="continuous"
        )
        tl.store(
            T + offsets + D // 2,
            x1 * inv[:, None],
            offset_state_policy="continuous",
        )


@triton.jit
def _stats_parts4(
    X,
    T,
    M: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    NP: tl.constexpr,
):
    c = tl.arange(0, D // 4)
    r = tl.arange(0, R)
    for block in tl.range(
        tl.program_id(0) * (M // R) // NP,
        (tl.program_id(0) + 1) * (M // R) // NP,
        num_stages=1,
    ):
        row = block * R + r
        offsets = row[:, None] * D + c[None, :]
        x0 = tl.load(
            X + offsets + 0 * (D // 4), offset_state_policy="continuous"
        ).to(tl.float32)
        x1 = tl.load(
            X + offsets + 1 * (D // 4), offset_state_policy="continuous"
        ).to(tl.float32)
        x2 = tl.load(
            X + offsets + 2 * (D // 4), offset_state_policy="continuous"
        ).to(tl.float32)
        x3 = tl.load(
            X + offsets + 3 * (D // 4), offset_state_policy="continuous"
        ).to(tl.float32)
        inv = tl.rsqrt(
            tl.sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 1) / D + EPS
        )
        tl.store(
            T + offsets + 0 * (D // 4),
            x0 * inv[:, None],
            offset_state_policy="continuous",
        )
        tl.store(
            T + offsets + 1 * (D // 4),
            x1 * inv[:, None],
            offset_state_policy="continuous",
        )
        tl.store(
            T + offsets + 2 * (D // 4),
            x2 * inv[:, None],
            offset_state_policy="continuous",
        )
        tl.store(
            T + offsets + 3 * (D // 4),
            x3 * inv[:, None],
            offset_state_policy="continuous",
        )


@triton.jit
def _affine_rows(
    T,
    W,
    S,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    NP: tl.constexpr,
):
    c = tl.arange(0, D)
    r = tl.arange(0, R)
    w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
    for block in tl.range(
        tl.program_id(0) * (M // R) // NP,
        (tl.program_id(0) + 1) * (M // R) // NP,
        num_stages=1,
    ):
        row = block * R + r
        offsets = row[:, None] * D + c[None, :]
        v = tl.load(T + offsets, offset_state_policy="continuous")
        n = v.to(tl.bfloat16).to(tl.float32)
        z = _barrier_round(n, w[None, :])
        s = tl.load(S + offsets, offset_state_policy="continuous").to(
            tl.float32
        )
        tl.store(
            Y + offsets, _barrier_round(z, s), offset_state_policy="continuous"
        )


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
    m, sm = (x.numel() // d, scale.numel() // d)
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
    if (
        xc
        and sc
        and (m >= 1024)
        and (m == sm)
        and (d in (2048, 4096))
        and (m * d < 2147483648)
        and ((x.data_ptr() | weight.data_ptr() | scale.data_ptr()) & 63 == 0)
    ):
        groups = d // 1024 if m < 4096 else 0
        programs = (
            min(12, (m + 64 // groups - 1) // (64 // groups))
            if groups
            else min(12, (m + 63) // 64)
        )
        _entry135.run(
            x,
            weight,
            scale,
            y,
            (m, d, float(eps), groups),
            grid=(programs,),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return y
    if m:
        b = 1 << max(6, (d - 1).bit_length())
        if xc and sc and (d in (2048, 4096)) and (m >= 1024) and (m % 32 == 0):
            if m == sm:
                norm = torch.empty_like(x, dtype=torch.float32)
                four_parts = d == 4096 or (m >= 8192 and m % 64 == 0)
                stats_rows = (64 if d == 2048 else 32) if four_parts else 32
                stats_kernel = _stats_parts4 if four_parts else _split_stats
                affine_rows = 16 if d == 2048 and m >= 8192 else 8
                stats_kernel.run(
                    x,
                    norm,
                    m,
                    d,
                    float(eps),
                    stats_rows,
                    12,
                    grid=(12,),
                    warmup=False,
                    num_warps=1,
                    groups_per_cluster=stats_rows,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                    unroll_num=8,
                )
                _affine_rows.run(
                    norm,
                    weight,
                    scale,
                    y,
                    m,
                    d,
                    affine_rows,
                    12,
                    grid=(12,),
                    warmup=False,
                    num_warps=1,
                    groups_per_cluster=affine_rows,
                    isCloseVectorization=True,
                    isCloseDtypeConvert=True,
                    isCloseMemoryAsync=False,
                    unroll_num=16,
                    enable_fp_fusion=False,
                )
            else:
                rows = (32 if d == 2048 else 16) if m >= 8192 else 8
                _tile.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), b, rows, True, True),
                    grid=(min(m // rows, 12),),
                    warmup=False,
                    num_warps=1,
                    groups_per_cluster=rows,
                    isCloseVectorization=True,
                    isCloseDtypeConvert=True,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                    unroll_num=8,
                )
        elif xc and sc and (m < 1024) and (d <= 4096):
            if m >= 32 and (
                d <= 2048
                and d % 64 == 0
                and (m % 4 == 0)
                or (d == 4096 and m % 2 == 0)
            ):
                rows = 2 if d == 4096 else 4
                _tile.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d, m // sm, float(eps), b, rows, d == b, True),
                    grid=(min(m // rows, 12),),
                    warmup=False,
                    num_warps=1,
                    groups_per_cluster=rows,
                    isCloseVectorization=True,
                    isCloseDtypeConvert=True,
                    enable_fp_fusion=False,
                )
            else:
                _native_lean.run(
                    x,
                    weight,
                    scale,
                    y,
                    (m, d == b, d, m // sm, float(eps), b),
                    grid=(min(m, 12),),
                    warmup=False,
                    num_warps=1,
                    isCloseVectorization=True,
                    isCloseDtypeConvert=True,
                    enable_fp_fusion=False,
                )
        elif xc and sc and (d == b) and (d >= 64) and (m < 1024):
            _materialized.run(
                x,
                weight,
                scale,
                y,
                (m, d, m // sm, float(eps)),
                grid=(min(m, 12),),
                warmup=False,
                num_warps=1,
                isCloseMemoryAsync=True,
                enable_fp_fusion=False,
                isCloseVectorization=True,
                isCloseDtypeConvert=True,
            )
        elif xc and sc and (d % 64 == 0) and (m < 1024):
            _small_sync.run(
                x,
                weight,
                scale,
                y,
                m,
                d,
                m // sm,
                float(eps),
                b,
                True,
                grid=(min(m, 12),),
                warmup=False,
                num_warps=1,
                isCloseMemoryAsync=True,
                enable_fp_fusion=False,
                isCloseVectorization=True,
                isCloseDtypeConvert=True,
            )
        elif xc and sc:
            _lean.run(
                x,
                weight,
                scale,
                y,
                (m, d == b, d, m // sm, float(eps), b),
                grid=(min(m, 12),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
                isCloseVectorization=True,
                isCloseDtypeConvert=True,
            )
        else:
            _row.run(
                x,
                weight,
                scale,
                y,
                (
                    d,
                    m // sm,
                    tuple(x.shape[:-1]),
                    tuple(x.stride()),
                    tuple(scale.shape[:-1]),
                    tuple(scale.stride()),
                    float(eps),
                    b,
                ),
                grid=(m,),
                warmup=False,
                num_warps=1,
                num_stages=1,
                enable_fp_fusion=False,
                isCloseVectorization=True,
                isCloseDtypeConvert=True,
            )
    return y


RAW135 = (
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned short u16v __attribute__((ext_vector_type(16)));\n"
    "union fvbits { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) f32v rne135(f32v v) {\n"
    "    fvbits b;\n"
    "    b.f=v;\n"
    "    u32v bits=b.u;\n"
    "    b.u=(bits+32767u+((bits>>16)&1u))&0xffff0000u;\n"
    "    u32v diff=0x7f800000u-(bits&0x7fffffffu);\n"
    '    __asm__("" : "+v"(diff));\n'
    "    u32v nanmask=0u-(diff>>31);\n"
    '    __asm__("" : "+v"(nanmask));\n'
    "    b.u=(b.u&~nanmask)|(0x7fc00000u&nanmask);\n"
    "    return b.f;\n"
    "}\n"
    "\n"
    "namespace xpu { __attribute__((device)) float rsqrtf(float); }\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_fused135(gu16* X,gu16* W,gu16* S,gu16* Y,i'
    "nt M,int D,float EPS,int pid,int np) {\n"
    "    unsigned int xb[512] __attribute__((aligned(64)));\n"
    "    unsigned int wb[512] __attribute__((aligned(64)));\n"
    "    unsigned int sb[512] __attribute__((aligned(64)));\n"
    "    float sums[16] __attribute__((aligned(64)));\n"
    "    int core=__builtin_xpu_core_id();\n"
    "    for(int row=pid*64+core;row<M;row+=np*64) {\n"
    "        f32v acc=(f32v)(0.0f);\n"
    "        for(int col=0;col<D;col+=1024) {\n"
    "            __builtin_xpu_gm2lm_v3(xb,X+row*D+col,0,2048);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for(int i=0;i<512;i+=16) {\n"
    "                u32v packed=*((u32v*)(xb+i));\n"
    "                fvbits lo,hi;\n"
    "                lo.u=packed<<16;\n"
    "                hi.u=packed&0xffff0000u;\n"
    "                acc=acc+(lo.f*lo.f+hi.f*hi.f);\n"
    "            }\n"
    "        }\n"
    "        *((f32v*)sums)=acc;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        float sum=0.0f;\n"
    "        for(int i=0;i<16;++i) sum=sum+sums[i];\n"
    "        float inv=xpu::rsqrtf(sum/(float)D+EPS);\n"
    "        for(int col=0;col<D;col+=1024) {\n"
    "            __builtin_xpu_gm2lm_v3(xb,X+row*D+col,0,2048);\n"
    "            __builtin_xpu_gm2lm_v3(wb,W+col,0,2048);\n"
    "            __builtin_xpu_gm2lm_v3(sb,S+row*D+col,0,2048);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for(int i=0;i<512;i+=16) {\n"
    "                u32v xp=*((u32v*)(xb+i));\n"
    "                u32v wp=*((u32v*)(wb+i));\n"
    "                u32v sp=*((u32v*)(sb+i));\n"
    "                fvbits xl,xh,wl,wh,sl,sh,yl,yh;\n"
    "                xl.u=xp<<16; xh.u=xp&0xffff0000u;\n"
    "                wl.u=wp<<16; wh.u=wp&0xffff0000u;\n"
    "                sl.u=sp<<16; sh.u=sp&0xffff0000u;\n"
    "                f32v nl=rne135(xl.f*inv);\n"
    "                f32v nh=rne135(xh.f*inv);\n"
    "                f32v zl=rne135(nl*wl.f);\n"
    "                f32v zh=rne135(nh*wh.f);\n"
    "                yl.f=rne135(zl*sl.f);\n"
    "                yh.f=rne135(zh*sh.f);\n"
    "                *((u32v*)(xb+i))=(yl.u>>16)|(yh.u&0xffff0000u);\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(Y+row*D+col,xb,0,2048);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "namespace xpu { __attribute__((device)) float rsqrtf(float); }\n"
    "__attribute__((shared)) static float partials135[64];\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_coop135(gu16* X,gu16* W,gu16* S,gu16* Y,in'
    "t M,int D,float EPS,int G,int pid,int np) {\n"
    "    unsigned int xb[512] __attribute__((aligned(64)));\n"
    "    unsigned int wb[512] __attribute__((aligned(64)));\n"
    "    unsigned int sb[512] __attribute__((aligned(64)));\n"
    "    float sums[16] __attribute__((aligned(64)));\n"
    "    int core=__builtin_xpu_core_id();\n"
    "    int group_size=G;\n"
    "    int chunk=D/group_size;\n"
    "    int rows=64/group_size;\n"
    "    int group=core/group_size;\n"
    "    int col=(core%group_size)*chunk;\n"
    "    int blocks=(M+rows-1)/rows;\n"
    "    if(pid>=blocks) return;\n"
    "    __builtin_xpu_gm2lm_v3(wb,W+col,0,chunk*2);\n"
    "    for(int block=pid;block<blocks;block+=np) {\n"
    "        int row=block*rows+group;\n"
    "        float partial=0.0f;\n"
    "        if(row<M) {\n"
    "            __builtin_xpu_gm2lm_v3(xb,X+row*D+col,0,chunk*2);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            f32v acc=(f32v)(0.0f);\n"
    "            for(int i=0;i<chunk/2;i+=16) {\n"
    "                u32v packed=*((u32v*)(xb+i));\n"
    "                fvbits lo,hi;\n"
    "                lo.u=packed<<16;\n"
    "                hi.u=packed&0xffff0000u;\n"
    "                acc=acc+(lo.f*lo.f+hi.f*hi.f);\n"
    "            }\n"
    "            *((f32v*)sums)=acc;\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for(int i=0;i<16;++i) partial=partial+sums[i];\n"
    "        }\n"
    "        partials135[core]=partial;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        float total=0.0f;\n"
    "        for(int i=0;i<group_size;++i) total+=partials135[group*group_size+i];\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        if(row<M) {\n"
    "            float inv=xpu::rsqrtf(total/(float)D+EPS);\n"
    "            __builtin_xpu_gm2lm_v3(sb,S+row*D+col,0,chunk*2);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for(int i=0;i<chunk/2;i+=16) {\n"
    "                u32v xp=*((u32v*)(xb+i));\n"
    "                u32v wp=*((u32v*)(wb+i));\n"
    "                u32v sp=*((u32v*)(sb+i));\n"
    "                fvbits xl,xh,wl,wh,sl,sh,yl,yh;\n"
    "                xl.u=xp<<16; xh.u=xp&0xffff0000u;\n"
    "                wl.u=wp<<16; wh.u=wp&0xffff0000u;\n"
    "                sl.u=sp<<16; sh.u=sp&0xffff0000u;\n"
    "                f32v nl=rne135(xl.f*inv);\n"
    "                f32v nh=rne135(xh.f*inv);\n"
    "                f32v zl=rne135(nl*wl.f);\n"
    "                f32v zh=rne135(nh*wh.f);\n"
    "                yl.f=rne135(zl*sl.f);\n"
    "                yh.f=rne135(zh*sh.f);\n"
    "                *((u32v*)(xb+i))=(yl.u>>16)|(yh.u&0xffff0000u);\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(Y+row*D+col,xb,0,chunk*2);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW135, flags=("-ffp-contract=off",))
def _device_fused135(X, W, S, Y, M, D, EPS, pid, np):
    pass


@tle.raw.dialect("xpu3", source=RAW135, flags=("-ffp-contract=off",))
def _device_coop135(X, W, S, Y, M, D, EPS, G, pid, np):
    pass


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _entry135(X, W, S, Y, P: tl.constexpr):
    M: tl.constexpr = P[0]
    D: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    G: tl.constexpr = P[3]
    if G:
        tle.raw.call(
            _device_coop135,
            (
                X,
                W,
                S,
                Y,
                tl.full((), M, tl.int32),
                tl.full((), D, tl.int32),
                tl.full((), EPS, tl.float32),
                tl.full((), G, tl.int32),
                tl.program_id(0),
                tl.num_programs(0),
            ),
        )
    else:
        tle.raw.call(
            _device_fused135,
            (
                X,
                W,
                S,
                Y,
                tl.full((), M, tl.int32),
                tl.full((), D, tl.int32),
                tl.full((), EPS, tl.float32),
                tl.program_id(0),
                tl.num_programs(0),
            ),
        )


__all__ = ["rmsnorm_scale"]
