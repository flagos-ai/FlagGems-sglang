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
from triton.experimental import tle


@triton.jit
def _pack(
    X,
    Y,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    EXP: tl.constexpr,
    SQUARE: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rem = i
    offset = tl.full((BLOCK,), 0, tl.int32)
    for axis in tl.static_range(len(SH) - 1, -1, -1):
        offset += (rem % SH[axis]) * ST[axis]
        rem //= SH[axis]
    x = tl.load(X + offset, i < N, 0).to(tl.float32)
    if EXP:
        x = tl.exp(x)
    if SQUARE:
        x = tl.fma(x, x, -0.0)
    if SCALE != 1.0:
        x = tl.fma(x, SCALE, -0.0)
    tl.store(Y + i, x, i < N)


@triton.jit
def _pairs(
    X,
    Y,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // M
    c = i % M * 2
    if N % 2 == 0:
        a = tl.load(X + 2 * i, i < ROWS * M, 0)
        b = tl.load(X + 2 * i + 1, i < ROWS * M, 0)
    else:
        a = tl.load(X + row * N + c, row < ROWS, 0)
        b = tl.load(X + row * N + tl.minimum(c + 1, N - 1), row < ROWS, 0)
        b = tl.where(c + 1 < N, b, 0.0)
    tl.store(Y + i, a + b, i < ROWS * M)


@triton.jit
def _normalize(
    X,
    S,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    if NORM:
        s = tl.load(S + i // K, i < N, 0)
        x = tl.div_rn(x, tl.sqrt(s + 1.0e-6))
    tl.store(Y + i, tl.fma(x, SCALE, -0.0), i < N)


@triton.jit(do_not_specialize=["STEP"])
def _decay(
    S0,
    S1,
    K,
    D,
    P,
    STEP,
    T: tl.constexpr,
    H: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // KD
    ki = i % KD
    head = row // VD % H
    bi = row // (VD * H)
    offset = ((bi * T + STEP) * H + head) * KD + ki
    s = tl.load(S0 + i, i < N, 0)
    d = tl.load(D + offset, i < N, 0)
    k = tl.load(K + offset, i < N, 0)
    s = tl.fma(s, d, -0.0)
    tl.store(S1 + i, s, i < N)
    tl.store(P + i, tl.fma(s, k, -0.0), i < N)


@triton.jit(do_not_specialize=["STEP"])
def _update(
    S,
    K,
    Q,
    V,
    Beta,
    Pred,
    P,
    STEP,
    T: tl.constexpr,
    H: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    N: tl.constexpr,
    PV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // KD
    ki = i % KD
    vi = row % VD
    head = row // VD % H
    bi = row // (VD * H)
    token = (bi * T + STEP) * H + head
    q = tl.load(Q + token * KD + ki, i < N, 0)
    k = tl.load(K + token * KD + ki, i < N, 0)
    v = tl.load(V + token * VD + vi, i < N, 0)
    if PV:
        beta = tl.load(Beta + token * VD + vi, i < N, 0)
    else:
        beta = tl.load(Beta + token, i < N, 0)
    pred = tl.load(Pred + row, i < N, 0)
    delta = tl.fma(v - pred, beta, -0.0)
    s = tl.load(S + i, i < N, 0)
    s = s + tl.fma(delta, k, -0.0)
    tl.store(S + i, s, i < N)
    tl.store(P + i, tl.fma(s, q, -0.0), i < N)


@triton.jit(do_not_specialize=["STEP"])
def _write(
    P,
    O,
    STEP,
    ROWS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    vi = row % VD
    head = row // VD % H
    bi = row // (VD * H)
    x = tl.load(P + row, row < ROWS, 0)
    tl.store(O + ((bi * T + STEP) * H + head) * VD + vi, x, row < ROWS)


def _packed(x, exponent=False, square=False, scale=1.0):
    y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    if x.numel():
        _pack[(triton.cdiv(x.numel(), 4096),)](
            x,
            y,
            x.numel(),
            tuple(x.shape),
            x.stride(),
            exponent,
            square,
            4096,
            scale,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y


def _reduce(x, rows, width):
    while width > 1:
        next_width = (width + 1) // 2
        y = torch.empty(
            (rows, next_width), device=x.device, dtype=torch.float32
        )
        _pairs[(triton.cdiv(rows * next_width, 128),)](
            x,
            y,
            rows,
            width,
            next_width,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        x = y
        width = next_width
    return x


def _fallback(q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel):
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    rows = b * h * vd
    q32 = _packed(q)
    k32 = _packed(k)
    v32 = _packed(v)
    decay = _packed(g, True)
    beta32 = _packed(beta)
    norm = use_qk_l2norm_in_kernel
    qs = _reduce(_packed(q, False, True), b * t * h, kd) if norm else q32
    ks = _reduce(_packed(k, False, True), b * t * h, kd) if norm else k32
    qn = torch.empty_like(q32)
    kn = torch.empty_like(k32)
    _normalize[(triton.cdiv(q.numel(), 256),)](
        q32,
        qs,
        qn,
        q.numel(),
        kd,
        scale,
        norm,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _normalize[(triton.cdiv(k.numel(), 256),)](
        k32,
        ks,
        kn,
        k.numel(),
        kd,
        1.0,
        norm,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    previous = initial_state.float().contiguous()
    final = torch.empty(
        initial_state.shape, device=initial_state.device, dtype=torch.float32
    )
    product = torch.empty((rows, kd), device=q.device, dtype=torch.float32)
    output = torch.empty((b, t, h, vd), device=q.device, dtype=q.dtype)
    total = rows * kd
    for step in range(t):
        _decay[(triton.cdiv(total, 256),)](
            previous,
            final,
            kn,
            decay,
            product,
            step,
            t,
            h,
            kd,
            vd,
            total,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        pred = _reduce(product, rows, kd)
        _update[(triton.cdiv(total, 256),)](
            final,
            kn,
            qn,
            v32,
            beta32,
            pred,
            product,
            step,
            t,
            h,
            kd,
            vd,
            total,
            beta.ndim == v.ndim,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        value = _reduce(product, rows, kd)
        _write[(triton.cdiv(rows, 128),)](
            value,
            output,
            step,
            rows,
            t,
            h,
            vd,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        previous = final
    return output, final.to(initial_state.dtype)


RAW201 = (
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mul32(f32v x, f32v y) {\n"
    "    f32v z=x*y;\n"
    '    __asm__ volatile("" : "+v"(z));\n'
    "    return z;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v pair16(f32v x) {\n"
    "    x=__builtin_shufflevector(x,x,0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14)\n"
    "      +__builtin_shufflevector(x,x,1,3,5,7,9,11,13,15,1,3,5,7,9,11,13,15);\n"
    "    x=__builtin_shufflevector(x,x,0,2,4,6,0,2,4,6,0,2,4,6,0,2,4,6)\n"
    "      +__builtin_shufflevector(x,x,1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7);\n"
    "    x=__builtin_shufflevector(x,x,0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2)\n"
    "      +__builtin_shufflevector(x,x,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3);\n"
    "    return __builtin_shufflevector(x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)\n"
    "      +__builtin_shufflevector(x,x,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_recurrent201(\n'
    "    gf32* Q,gf32* K,gf32* D,gf32* V,gf32* Beta,gf32* S0,gf32* O,gf32* S1,\n"
    "    int ROWS,int T,int H,int KD,int VD,int PV,int pid,int np) {\n"
    "    float sb[256] __attribute__((aligned(64)));\n"
    "    float qb[256] __attribute__((aligned(64)));\n"
    "    float kb[256] __attribute__((aligned(64)));\n"
    "    float db[256] __attribute__((aligned(64)));\n"
    "    float value[16] __attribute__((aligned(64)));\n"
    "    float beta[16] __attribute__((aligned(64)));\n"
    "    float ob[16] __attribute__((aligned(64)));\n"
    "    f32v partial[16];\n"
    "    int core=__builtin_xpu_core_id();\n"
    "    for(int row=pid*64+core;row<ROWS;row+=np*64) {\n"
    "        int vi=row%VD;\n"
    "        int hi=(row/VD)%H;\n"
    "        int bi=row/(VD*H);\n"
    "        __builtin_xpu_gm2lm_v3(sb,S0+row*KD,0,KD*4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for(int ti=0;ti<T;++ti) {\n"
    "            int token=(bi*T+ti)*H+hi;\n"
    "            __builtin_xpu_gm2lm_v3(qb,Q+token*KD,0,KD*4);\n"
    "            __builtin_xpu_gm2lm_v3(kb,K+token*KD,0,KD*4);\n"
    "            __builtin_xpu_gm2lm_v3(db,D+token*KD,0,KD*4);\n"
    "            __builtin_xpu_gm2lm_v3(value,V+token*VD+vi,0,4);\n"
    "            __builtin_xpu_gm2lm_v3(beta,Beta+(PV?token*VD+vi:token),0,4);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for(int j=0;j<KD;j+=16) {\n"
    "                f32v s=mul32(*((f32v*)(sb+j)),*((f32v*)(db+j)));\n"
    "                *((f32v*)(sb+j))=s;\n"
    "                partial[j/16]=pair16(mul32(s,*((f32v*)(kb+j))));\n"
    "            }\n"
    "            int width=KD/16;\n"
    "            while(width>1) {\n"
    "                int next=(width+1)/2;\n"
    "                for(int j=0;j<width/2;++j) partial[j]=partial[j*2]+partial[j*2+1];\n"
    "                if(width&1) partial[next-1]=partial[width-1];\n"
    "                width=next;\n"
    "            }\n"
    "            f32v delta=mul32((f32v)(value[0])-partial[0],(f32v)(beta[0]));\n"
    "            for(int j=0;j<KD;j+=16) {\n"
    "                f32v s=*((f32v*)(sb+j))+mul32(delta,*((f32v*)(kb+j)));\n"
    "                *((f32v*)(sb+j))=s;\n"
    "                partial[j/16]=pair16(mul32(s,*((f32v*)(qb+j))));\n"
    "            }\n"
    "            width=KD/16;\n"
    "            while(width>1) {\n"
    "                int next=(width+1)/2;\n"
    "                for(int j=0;j<width/2;++j) partial[j]=partial[j*2]+partial[j*2+1];\n"
    "                if(width&1) partial[next-1]=partial[width-1];\n"
    "                width=next;\n"
    "            }\n"
    "            *((f32v*)ob)=partial[0];\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(O+token*VD+vi,ob,0,4);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "        __builtin_xpu_lm2gm_v3(S1+row*KD,sb,0,KD*4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW201, flags=("-ffp-contract=off",))
def _device_recurrent201(
    Q, K, D, V, Beta, S0, Out, S1, ROWS, T, H, KD, VD, PV, pid, np
):
    pass


@triton.jit
def _entry201(Q, K, D, V, Beta, S0, Out, S1, P: tl.constexpr):
    tle.raw.call(
        _device_recurrent201,
        (
            Q,
            K,
            D,
            V,
            Beta,
            S0,
            Out,
            S1,
            tl.full((), P[0], tl.int32),
            tl.full((), P[1], tl.int32),
            tl.full((), P[2], tl.int32),
            tl.full((), P[3], tl.int32),
            tl.full((), P[4], tl.int32),
            tl.full((), P[5], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def fused_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel
):
    b, t, h, kd = k.shape
    vd = v.shape[-1]
    if t == 0:
        return torch.empty(
            (b, t, h, vd), dtype=q.dtype, device=q.device
        ), _packed(initial_state).to(initial_state.dtype)
    if kd % 16 or kd > 256 or initial_state.dtype != torch.float32:
        return _fallback(
            q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel
        )
    q32 = _packed(q, scale=1.0 if use_qk_l2norm_in_kernel else scale)
    k32 = _packed(k)
    v32 = _packed(v)
    decay = _packed(g, True)
    beta32 = _packed(beta)
    norm = use_qk_l2norm_in_kernel
    qs = _reduce(_packed(q, False, True), b * t * h, kd) if norm else q32
    ks = _reduce(_packed(k, False, True), b * t * h, kd) if norm else k32
    if norm:
        qn = torch.empty_like(q32)
        kn = torch.empty_like(k32)
        _normalize[(triton.cdiv(q.numel(), 256),)](
            q32,
            qs,
            qn,
            q.numel(),
            kd,
            scale,
            True,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _normalize[(triton.cdiv(k.numel(), 256),)](
            k32,
            ks,
            kn,
            k.numel(),
            kd,
            1.0,
            True,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        qn, kn = q32, k32
    output32 = torch.empty((b, t, h, vd), dtype=torch.float32, device=q.device)
    output = torch.empty((b, t, h, vd), dtype=q.dtype, device=q.device)
    final = torch.empty(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    rows = b * h * vd
    if rows:
        _entry201[(min(12, triton.cdiv(rows, 64)),)](
            qn,
            kn,
            decay,
            v32,
            beta32,
            initial_state.contiguous(),
            output32,
            final,
            (rows, t, h, kd, vd, int(beta.ndim == v.ndim)),
            num_warps=1,
            enable_fp_fusion=False,
        )
    if output.numel():
        _pack[(triton.cdiv(output.numel(), 4096),)](
            output32,
            output,
            output.numel(),
            tuple(output32.shape),
            output32.stride(),
            False,
            False,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, final


__all__ = ["fused_recurrent_kda"]
