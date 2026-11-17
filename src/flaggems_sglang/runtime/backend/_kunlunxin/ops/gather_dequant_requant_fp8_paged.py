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
def _decode(raw):
    raw = raw.to(tl.int32)
    mag = raw & 127
    exp = mag >> 3
    mant = mag & 7
    normal = (exp + 120 << 23 | mant << 20).to(tl.float32, bitcast=True)
    value = tl.where(exp == 0, mant.to(tl.float32) * 0.001953125, normal)
    bits = value.to(tl.int32, bitcast=True) | (raw & 128) << 24
    bits = tl.where(mag == 127, (raw & 128) << 24 | 2143289344, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _encode(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = bits & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    rounded = mag.to(tl.float32, bitcast=True) + 16384.0
    sub = rounded.to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < 1015021568, sub, normal)
    code = tl.where(mag > 1139277824, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


RAW184 = (
    "typedef unsigned int __attribute__((address_space(1))) gu32;"
    "\n"
    "typedef int __attribute__((address_space(1))) gi32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short u16v __attribute__((ext_vector_type(1"
    "6)));\n"
    "union fvbits { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v magnitude184(f32v"
    " v) {\n"
    "    fvbits b;\n"
    "    b.f=v;\n"
    "    u32v mag=__builtin_elementwise_min(b.u&0x7fffffffu,(u32v"
    ")(0x43f00000u));\n"
    "    u32v exponent=__builtin_elementwise_max(mag>>23,(u32v)(1"
    "21u));\n"
    "    fvbits rounded,bias;\n"
    "    bias.u=(exponent+20u)<<23;\n"
    "    rounded.u=mag;\n"
    "    rounded.f=rounded.f+bias.f;\n"
    "    u32v code=rounded.u-exponent*8388600u-167773128u;\n"
    "    return code;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode184(f32v v)"
    " {\n"
    "    fvbits b;\n"
    "    b.f=v;\n"
    "    return magnitude184(v)|((b.u>>24)&128u);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v nope184(u32v raw,"
    "float scale) {\n"
    "    u32v mag=raw&127u;\n"
    "    fvbits value;\n"
    "    value.u=(mag+960u)<<20;\n"
    "    f32v sub=value.f+value.f-0.015625f;\n"
    "    value.f=__builtin_elementwise_min(sub,value.f);\n"
    "    union {float f;unsigned int u;} s;\n"
    "    s.f=scale;\n"
    "    unsigned int sign=s.u>>24;\n"
    "    s.u=s.u&0x7fffffffu;\n"
    "    u32v result=magnitude184(value.f*s.f);\n"
    "    result=result|((raw^sign)&128u);\n"
    "    return result;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_gather184(gu'
    "32* Q,gi32* P,gu32* O,int N,int TOTAL,int M,int QS,int PS,in"
    "t OS,int pid,int np) {\n"
    "    unsigned int qbuf[2][176] __attribute__((aligned(64)));\n"
    "    unsigned int rbuf[2][32] __attribute__((aligned(64)));\n"
    "    unsigned int obuf[2][160] __attribute__((aligned(64)));\n"
    "    int pb[16] __attribute__((aligned(64)));\n"
    "    int core=__builtin_xpu_core_id();\n"
    "    int row=pid*64+core;\n"
    "    int slot=0;\n"
    "    if(row<N) {\n"
    "        __builtin_xpu_gm2lm_v3(pb,P+row*PS,0,4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        int src=pb[0];\n"
    "        if(src<0) src+=M;\n"
    "        __builtin_xpu_gm2lm_v3(qbuf[0],Q+(long long)src*QS,0"
    ",528);\n"
    "        __builtin_xpu_gm2lm_v3(rbuf[0],Q+(long long)src*QS+1"
    "32,0,128);\n"
    "        int next=row+np*64;\n"
    "        if(next<N) __builtin_xpu_gm2lm_v3(pb,P+next*PS,0,4);"
    "\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    for(;row<TOTAL;row+=np*64) {\n"
    "        int next=row+np*64;\n"
    "        if(next<N) {\n"
    "            int src=pb[0];\n"
    "            if(src<0) src+=M;\n"
    "            __builtin_xpu_gm2lm_v3(qbuf[1-slot],Q+(long long"
    ")src*QS,0,528);\n"
    "            __builtin_xpu_gm2lm_v3(rbuf[1-slot],Q+(long long"
    ")src*QS+132,0,128);\n"
    "            int after=next+np*64;\n"
    "            if(after<N) __builtin_xpu_gm2lm_v3(pb,P+after*PS"
    ",0,4);\n"
    "        }\n"
    "        unsigned int* qb=qbuf[slot];\n"
    "        unsigned int* rb=rbuf[slot];\n"
    "        unsigned int* ob=obuf[slot];\n"
    "        if(row<N) {\n"
    "            #pragma unroll\n"
    "            for(int i=0;i<128;i+=16) {\n"
    "                u32v word=*((u32v*)(qb+i));\n"
    "                float scale=((float*)(qb+128))[i/32];\n"
    "                u32v result=(u32v)(0u);\n"
    "            #pragma unroll\n"
    "                for(int lane=0;lane<4;++lane) {\n"
    "                    u32v raw=(word>>(lane*8))&255u;\n"
    "                    result=result|(nope184(raw,scale)<<(lane"
    "*8));\n"
    "                }\n"
    "                u32v absent=(~word)&(u32v)(0x7f7f7f7fu);\n"
    "                u32v carry=absent+(u32v)(0x7f7f7f7fu);\n"
    '                __asm__("" : "+v"(carry));\n'
    "                u32v nanmask=((~carry)&(u32v)(0x80808080u))>"
    ">7;\n"
    "                nanmask=nanmask*255u;\n"
    '                __asm__("" : "+v"(nanmask));\n'
    "                result=result^((result^word)&nanmask);\n"
    "                *((u32v*)(ob+i))=result;\n"
    "            }\n"
    "            for(int i=0;i<2;++i) {\n"
    "                u32v word=*((u32v*)(rb+i*16));\n"
    "                fvbits lo,hi;\n"
    "                lo.u=word<<16;\n"
    "                hi.u=word&0xffff0000u;\n"
    "                u32v code=encode184(lo.f)|(encode184(hi.f)<<"
    "8);\n"
    "                *((u16v*)((unsigned short*)(ob+128)+i*16))=_"
    "_builtin_convertvector(code,u16v);\n"
    "            }\n"
    "        } else {\n"
    "            for(int i=0;i<144;i+=16) *((u32v*)(ob+i))=(u32v)"
    "(0u);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(O+(long long)row*OS,ob,0,576)"
    ";\n"
    "        slot=1-slot;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3",
    source=RAW184,
    flags=("-ffp-contract=off", "-fno-vectorize", "-fno-slp-vectorize"),
)
def _device_gather184(Q, P, OUT, N, TOTAL, M, QS, PS, OS, pid, np):
    pass


@triton.jit
def _raw_entry(
    Q,
    P,
    OUT,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    M: tl.constexpr,
    QS: tl.constexpr,
    PS: tl.constexpr,
    OS: tl.constexpr,
):
    tle.raw.call(
        _device_gather184,
        (
            Q,
            P,
            OUT,
            tl.full((), N, tl.int32),
            tl.full((), TOTAL, tl.int32),
            tl.full((), M, tl.int32),
            tl.full((), QS, tl.int32),
            tl.full((), PS, tl.int32),
            tl.full((), OS, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


@triton.jit
def _general_layout(
    Q,
    P,
    OUT,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    M: tl.constexpr,
    QS: tl.constexpr,
    QC: tl.constexpr,
    PS: tl.constexpr,
    OS: tl.constexpr,
    OC: tl.constexpr,
    G: tl.constexpr,
    NORMALIZE_ZERO: tl.constexpr,
):
    col = tl.arange(0, 512).to(tl.int64)
    tail = tl.arange(0, 64).to(tl.int64)
    for row in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        if row < N:
            src = tl.load(P + row64 * PS).to(tl.int64)
            src = tl.where(src < 0, src + M, src)
            base = Q + src * QS
            raw = tl.load(base + col * QC)
            s = 512 + col // G * 4
            s0 = tl.load(base + s * QC).to(tl.uint32)
            s1 = tl.load(base + (s + 1) * QC).to(tl.uint32)
            s2 = tl.load(base + (s + 2) * QC).to(tl.uint32)
            s3 = tl.load(base + (s + 3) * QC).to(tl.uint32)
            bits = s0 | s1 << 8 | s2 << 16 | s3 << 24
            scale = bits.to(tl.float32, bitcast=True)
            value = _decode(raw) * scale
            sign = (raw.to(tl.int32) ^ (bits >> 24).to(tl.int32)) & 128
            sign = tl.where(
                raw.to(tl.int32) & 127 == 127, raw.to(tl.int32) & 128, sign
            )
            if NORMALIZE_ZERO:
                exact_zero = (
                    (raw.to(tl.int32) & 127 == 0) | (bits & 2147483647 == 0)
                ) & (raw.to(tl.int32) & 127 != 127)
                sign = tl.where((row < N // 4 * 4) & exact_zero, 0, sign)
            encoded = (_encode(value).to(tl.int32) & 127 | sign).to(tl.uint8)
            tl.store(OUT + row64 * OS + col * OC, encoded)
            r = 512 + 512 // G * 4 + tail * 2
            r0 = tl.load(base + r * QC).to(tl.uint32)
            r1 = tl.load(base + (r + 1) * QC).to(tl.uint32)
            rb = r0 | r1 << 8
            rope = (rb << 16).to(tl.float32, bitcast=True)
            encoded_rope = (
                _encode(rope).to(tl.int32) & 127 | (rb >> 8).to(tl.int32) & 128
            ).to(tl.uint8)
            tl.store(OUT + row64 * OS + (512 + tail) * OC, encoded_rope)
        else:
            tl.store(OUT + row64 * OS + col * OC, tl.full((512,), 0, tl.uint8))
            tl.store(
                OUT + row64 * OS + (512 + tail) * OC,
                tl.full((64,), 0, tl.uint8),
            )


def _compute(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total == 0:
        return out
    qs = quant_k_cache.stride()
    os = out.stride()
    ps = page_table_1_flattened.stride(0)
    m = quant_k_cache.shape[0]
    aligned = (
        qs[-1] == 1
        and os[-1] == 1
        and (
            (
                quant_k_cache.data_ptr()
                | out.data_ptr()
                | qs[0]
                | qs[1]
                | os[0]
                | os[1]
            )
            & 3
            == 0
        )
    )
    compatible = (
        aligned
        and m * qs[0] < 2147483648
        and (out.shape[0] * os[0] < 2147483648)
        and (n * ps < 2147483648)
    )
    if compatible:
        _raw_entry[12,](
            quant_k_cache.view(torch.uint32),
            page_table_1_flattened,
            out.view(torch.uint32),
            n,
            total,
            m,
            qs[0] // 4,
            ps,
            os[0] // 4,
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        _general_layout[min(total, 32768),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            n,
            total,
            m,
            qs[0],
            qs[-1],
            ps,
            os[0],
            os[-1],
            group_size,
            False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _alias_writeback(
    T, OUT, TOTAL: tl.constexpr, OS: tl.constexpr, OC: tl.constexpr
):
    col = tl.arange(0, 512).to(tl.int64)
    tail = tl.arange(0, 64).to(tl.int64)
    for row in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        r = row.to(tl.int64)
        first = tl.load(T + r * 576 + col)
        last = tl.load(T + r * 576 + 512 + tail)
        tl.store(OUT + r * OS + col * OC, first)
        tl.store(OUT + r * OS + (512 + tail) * OC, last)


def gather_dequant_requant_fp8_paged(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total == 0:
        return out
    qs = quant_k_cache.stride()
    os = out.stride()
    ps = page_table_1_flattened.stride(0)
    m = quant_k_cache.shape[0]
    qp = quant_k_cache.data_ptr()
    pp = page_table_1_flattened.data_ptr()
    start = out.data_ptr()
    end = start + (total - 1) * os[0] + 575 * os[-1] + 1
    qe = qp + (quant_k_cache.shape[0] - 1) * qs[0] + 655 * qs[-1] + 1
    pe = pp + (n - 1) * ps * 4 + 4
    if start < qe and qp < end or (start < pe and pp < end):
        temp = torch.empty((total, 1, 576), dtype=out.dtype, device=out.device)
        _compute(
            quant_k_cache, page_table_1_flattened, group_size, extra_rows, temp
        )
        _alias_writeback[min(total, 80),](
            temp.view(torch.uint8),
            out.view(torch.uint8),
            total,
            os[0],
            os[-1],
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    aligned = (
        qs[-1] == 1
        and os[-1] == 1
        and ((qp | start | qs[0] | qs[1] | os[0] | os[1]) & 3 == 0)
    )
    compatible = (
        aligned
        and m * qs[0] < 2147483648
        and (out.shape[0] * os[0] < 2147483648)
        and (n * ps < 2147483648)
    )
    if compatible:
        _raw_entry[12,](
            quant_k_cache.view(torch.uint32),
            page_table_1_flattened,
            out.view(torch.uint32),
            n,
            total,
            m,
            qs[0] // 4,
            ps,
            os[0] // 4,
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        _general_layout[min(total, 32768),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            out.view(torch.uint8),
            n,
            total,
            m,
            qs[0],
            qs[-1],
            ps,
            os[0],
            os[-1],
            group_size,
            False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["gather_dequant_requant_fp8_paged"]
