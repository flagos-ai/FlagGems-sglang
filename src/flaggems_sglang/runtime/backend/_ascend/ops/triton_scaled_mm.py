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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f16_i16(x_bytes):
    v = x_bytes.to(tl.int16)
    e = (v >> 3) & 0xF
    m = v & 0x7
    bits = ((e + 8) << 10) | (m << 7) | ((v & 0x80) << 8)
    norm = bits.to(tl.float16, bitcast=True)
    sub = m.to(tl.float16) * 0.001953125
    sub = tl.where(v >= 128, -sub, sub)
    return tl.where(e == 0, sub, norm)


@triton.jit
def _decode_kernel(
    x_ptr,
    y_ptr,
    numel,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    raw = tl.load(x_ptr + offs, mask=mask, other=0)
    if MODE == 1:
        val = _e4m3_to_f16_i16(raw)
    elif MODE == 2:
        val = (raw.to(tl.uint16) << 8).to(tl.float16, bitcast=True)
    else:
        val = raw.to(tl.float16)
    tl.store(y_ptr + offs, val, mask=mask)


@triton.jit
def _scaled_mm_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,
    bias_s0,
    bias_s1,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M > 1:
        group_size = GROUP_M * num_n
        gid = pid // group_size
        first_m = gid * GROUP_M
        gsize = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % group_size) % gsize)
        pid_n = (pid % group_size) // gsize
    else:
        pid_m = pid // num_n
        pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=msk_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=msk_n[None, :], other=0.0)
        else:
            k_ok = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=msk_m[:, None] & k_ok[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=msk_n[None, :] & k_ok[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    if SA_BCAST:
        sa = tl.load(sa_ptr + tl.zeros((BLOCK_M,), dtype=tl.int32)).to(
            tl.float32
        )
    else:
        sa = tl.load(sa_ptr + offs_m, mask=msk_m, other=0.0).to(tl.float32)
    if SB_BCAST:
        sb = tl.load(sb_ptr + tl.zeros((BLOCK_N,), dtype=tl.int32)).to(
            tl.float32
        )
    else:
        sb = tl.load(sb_ptr + offs_n, mask=msk_n, other=0.0).to(tl.float32)
    acc = acc * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * bias_s0 + offs_n[None, :] * bias_s1,
            mask=msk_m[:, None] & msk_n[None, :],
            other=0.0,
        )
        acc += bias.to(tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=msk_m[:, None] & msk_n[None, :],
    )


@triton.jit
def _scaled_mm_gemm_fused(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,
    bias_s0,
    bias_s1,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
    EVEN_K: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a_raw = tl.load(a_ptrs, mask=msk_m[:, None], other=0)
            b_raw = tl.load(b_ptrs, mask=msk_n[None, :], other=0)
        else:
            k_ok = offs_k < K - k * BLOCK_K
            a_raw = tl.load(
                a_ptrs, mask=msk_m[:, None] & k_ok[None, :], other=0
            )
            b_raw = tl.load(
                b_ptrs, mask=msk_n[None, :] & k_ok[:, None], other=0
            )
        if MODE == 1:
            a = _e4m3_to_f16_i16(a_raw)
            b = _e4m3_to_f16_i16(b_raw)
        elif MODE == 2:
            a = (a_raw.to(tl.uint16) << 8).to(tl.float16, bitcast=True)
            b = (b_raw.to(tl.uint16) << 8).to(tl.float16, bitcast=True)
        else:
            a = a_raw.to(tl.float16)
            b = b_raw.to(tl.float16)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    if SA_BCAST:
        sa = tl.load(sa_ptr + tl.zeros((BLOCK_M,), dtype=tl.int32)).to(
            tl.float32
        )
    else:
        sa = tl.load(sa_ptr + offs_m, mask=msk_m, other=0.0).to(tl.float32)
    if SB_BCAST:
        sb = tl.load(sb_ptr + tl.zeros((BLOCK_N,), dtype=tl.int32)).to(
            tl.float32
        )
    else:
        sb = tl.load(sb_ptr + offs_n, mask=msk_n, other=0.0).to(tl.float32)
    acc = acc * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * bias_s0 + offs_n[None, :] * bias_s1,
            mask=msk_m[:, None] & msk_n[None, :],
            other=0.0,
        )
        acc += bias.to(tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=msk_m[:, None] & msk_n[None, :],
    )


_DECODE_BLOCK = 8192
_DECODE_WARPS = 4
_FP8_FUSED_M_MAX = 64
_FP8_FUSED_N_MAX = 4608
_I8_FUSED_M_MAX = 128


def _decode_mode(dtype):
    if dtype == torch.float8_e4m3fn:
        return 1
    if dtype == torch.float8_e5m2:
        return 2
    return 0


def _launch_gemm(
    a16,
    b16,
    out,
    scale_a_c,
    scale_b_c,
    bias_arg,
    bias_s0,
    bias_s1,
    has_bias,
    sa_bcast,
    sb_bcast,
    M,
    N,
    K,
    even_k,
    cfg,
):
    block_m, block_n, block_k, group_m, warps, stages = cfg
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _scaled_mm_gemm[grid](
        a16,
        b16,
        out,
        scale_a_c,
        scale_b_c,
        bias_arg,
        M,
        N,
        K,
        K,
        N,
        N,
        bias_s0,
        bias_s1,
        HAS_BIAS=has_bias,
        SA_BCAST=sa_bcast,
        SB_BCAST=sb_bcast,
        EVEN_K=even_k,
        GROUP_M=group_m,
        NUM_STAGES=stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
    )


def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    M, K = input.shape
    Kw, N = weight.shape

    in_dtype = input.dtype
    mode = _decode_mode(in_dtype)

    input_c = input.contiguous()
    weight_c = weight.contiguous()

    scale_a_c = scale_a.contiguous()
    scale_b_c = scale_b.contiguous()
    sa_bcast = scale_a_c.shape[0] == 1
    sb_bcast = scale_b_c.shape[0] == 1

    has_bias = bias is not None
    if has_bias:
        bias_c = bias.contiguous()
        if bias_c.dim() == 1:
            bias_c = bias_c.view(1, -1)
        bias_s0 = bias_c.stride(0) if bias_c.shape[0] > 1 else 0
        bias_s1 = bias_c.stride(1) if bias_c.shape[1] > 1 else 0
        bias_arg = bias_c
    else:
        bias_s0 = 0
        bias_s1 = 0
        bias_arg = input_c

    out = torch.empty((M, N), device=input.device, dtype=out_dtype)
    even_k = K % 128 == 0

    a_src = input_c.view(torch.uint8) if mode != 0 else input_c
    b_src = weight_c.view(torch.uint8) if mode != 0 else weight_c

    fused = (
        (M <= _I8_FUSED_M_MAX)
        if mode == 0
        else (M <= _FP8_FUSED_M_MAX and N <= _FP8_FUSED_N_MAX)
    )
    if fused:
        if mode == 0:
            block_m = 16 if M <= 32 else 64
            cfg = (block_m, 512, 64)
            warps = 8
        else:
            cfg = (64, 256, 32)
            warps = 4
        block_m, block_n, block_k = cfg
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        _scaled_mm_gemm_fused[grid](
            a_src,
            b_src,
            out,
            scale_a_c,
            scale_b_c,
            bias_arg,
            M,
            N,
            K,
            K,
            N,
            N,
            bias_s0,
            bias_s1,
            HAS_BIAS=has_bias,
            SA_BCAST=sa_bcast,
            SB_BCAST=sb_bcast,
            EVEN_K=even_k,
            MODE=mode,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
        )
        return out

    a16 = torch.empty((M, K), device=input.device, dtype=torch.float16)
    b16 = torch.empty((K, N), device=weight.device, dtype=torch.float16)
    numel_a = a_src.numel()
    numel_b = b_src.numel()
    _decode_kernel[(triton.cdiv(numel_a, _DECODE_BLOCK),)](
        a_src,
        a16,
        numel_a,
        BLOCK=_DECODE_BLOCK,
        MODE=mode,
        num_warps=_DECODE_WARPS,
    )
    _decode_kernel[(triton.cdiv(numel_b, _DECODE_BLOCK),)](
        b_src,
        b16,
        numel_b,
        BLOCK=_DECODE_BLOCK,
        MODE=mode,
        num_warps=_DECODE_WARPS,
    )

    if M <= 32:
        cfg = (16, 128, 256, 1, 4, 3)
    elif M <= 64:
        cfg = (64, 256, 128, 1, 8, 3)
    elif M <= 256:
        cfg = (128, 256, 128, 8, 8, 3)
    else:
        cfg = (128, 256, 128, 16, 8, 2)
    _launch_gemm(
        a16,
        b16,
        out,
        scale_a_c,
        scale_b_c,
        bias_arg,
        bias_s0,
        bias_s1,
        has_bias,
        sa_bcast,
        sb_bcast,
        M,
        N,
        K,
        even_k,
        cfg,
    )
    return out


__all__ = ["triton_scaled_mm"]
