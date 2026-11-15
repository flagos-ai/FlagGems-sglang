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
def _fp8_byte_S(x):
    u = x.to(tl.uint32)
    magnitude = u & 127
    bits = ((u & 128) << 24) | (magnitude << 20) | 0
    normal = (bits + (120 << 23)).to(tl.float32, bitcast=True)
    subnormal = tl.fma(tl.abs(normal), 2.0, -0.015625)
    subnormal = (subnormal.to(tl.uint32, bitcast=True) | ((u & 128) << 24)).to(
        tl.float32, bitcast=True
    )
    value = tl.where(magnitude < 8, subnormal, normal)
    return tl.where(magnitude == 127, float("nan"), value)


@triton.jit
def _fast_S(Q, output_ptr, QS: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.uint8))
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, 512)
    r = tl.arange(0, 64)
    base = Q + row * QS
    rope = tl.load((base + 528 + 2 * r).to(tl.pointer_type(tl.uint16)))
    value = _fp8_byte_S(tl.load(base + c))
    scale = tl.load(
        (base + 512 + (c // 128) * 4).to(tl.pointer_type(tl.float32))
    )
    tl.store(output_ptr + row * 576 + c, value * scale)
    tl.store(
        (output_ptr + row * 576 + 512 + r).to(tl.pointer_type(tl.uint16)), rope
    )


def dequantize_k_cache_fast_small(quant_k_cache, group_size=128):
    n = quant_k_cache.shape[0]
    out = torch.empty(
        (n, 576), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    if n:
        _fast_S[(n,)](
            quant_k_cache.view(torch.uint8),
            out,
            quant_k_cache.stride(0),
            num_warps=4,
        )
    return out


@triton.jit
def _decode_L(q):
    q = q.to(tl.uint16)
    bits = (q & 127) << 7 | (q & 128) << 8
    value = bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where(q & 127 == 127, float("nan"), value)


@triton.jit
def _kernel_L(X, Y, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    X = X.to(tl.pointer_type(tl.uint8))
    groups = tl.program_id(0) * B + tl.arange(0, B)
    rows = groups // 9
    parts = groups % 9
    live = groups < N * 9
    c = tl.arange(0, 32)
    partq = parts % 8
    base = X + rows[:, None] * S
    q = tl.load(
        base.to(tl.pointer_type(tl.uint16)) + partq[:, None] * 32 + c[None, :],
        live[:, None],
        other=0,
        cache_modifier=".cg",
    ).to(tl.uint32)
    s = tl.load(
        (base + 512).to(tl.pointer_type(tl.float32)) + partq[:, None] // 2,
        live[:, None],
        other=0,
        cache_modifier=".cg",
    )
    lo = (
        (_decode_L(q & 255) * s)
        .to(tl.bfloat16)
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    hi = (
        (_decode_L(q >> 8) * s)
        .to(tl.bfloat16)
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    decoded = lo | hi << 16
    rope = tl.load(
        (base + 528).to(tl.pointer_type(tl.uint32)) + c[None, :],
        live[:, None] & (parts[:, None] == 8),
        other=0,
        cache_modifier=".cg",
    )
    result = tl.where(parts[:, None] < 8, decoded, rope)
    tl.store(
        Y.to(tl.pointer_type(tl.uint32)) + groups[:, None] * 32 + c[None, :],
        result,
        live[:, None],
        cache_modifier=".cs",
    )


@triton.jit
def _split_kernel_L(
    X, Y, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr, RR: tl.constexpr
):
    pid = tl.program_id(0)
    quant_blocks: tl.constexpr = triton.cdiv(N * 8, B)
    c = tl.arange(0, 32)
    if pid < quant_blocks:
        groups = pid * B + tl.arange(0, B)
        rows = groups // 8
        parts = groups % 8
        live = rows < N
        q = tl.load(
            X.to(tl.pointer_type(tl.uint16))
            + rows[:, None] * (S // 2)
            + parts[:, None] * 32
            + c[None, :],
            live[:, None],
            other=0,
            cache_modifier=".cg",
        ).to(tl.uint32)
        scales = tl.load(
            X.to(tl.pointer_type(tl.float32))
            + rows[:, None] * (S // 4)
            + 128
            + parts[:, None] // 2,
            live[:, None],
            other=0,
            cache_modifier=".cg",
        )
        lo = (
            (_decode_L(q & 255) * scales)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        hi = (
            (_decode_L(q >> 8) * scales)
            .to(tl.bfloat16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        tl.store(
            Y.to(tl.pointer_type(tl.uint32))
            + rows[:, None] * 288
            + parts[:, None] * 32
            + c[None, :],
            lo | hi << 16,
            live[:, None],
            cache_modifier=".cs",
        )
    else:
        rows = (pid - quant_blocks) * RR + tl.arange(0, RR)
        rope = tl.load(
            X.to(tl.pointer_type(tl.uint32))
            + rows[:, None] * (S // 4)
            + 132
            + c[None, :],
            rows[:, None] < N,
            other=0,
            cache_modifier=".cg",
        )
        tl.store(
            Y.to(tl.pointer_type(tl.uint32))
            + rows[:, None] * 288
            + 256
            + c[None, :],
            rope,
            rows[:, None] < N,
            cache_modifier=".cs",
        )


def dequantize_k_cache_fast_large(quant_k_cache, group_size=128):
    n = quant_k_cache.shape[0]
    output = quant_k_cache.new_empty((n, 576), dtype=torch.bfloat16)
    if n:
        if n < 4096:
            _split_kernel_L.run(
                quant_k_cache.view(torch.uint8),
                output,
                n,
                quant_k_cache.stride(0),
                32,
                32,
                grid=(triton.cdiv(n * 8, 32) + triton.cdiv(n, 32),),
                warmup=False,
                num_warps=8,
            )
        else:
            _kernel_L.run(
                quant_k_cache.view(torch.uint8),
                output,
                n,
                quant_k_cache.stride(0),
                32,
                grid=(triton.cdiv(n * 9, 32),),
                warmup=False,
                num_warps=8,
            )
    return output


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    if quant_k_cache.shape[0] <= 1:
        return dequantize_k_cache_fast_small(quant_k_cache, group_size)
    return dequantize_k_cache_fast_large(quant_k_cache, group_size)


__all__ = ["dequantize_k_cache_fast"]
