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

DIM_NOPE = 512

DIM_ROPE = 64

WARPS = 2

_DIM_NOPE = 512

_DIM_ROPE = 64


@triton.jit
def _decode_S(byte):

    code = byte.to(tl.int32) & 0xFF
    payload = code & 0x7F

    normal = ((payload << 20) + (120 << 23)).to(tl.float32, bitcast=True)

    magnitude = tl.where(
        payload < 8, payload.to(tl.float32) * 0.001953125, normal
    )

    magnitude = tl.where(payload == 0x7F, float("nan"), magnitude)
    return tl.where(code >= 128, -magnitude, magnitude)


@triton.jit
def _dequantize_S(OUT, ROW, ROW_S0: tl.constexpr, GROUP: tl.constexpr):

    token = tl.program_id(0)
    column = tl.arange(0, 512)
    byte = tl.load(ROW + token * ROW_S0 + column)
    scale = tl.load(
        ROW.to(tl.pointer_type(tl.float32))
        + token * (ROW_S0 // 4)
        + 128
        + column // GROUP
    )
    tl.store(
        OUT + token * 576 + column, (_decode_S(byte) * scale).to(tl.bfloat16)
    )

    tail = tl.arange(0, 64)
    rope = tl.load(
        ROW.to(tl.pointer_type(tl.bfloat16))
        + token * (ROW_S0 // 2)
        + (512 + (512 // GROUP) * 4) // 2
        + tail
    )
    tl.store(OUT + token * 576 + 512 + tail, rope)


def dequantize_k_cache_fast_small(quant_k_cache, group_size=128):

    rows = quant_k_cache.shape[0]
    out = torch.empty(
        rows,
        DIM_NOPE + DIM_ROPE,
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if rows:
        _dequantize_S[(rows,)](
            out,
            triton.reinterpret(quant_k_cache, tl.int8),
            quant_k_cache.stride(0),
            group_size,
            num_warps=WARPS,
        )
    return out


@triton.jit
def _dequant_kernel_L(
    q_ptr,
    out_ptr,
    n_tokens,
    stride_q,
    stride_o,
    NUM_TILES: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EVEN_T: tl.constexpr,
    WIDE: tl.constexpr,
    PACKED: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        rows = pid.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    else:
        rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t = tl.arange(0, NUM_TILES)
    g = tl.arange(0, GROUP_SIZE)
    offs = t[None, :, None] * GROUP_SIZE + g[None, None, :]
    if EVEN_T:
        c = tl.load(q_ptr + rows[:, None, None] * stride_q + offs)
        if PACKED:
            s_ptr = q_ptr.to(tl.pointer_type(tl.float32))
            s = tl.load(
                s_ptr
                + rows[:, None] * (stride_q // 4)
                + (512 // 4)
                + t[None, :]
            )
            b_ptr = q_ptr.to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(
                b_ptr
                + rows[:, None] * (stride_q // 2)
                + ((512 + NUM_TILES * 4) // 2 + tl.arange(0, 64))[None, :]
            )
        else:
            lanes = tl.arange(0, 4)
            sb = tl.load(
                q_ptr
                + rows[:, None, None] * stride_q
                + (512 + t[None, :, None] * 4 + lanes[None, None, :])
            )
            su = sb.to(tl.uint8, bitcast=True).to(tl.uint32)
            s_word = tl.sum(su * (1 << (lanes[None, None, :] * 8)), axis=2)
            s = s_word.to(tl.float32, bitcast=True)
            rlanes = tl.arange(0, 2)
            rb = tl.load(
                q_ptr
                + rows[:, None, None] * stride_q
                + (
                    512
                    + NUM_TILES * 4
                    + tl.arange(0, 64)[None, :, None] * 2
                    + rlanes[None, None, :]
                )
            )
            ru = rb.to(tl.uint8, bitcast=True).to(tl.uint32)
            r_word = tl.sum(ru * (1 << (rlanes[None, None, :] * 8)), axis=2)
            rope = r_word.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        rmask = (rows < n_tokens)[:, None, None]
        c = tl.load(
            q_ptr + rows[:, None, None] * stride_q + offs,
            mask=rmask,
            other=0.0,
        )
        if PACKED:
            s_ptr = q_ptr.to(tl.pointer_type(tl.float32))
            s = tl.load(
                s_ptr
                + rows[:, None] * (stride_q // 4)
                + (512 // 4)
                + t[None, :],
                mask=(rows < n_tokens)[:, None],
                other=0.0,
            )
            b_ptr = q_ptr.to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(
                b_ptr
                + rows[:, None] * (stride_q // 2)
                + ((512 + NUM_TILES * 4) // 2 + tl.arange(0, 64))[None, :],
                mask=(rows < n_tokens)[:, None],
                other=0.0,
            )
        else:
            lanes = tl.arange(0, 4)
            sb = tl.load(
                q_ptr
                + rows[:, None, None] * stride_q
                + (512 + t[None, :, None] * 4 + lanes[None, None, :]),
                mask=rmask,
                other=0.0,
            )
            su = sb.to(tl.uint8, bitcast=True).to(tl.uint32)
            s_word = tl.sum(su * (1 << (lanes[None, None, :] * 8)), axis=2)
            s = s_word.to(tl.float32, bitcast=True)
            rlanes = tl.arange(0, 2)
            rb = tl.load(
                q_ptr
                + rows[:, None, None] * stride_q
                + (
                    512
                    + NUM_TILES * 4
                    + tl.arange(0, 64)[None, :, None] * 2
                    + rlanes[None, None, :]
                ),
                mask=rmask,
                other=0.0,
            )
            ru = rb.to(tl.uint8, bitcast=True).to(tl.uint32)
            r_word = tl.sum(ru * (1 << (rlanes[None, None, :] * 8)), axis=2)
            rope = r_word.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    u8 = c.to(tl.uint8, bitcast=True).to(tl.uint32)
    bits = ((u8 & 0x7F) << 20) | ((u8 & 0x80) << 24)
    nanb = (bits | 0x7FC00000).to(tl.float32, bitcast=True)
    bitsf = bits.to(tl.float32, bitcast=True)
    xf = tl.where((u8 & 0x7F) == 0x7F, nanb, bitsf)

    sf = s * 1.3292279957849159e36
    prod = xf * sf[:, :, None]

    pu = prod.to(tl.uint32, bitcast=True)
    y = (pu >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)

    if EVEN_T:
        tl.store(out_ptr + rows[:, None, None] * stride_o + offs, y)
        tl.store(
            out_ptr
            + rows[:, None] * stride_o
            + (512 + tl.arange(0, 64))[None, :],
            rope,
        )
    else:
        omask = (rows < n_tokens)[:, None, None]
        tl.store(
            out_ptr + rows[:, None, None] * stride_o + offs, y, mask=omask
        )
        tl.store(
            out_ptr
            + rows[:, None] * stride_o
            + (512 + tl.arange(0, 64))[None, :],
            rope,
            mask=(rows < n_tokens)[:, None],
        )


def dequantize_k_cache_fast_large(quant_k_cache, group_size=128):
    num_tokens = quant_k_cache.shape[0]
    out = torch.empty(
        (num_tokens, _DIM_NOPE + _DIM_ROPE),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if num_tokens == 0:
        return out

    num_tiles = _DIM_NOPE // group_size
    stride_q = quant_k_cache.stride(0)
    wide = num_tokens * stride_q >= 2**31
    packed = (stride_q % 4 == 0) and (quant_k_cache.stride(1) == 1)
    if num_tokens >= 8192:
        block_t, num_warps = 8, 2
    elif num_tokens >= 512:
        block_t, num_warps = 4, 4
    else:
        block_t, num_warps = 2, 2
    grid = (triton.cdiv(num_tokens, block_t),)
    _dequant_kernel_L[grid](
        quant_k_cache,
        out,
        num_tokens,
        stride_q,
        out.stride(0),
        NUM_TILES=num_tiles,
        GROUP_SIZE=group_size,
        BLOCK_T=block_t,
        EVEN_T=(num_tokens % block_t == 0),
        WIDE=wide,
        PACKED=packed,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    if quant_k_cache.shape[0] <= 128:
        return dequantize_k_cache_fast_small(quant_k_cache, group_size)
    return dequantize_k_cache_fast_large(quant_k_cache, group_size)


__all__ = ["dequantize_k_cache_fast"]
