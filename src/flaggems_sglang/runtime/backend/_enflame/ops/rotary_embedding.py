# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Triton implementation of diffusion/rotary_embedding (GCU-adapted).

Semantics (mirrors the reference): x is rotated pair-wise on adjacent
lanes -- out[..., 2p] = x[..., 2p] * cos[t, p] - x[..., 2p+1] * sin[t, p],
out[..., 2p+1] = x[..., 2p] * sin[t, p] + x[..., 2p+1] * cos[t, p], with
arithmetic in float32 and one final rounding back to the input dtype.
The ``interleaved`` flag is ignored by the reference ("half-width form"),
so it is ignored here as well.

GCU adaptation notes (2026-11-08/12 toolchain, do_bench kernel-mode median,
GCU400/L600):
* The historical 3D/2D-tile kernels with value-level cos/sin broadcast
  (``x1 * cos[:, None, :]``) materialize the broadcast operand into a
  whole [BT,H,HALF] fp32 DSM tile per FMA -- +230us of the old 415us at
  T=32768/H=16/D=128 (the shift/rotate/pack math chain itself is only
  ~+35us over the 126us copy floor).  Any arithmetic broadcast is banned
  from the fast path.  bf16 pairs travel as one int32 word, unpacked and
  repacked with shift/mask bitcasts (bit-exact vs the float32 reference
  math); index arithmetic uses shift/AND only -- ``e - (e // H) * H``
  div/mod forms silently miscompile on this toolchain.
* Large-T arm ``_rotary_gcu_v5`` (T >= 6144): pair-chain head loop over
  2D [BT,HALF] tiles; cos/sin are loaded once per program and consumed
  elementwise (zero broadcast); each iteration issues BOTH head-chain
  loads before any math of the pair (load-pairing ILP; odd head counts
  emit only the a-chain on the trailing iteration).  Issuing the
  b-chain load after the a-chain math+store costs +7-9% (187 -> 174us at
  T=32768); pre-loading 4 tiles instead (peel2, ~384KB DSM/program)
  collapses throughput instead (+53%) -- 2 tiles ahead is the sweet spot.
  bt=256/nw2 re-swept optimal after the restructure (nw1 +9%, nw4 +89%,
  bt128 +32%).  T=32768 174us (1750 GB/s, 1.08x the copy+math floor),
  T=8192 56.4us.
* Small-T arm ``_rotary_gcu_v6`` (T < 6144): one-shot 3D [BT,H,HALF]
  tile with cos/sin expanded through stride-0 3D loads (load-side
  expansion, cheaper than an explicit broadcast op at these sizes).
  nw2 beats nw1 at every size; bt=16 for T <= 512 (more programs hide
  prologue latency: 8.0 vs 9.6us at T=64), bt=32 above (13.8 vs 18.1us
  at T=1024).  A no-mask twin ``_rotary_gcu_v6_exact`` is dispatched
  when H, half and T are all exactly tile-shaped (-2.3% at T=1024).
* Everything else (other dtypes, non-contiguous inputs, non-power-of-two
  half) falls back to a strided 1D row-loop kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_gcu_v5(
    w_ptr,  # int32 view of bf16 x: (T, H, HALF)
    cos_ptr,  # (T, HALF) bf16
    sin_ptr,  # (T, HALF) bf16
    o_ptr,  # int32 view of bf16 out: (T, H, HALF)
    NUM_TOKENS,
    STRIDE_W_S: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAD_HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    ts = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    pair = tl.arange(0, PAD_HALF)
    t_ok = ts < NUM_TOKENS
    p_ok = pair < HALF
    cs_mask = t_ok[:, None] & p_ok[None, :]
    cos = tl.load(
        cos_ptr + ts[:, None] * STRIDE_COS_S + pair[None, :] * STRIDE_COS_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + ts[:, None] * STRIDE_SIN_S + pair[None, :] * STRIDE_SIN_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)
    for hh in tl.static_range(0, NUM_HEADS, 2):
        off_a = ts[:, None] * STRIDE_W_S + hh * STRIDE_W_H + pair[None, :]
        wa = tl.load(w_ptr + off_a, mask=cs_mask, other=0)
        if hh + 1 < NUM_HEADS:
            off_b = (
                ts[:, None] * STRIDE_W_S
                + (hh + 1) * STRIDE_W_H
                + pair[None, :]
            )
            wb = tl.load(w_ptr + off_b, mask=cs_mask, other=0)
        a1 = (wa << 16).to(tl.float32, bitcast=True)
        a2 = (wa & -65536).to(tl.float32, bitcast=True)
        oa1 = a1 * cos - a2 * sin
        oa2 = a1 * sin + a2 * cos
        pa1 = oa1.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
        pa2 = oa2.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
        tl.store(o_ptr + off_a, (pa2 << 16) | pa1, mask=cs_mask)
        if hh + 1 < NUM_HEADS:
            b1 = (wb << 16).to(tl.float32, bitcast=True)
            b2 = (wb & -65536).to(tl.float32, bitcast=True)
            ob1 = b1 * cos - b2 * sin
            ob2 = b1 * sin + b2 * cos
            pb1 = ob1.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
            pb2 = ob2.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(o_ptr + off_b, (pb2 << 16) | pb1, mask=cs_mask)


@triton.jit
def _rotary_gcu_v6(
    w_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    NUM_TOKENS: tl.constexpr,
    STRIDE_W_S: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    ts = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    hs = tl.arange(0, PAD_H)
    pair = tl.arange(0, PAD_HALF)
    t_ok = ts < NUM_TOKENS
    h_ok = hs < NUM_HEADS
    p_ok = pair < HALF
    mask3d = t_ok[:, None, None] & h_ok[None, :, None] & p_ok[None, None, :]
    cos = tl.load(
        cos_ptr
        + ts[:, None, None] * STRIDE_COS_S
        + 0 * hs[None, :, None]
        + pair[None, None, :] * STRIDE_COS_D,
        mask=mask3d,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr
        + ts[:, None, None] * STRIDE_SIN_S
        + 0 * hs[None, :, None]
        + pair[None, None, :] * STRIDE_SIN_D,
        mask=mask3d,
        other=0.0,
    ).to(tl.float32)
    w_offs = (
        ts[:, None, None] * STRIDE_W_S
        + hs[None, :, None] * STRIDE_W_H
        + pair[None, None, :]
    )
    w = tl.load(w_ptr + w_offs, mask=mask3d, other=0)
    x1 = (w << 16).to(tl.float32, bitcast=True)
    x2 = (w & -65536).to(tl.float32, bitcast=True)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    b1 = o1.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
    b2 = o2.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
    tl.store(o_ptr + w_offs, (b2 << 16) | b1, mask=mask3d)


@triton.jit
def _rotary_gcu_v6_exact(
    w_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    NUM_TOKENS: tl.constexpr,
    STRIDE_W_S: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # mask-free twin of _rotary_gcu_v6: host guarantees PAD_H == NUM_HEADS,
    # PAD_HALF == HALF and NUM_TOKENS % BLOCK_T == 0.
    ts = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    hs = tl.arange(0, PAD_H)
    pair = tl.arange(0, PAD_HALF)
    cos = tl.load(
        cos_ptr
        + ts[:, None, None] * STRIDE_COS_S
        + 0 * hs[None, :, None]
        + pair[None, None, :] * STRIDE_COS_D,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr
        + ts[:, None, None] * STRIDE_SIN_S
        + 0 * hs[None, :, None]
        + pair[None, None, :] * STRIDE_SIN_D,
    ).to(tl.float32)
    w_offs = (
        ts[:, None, None] * STRIDE_W_S
        + hs[None, :, None] * STRIDE_W_H
        + pair[None, None, :]
    )
    w = tl.load(w_ptr + w_offs)
    x1 = (w << 16).to(tl.float32, bitcast=True)
    x2 = (w & -65536).to(tl.float32, bitcast=True)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    b1 = o1.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
    b2 = o2.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
    tl.store(o_ptr + w_offs, (b2 << 16) | b1)


@triton.jit
def _rotary_embedding_strided_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    T,
    x_stride_s,
    x_stride_h,
    x_stride_d,
    cos_stride_s,
    sin_stride_s,
    out_stride_s,
    out_stride_h,
    out_stride_d,
    NUM_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HALF_B: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = tl.arange(0, HALF_B)
    lm = pair < HALF
    for tok in range(pid, T, 65535):
        c = tl.load(
            cos_ptr + tok * cos_stride_s + pair, mask=lm, other=0.0
        ).to(tl.float32)
        s = tl.load(
            sin_ptr + tok * sin_stride_s + pair, mask=lm, other=0.0
        ).to(tl.float32)
        x_row = x_ptr + tok * x_stride_s
        o_row = out_ptr + tok * out_stride_s
        for h in range(0, NUM_HEADS):
            base = h * x_stride_h
            even = base + 2 * pair * x_stride_d
            odd = even + x_stride_d
            x1 = tl.load(x_row + even, mask=lm, other=0.0).to(tl.float32)
            x2 = tl.load(x_row + odd, mask=lm, other=0.0).to(tl.float32)
            o1 = x1 * c - x2 * s
            o2 = x2 * c + x1 * s
            obase = h * out_stride_h
            tl.store(
                o_row + obase + 2 * pair * out_stride_d,
                o1.to(out_ptr.dtype.element_ty),
                mask=lm,
            )
            tl.store(
                o_row + obase + (2 * pair + 1) * out_stride_d,
                o2.to(out_ptr.dtype.element_ty),
                mask=lm,
            )


_PAIR_T_MIN = 6144
_PAIR_BT_ELEMS = 16384
_3D_TILE_ELEMS = 32768
_SMALL_T_MAX = 512


def rotary_embedding(x, cos, sin, interleaved):
    num_tokens, num_heads, head_size = x.shape
    half = head_size // 2
    out = torch.empty_like(x)
    if num_tokens == 0:
        return out
    if (
        x.dtype == torch.bfloat16
        and head_size % 2 == 0
        and half & (half - 1) == 0
        and x.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
    ):
        w = x.view(torch.int32)
        wo = out.view(torch.int32)
        if num_tokens >= _PAIR_T_MIN:
            block_t = _PAIR_BT_ELEMS // (1 << (half - 1).bit_length())
            if block_t < 1:
                block_t = 1
            elif block_t > 256:
                block_t = 256
            grid = ((num_tokens + block_t - 1) // block_t,)
            while grid[0] > 65535:
                block_t *= 2
                grid = ((num_tokens + block_t - 1) // block_t,)
            _rotary_gcu_v5[grid](
                w,
                cos,
                sin,
                wo,
                num_tokens,
                w.stride(0),
                w.stride(1),
                cos.stride(0),
                cos.stride(1),
                sin.stride(0),
                sin.stride(1),
                num_heads,
                half,
                1 << (half - 1).bit_length(),
                block_t,
                num_warps=2,
                num_stages=1,
            )
            return out
        pad_h = triton.next_power_of_2(num_heads)
        pad_half = 1 << (half - 1).bit_length()
        if pad_half < 8:
            pad_half = 8
        block_t = _3D_TILE_ELEMS // (pad_h * pad_half)
        if block_t < 1:
            block_t = 1
        elif block_t > 32:
            block_t = 32
        if num_tokens <= _SMALL_T_MAX and block_t > 16:
            block_t = 16
        grid = ((num_tokens + block_t - 1) // block_t,)
        if (
            pad_h == num_heads
            and pad_half == half
            and num_tokens % block_t == 0
        ):
            _rotary_gcu_v6_exact[grid](
                w,
                cos,
                sin,
                wo,
                num_tokens,
                w.stride(0),
                w.stride(1),
                cos.stride(0),
                cos.stride(1),
                sin.stride(0),
                sin.stride(1),
                num_heads,
                half,
                pad_h,
                pad_half,
                block_t,
                num_warps=2,
            )
            return out
        _rotary_gcu_v6[grid](
            w,
            cos,
            sin,
            wo,
            num_tokens,
            w.stride(0),
            w.stride(1),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            num_heads,
            half,
            pad_h,
            pad_half,
            block_t,
            num_warps=2,
        )
        return out
    half_b = max(triton.next_power_of_2(half), 2)
    grid = (min(num_tokens, 65535),)
    _rotary_embedding_strided_kernel[grid](
        x,
        cos,
        sin,
        out,
        num_tokens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        sin.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS=num_heads,
        HALF=half,
        HALF_B=half_b,
        num_warps=4,
    )
    return out


__all__ = ["rotary_embedding"]
