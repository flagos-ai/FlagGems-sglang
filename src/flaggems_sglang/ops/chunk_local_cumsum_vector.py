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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Generic per-chunk cumulative sum of a vector gate (Triton).

Computes the within-chunk cumulative sum of ``g`` of shape
``[B, T, H, S]`` along the time axis, where ``T`` is split into chunks
of ``chunk_size`` (a power of two, ``T % chunk_size == 0``):

    out = g.float().view(B, T // BT, BT, H, S).cumsum(dim=2)
          .reshape(B, T, H, S)

with optional ``reverse`` (cumsum from the chunk end) and ``scale``.
All math is done in fp32; output is always float32, matching the
reference. This is the chip-agnostic generic fallback; vendor tiers may
override it with backend-specialized kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_local_cumsum_fwd_nomask_kernel(
    g_ptr,
    out_ptr,
    BT: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
):
    """Forward cumsum (reverse=False, scale=None) for unmasked power-of-2 S.

    Grid: (B * NT * H,).  One program handles one (batch*chunk, head) tile of
    shape [BT, S].  For power-of-2 S <= 128, BLOCK_S == S and the s-mask is
    statically true, so the whole geometry derives from the three constexpr
    scalars BT / S / H and the launcher carries just 5 kernel args — the
    minimum this backend accepts (a 2D grid traps with an ATU fault, so the
    pid is linearized and decomposed with constexpr div/mod here).

    g is [B, T, H, S] contiguous: element (b, t, h, s) is at (b*T+t)*H*S + h*S
    + s.  In chunk layout bn = b*NT + n covers t in [bn*BT, (bn+1)*BT).
    base: pointer to (bn*BT, h, 0) == bn*BT*H*S + h*S.
    """
    pid = tl.program_id(0)
    bn = pid // H
    h = pid % H

    offs_t = tl.arange(0, BT)[:, None]  # [BT, 1]
    offs_s = tl.arange(0, S)[None, :]  # [1, S]

    base = bn * (BT * H * S) + h * S
    offs = base + offs_t * (H * S) + offs_s

    x = tl.load(g_ptr + offs, cache_modifier=".cv").to(tl.float32)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, y, cache_modifier=".wt")


@triton.jit
def _chunk_local_cumsum_kernel(
    g_ptr,
    out_ptr,
    scale,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """General kernel: masked S (non-pow2 or >128), reverse, and/or scale."""
    pid_bnh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_bn = pid_bnh // H
    pid_h = pid_bnh % H

    offs_t = tl.arange(0, BT)[:, None]
    if REVERSE:
        # Load reversed, cumsum forward, store reversed => cumsum from the end.
        offs_t = BT - 1 - offs_t

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]

    base = pid_bn * (BT * H * S) + pid_h * S
    offs = base + offs_t * (H * S) + offs_s

    if NEED_MASK:
        mask_s = offs_s < S
        x = tl.load(
            g_ptr + offs, mask=mask_s, other=0.0, cache_modifier=".cv"
        ).to(tl.float32)
    else:
        x = tl.load(g_ptr + offs, cache_modifier=".cv").to(tl.float32)

    y = tl.cumsum(x, axis=0)
    if HAS_SCALE:
        y = y * scale

    if NEED_MASK:
        mask_s = offs_s < S
        tl.store(out_ptr + offs, y, mask=mask_s, cache_modifier=".wt")
    else:
        tl.store(out_ptr + offs, y, cache_modifier=".wt")


def _block_s(S):
    """BLOCK_S = next power of two >= S, capped at 128; (needs_mask, BLOCK_S)."""
    bs = 1 << (S - 1).bit_length()
    if bs > 128:
        bs = 128
    return bs > S, bs


def chunk_local_cumsum_vector(g, chunk_size, reverse=False, scale=None):
    """Per-chunk (within-chunk) cumulative sum of a vector per-token, per-head gate.

    Args:
        g: [B, T, H, S] input gate tensor (float32 / bfloat16 / float16).
        chunk_size: within-chunk width (power of 2), T must be a multiple of it.
        reverse: if True, cumsum runs from the chunk end toward its start.
        scale: optional scalar multiplier applied to the output (or None).

    Returns:
        [B, T, H, S] float32 tensor.
    """
    B, T, H, S = g.shape
    BT = chunk_size
    out = g.new_empty((B, T, H, S), dtype=torch.float32)

    # Fast path: unmasked power-of-2 S <= 128, forward, no scale.  The masked
    # check is inlined (S power of 2 and <= 128) so the hot call does not go
    # through _block_s; constexprs are passed positionally to keep the JIT
    # dispatch argument path minimal.
    if not reverse and scale is None and S <= 128 and (S & (S - 1)) == 0:
        _chunk_local_cumsum_fwd_nomask_kernel[(B * (T // BT) * H,)](
            g, out, BT, S, H, num_warps=4
        )
    else:
        need_mask, bs = _block_s(S)
        _chunk_local_cumsum_kernel[(B * (T // BT) * H, (S + bs - 1) // bs)](
            g,
            out,
            1.0 if scale is None else scale,
            H,
            S,
            BT,
            bs,
            need_mask,
            reverse,
            scale is not None,
            num_warps=4,
        )
    return out


__all__ = ["chunk_local_cumsum_vector"]
