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

"""chunk_local_cumsum_vector -- per-chunk cumulative sum of a vector
per-token, per-head gate.

FLA-family building block: turns raw log-decay gate values ``g`` into
within-chunk cumulative decays for gated linear attention.

Semantics (matches the PyTorch reference exactly):

    B, T, H, S = g.shape
    g_c = g.float().view(B, T // chunk_size, chunk_size, H, S)
    if reverse: g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None: out = out * scale
    if reverse: out = out.flip(2)
    return out.reshape(B, T, H, S)

Scope:
    - head_first=False (g is [B, T, H, S], contiguous)
    - fixed-size batching (no cu_seqlens)
    - T is always an exact multiple of chunk_size (chunk_size is a power of 2)
    - output dtype is always float32 (accumulation done in float32)

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no torch.cumsum, no flag_gems,
no ``_compiled`` handles) — the whole computation is done inside the Triton
kernel so it is portable across supported chips.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reference_fwd_kernel(
    g_ptr,
    out_ptr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Forward cumsum (reverse=False, no scale) — the hot path.

    Grid: (B * NT * H, ceil(S / BLOCK_S)).
    Each program handles one (batch*chunk, head, s-block) tile of shape [BT, BLOCK_S].
    """
    pid_bnh = tl.program_id(0)  # linearized (B*NT, H)
    pid_s = tl.program_id(1)  # s-block index

    # Decompose linear index back to (batch*chunk, head).
    pid_bn = pid_bnh // H
    pid_h = pid_bnh % H

    offs_t = tl.arange(0, BT)[:, None]  # [BT, 1]
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]  # [1, BLOCK_S]
    mask_s = offs_s < S

    # g is [B, T, H, S] contiguous: element (b, t, h, s) is at (b*T+t)*H*S + h*S + s.
    # In chunk layout bn = b*NT + n covers t in [bn*BT, (bn+1)*BT).
    # base: pointer to (bn*BT, h, 0) == bn*BT*H*S + h*S.
    base = pid_bn * (BT * H * S) + pid_h * S

    x = tl.load(
        g_ptr + base + offs_t * (H * S) + offs_s,
        mask=mask_s,
        other=0.0,
    ).to(tl.float32)

    y = tl.cumsum(x, axis=0)

    tl.store(
        out_ptr + base + offs_t * (H * S) + offs_s,
        y,
        mask=mask_s,
    )


@triton.jit
def _reference_kernel(
    g_ptr,
    out_ptr,
    scale,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """General kernel: handles reverse and/or scale."""
    pid_bnh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_bn = pid_bnh // H
    pid_h = pid_bnh % H

    offs_t = tl.arange(0, BT)[:, None]
    if REVERSE:
        # Load reversed, cumsum forward, store reversed => cumsum from the end.
        offs_t = BT - 1 - offs_t

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    mask_s = offs_s < S

    base = pid_bn * (BT * H * S) + pid_h * S

    x = tl.load(
        g_ptr + base + offs_t * (H * S) + offs_s,
        mask=mask_s,
        other=0.0,
    ).to(tl.float32)

    y = tl.cumsum(x, axis=0)
    if HAS_SCALE:
        y = y * scale

    tl.store(
        out_ptr + base + offs_t * (H * S) + offs_s,
        y,
        mask=mask_s,
    )


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
    NT = T // BT

    # next_power_of_2(S) capped at 128.
    block_s = 1 << (S - 1).bit_length()
    if block_s > 128:
        block_s = 128

    out = torch.empty_like(g, dtype=torch.float32)

    grid = (B * NT * H, (S + block_s - 1) // block_s)

    if not reverse and scale is None:
        _reference_fwd_kernel[grid](
            g,
            out,
            H=H,
            S=S,
            BT=BT,
            BLOCK_S=block_s,
            num_warps=4,
        )
    else:
        _reference_kernel[grid](
            g,
            out,
            1.0 if scale is None else scale,
            H=H,
            S=S,
            BT=BT,
            BLOCK_S=block_s,
            REVERSE=reverse,
            HAS_SCALE=scale is not None,
            num_warps=4,
        )
    return out


__all__ = ["chunk_local_cumsum_vector"]
