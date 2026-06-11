"""MRotaryEmbedding Triton kernel for sglang.

Persistent kernel with capped grid (max 1024 programs). Each program loops
over ceil(N / grid_size) tokens, reducing per-block launch overhead and
eliminating wave serialization on large prefill sequences. num_warps=8
for all cases with 256 threads for 512-element tile (2 elem/thread),
optimized for HBM latency hiding.

Pure Triton, in-place on q and k. Matches sglang's MRotaryEmbedding
dispatch: 2D positions with mrope_section -> triton_mrope_fused,
otherwise -> 1D RoPE.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _mrope_fused_fwd_persistent(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0,
    k_stride0,
    pos_stride0,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rd: tl.constexpr,
    section_t: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_interleaved_glm: tl.constexpr,
    is_neox_style: tl.constexpr,
    N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    axis_map_ptr,
):
    """Persistent kernel: each program loops over ceil(N/GRID_SIZE) tokens."""
    pid = tl.program_id(0)
    half_rd: tl.constexpr = rd // 2
    pad_rd_half: tl.constexpr = pad_rd // 2

    pair_idx = tl.arange(0, pad_rd_half)
    valid_pair = pair_idx < half_rd

    if is_neox_style:
        even_off = pair_idx
        odd_off = pair_idx + half_rd
    else:
        even_off = pair_idx * 2
        odd_off = even_off + 1

    for token_idx in range(pid, N, GRID_SIZE):
        q_row = q_ptr + token_idx * q_stride0
        k_row = k_ptr + token_idx * k_stride0

        t_pos = tl.load(positions_ptr + 0 * pos_stride0 + token_idx)
        h_pos = tl.load(positions_ptr + 1 * pos_stride0 + token_idx)
        w_pos = tl.load(positions_ptr + 2 * pos_stride0 + token_idx)

        if is_interleaved:
            if is_interleaved_glm:
                axes = tl.load(
                    axis_map_ptr + pair_idx, mask=valid_pair, other=0
                )
                pos = tl.where(
                    axes == 1, h_pos, tl.where(axes == 2, w_pos, t_pos)
                )
            else:
                mod3 = pair_idx % 3
                is_h = (mod3 == 1) & (pair_idx <= 3 * section_h)
                is_w = (mod3 == 2) & (pair_idx <= 3 * section_w)
                pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))
        else:
            t_end: tl.constexpr = section_t
            h_end: tl.constexpr = section_t + section_h
            is_h = (pair_idx >= t_end) & (pair_idx < h_end)
            is_w = pair_idx >= h_end
            pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))

        cos_base = cos_sin_cache_ptr + pos * rd
        cos_row = tl.load(cos_base + pair_idx, mask=valid_pair, other=0.0).to(
            tl.float32
        )
        sin_row = tl.load(
            cos_base + pair_idx + half_rd, mask=valid_pair, other=0.0
        ).to(tl.float32)

        q_heads = tl.arange(0, pad_n_qh)
        q_even_addr = q_heads[:, None] * hd + even_off[None, :]
        q_odd_addr = q_heads[:, None] * hd + odd_off[None, :]
        q_mask = (q_heads[:, None] < n_qh) & (pair_idx[None, :] < half_rd)

        qe = tl.load(q_row + q_even_addr, mask=q_mask, other=0.0).to(
            cos_row.dtype
        )
        qo = tl.load(q_row + q_odd_addr, mask=q_mask, other=0.0).to(
            cos_row.dtype
        )
        tl.store(
            q_row + q_even_addr,
            qe * cos_row[None, :] - qo * sin_row[None, :],
            mask=q_mask,
        )
        tl.store(
            q_row + q_odd_addr,
            qo * cos_row[None, :] + qe * sin_row[None, :],
            mask=q_mask,
        )

        k_heads = tl.arange(0, pad_n_kh)
        k_even_addr = k_heads[:, None] * hd + even_off[None, :]
        k_odd_addr = k_heads[:, None] * hd + odd_off[None, :]
        k_mask = (k_heads[:, None] < n_kh) & (pair_idx[None, :] < half_rd)

        ke = tl.load(k_row + k_even_addr, mask=k_mask, other=0.0).to(
            cos_row.dtype
        )
        ko = tl.load(k_row + k_odd_addr, mask=k_mask, other=0.0).to(
            cos_row.dtype
        )
        tl.store(
            k_row + k_even_addr,
            ke * cos_row[None, :] - ko * sin_row[None, :],
            mask=k_mask,
        )
        tl.store(
            k_row + k_odd_addr,
            ko * cos_row[None, :] + ke * sin_row[None, :],
            mask=k_mask,
        )


@triton.jit
def _rope_1d_fwd(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0,
    k_stride0,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rd: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    pid = tl.program_id(0)
    q_row = q_ptr + pid * q_stride0
    k_row = k_ptr + pid * k_stride0
    half_rd: tl.constexpr = rd // 2
    pad_rd_half: tl.constexpr = pad_rd // 2

    pos = tl.load(positions_ptr + pid)
    pair_idx = tl.arange(0, pad_rd_half)
    valid_pair = pair_idx < half_rd

    cos = tl.load(
        cos_sin_cache_ptr + pos * rd + pair_idx, mask=valid_pair, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rd + pair_idx + half_rd,
        mask=valid_pair,
        other=0.0,
    ).to(tl.float32)

    q_heads = tl.arange(0, pad_n_qh)
    k_heads = tl.arange(0, pad_n_kh)

    if is_neox_style:
        even_off = pair_idx
        odd_off = pair_idx + half_rd
    else:
        even_off = pair_idx * 2
        odd_off = even_off + 1

    q_even_addr = q_heads[:, None] * hd + even_off[None, :]
    q_odd_addr = q_heads[:, None] * hd + odd_off[None, :]
    q_mask = (q_heads[:, None] < n_qh) & (pair_idx[None, :] < half_rd)
    qe = tl.load(q_row + q_even_addr, mask=q_mask, other=0.0).to(cos.dtype)
    qo = tl.load(q_row + q_odd_addr, mask=q_mask, other=0.0).to(cos.dtype)
    tl.store(
        q_row + q_even_addr, qe * cos[None, :] - qo * sin[None, :], mask=q_mask
    )
    tl.store(
        q_row + q_odd_addr, qo * cos[None, :] + qe * sin[None, :], mask=q_mask
    )

    k_even_addr = k_heads[:, None] * hd + even_off[None, :]
    k_odd_addr = k_heads[:, None] * hd + odd_off[None, :]
    k_mask = (k_heads[:, None] < n_kh) & (pair_idx[None, :] < half_rd)
    ke = tl.load(k_row + k_even_addr, mask=k_mask, other=0.0).to(cos.dtype)
    ko = tl.load(k_row + k_odd_addr, mask=k_mask, other=0.0).to(cos.dtype)
    tl.store(
        k_row + k_even_addr, ke * cos[None, :] - ko * sin[None, :], mask=k_mask
    )
    tl.store(
        k_row + k_odd_addr, ko * cos[None, :] + ke * sin[None, :], mask=k_mask
    )


def triton_mrope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    mrope_interleaved_glm: bool,
    is_neox_style: bool,
    axis_map: Optional[torch.Tensor],
) -> None:
    """In-place mrotary embedding on q and k. Matches sglang's signature."""
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_rd = triton.next_power_of_2(rotary_dim)

    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)

    # Dummy tensor when axis_map is None — kernel expects a valid pointer.
    axis_map_arg = axis_map if axis_map is not None else q

    # Grid capped at 1024: max concurrent blocks for 256-thread blocks
    # (128 SMs × 8 blocks/SM on H20). Each program loops over
    # ceil(N / grid_size) tokens via the persistent loop.
    grid_size = min(num_tokens, 1024)

    _mrope_fused_fwd_persistent[(grid_size,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_rd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        mrope_interleaved_glm,
        is_neox_style,
        num_warps=8,
        N=num_tokens,
        GRID_SIZE=grid_size,
        axis_map_ptr=axis_map_arg,
    )


def _rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    if positions.ndim == 2:
        positions = positions[0].contiguous()
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_rd = triton.next_power_of_2(rotary_dim)
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)
    _rope_1d_fwd[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_rd,
        is_neox_style,
    )


def mrotary_embedding(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-level MRotaryEmbedding entry for sglang.

    Matches sglang's MRotaryEmbedding.forward_cuda dispatch:
      - 2D positions with mrope_section -> triton_mrope_fused
      - otherwise                       -> 1D RoPE
    """
    logger.debug("GEMS MROTARY_EMBEDDING")

    d = getattr(obj, "__dict__", None)
    if d is not None:
        mrope_section = d.get("mrope_section")
    else:
        mrope_section = getattr(obj, "mrope_section", None)

    if positions.ndim == 2 and mrope_section:
        if d is not None:
            triton_mrope_fused(
                query,
                key,
                d["cos_sin_cache"],
                positions,
                mrope_section,
                d["head_size"],
                d["rotary_dim"],
                d.get("mrope_interleaved", False),
                d.get("mrope_interleaved_glm", False),
                d["is_neox_style"],
                d.get("axis_map"),
            )
        else:
            triton_mrope_fused(
                query,
                key,
                obj.cos_sin_cache,
                positions,
                mrope_section,
                obj.head_size,
                obj.rotary_dim,
                getattr(obj, "mrope_interleaved", False),
                getattr(obj, "mrope_interleaved_glm", False),
                obj.is_neox_style,
                getattr(obj, "axis_map", None),
            )
    else:
        _rope_1d(
            query,
            key,
            obj.cos_sin_cache,
            positions,
            obj.head_size,
            obj.rotary_dim,
            obj.is_neox_style,
        )

    return query, key
