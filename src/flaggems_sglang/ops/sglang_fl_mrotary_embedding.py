# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of attention/sglang_fl_mrotary_embedding."""

import torch
import triton
import triton.language as tl

__all__ = ["sglang_fl_mrotary_embedding"]


@triton.jit
def _mrotary_embedding_kernel(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    axis_map_ptr,
    q_stride,
    k_stride,
    positions_stride,
    positions_token_stride,
    NQH: tl.constexpr,
    NKH: tl.constexpr,
    HD: tl.constexpr,
    RD: tl.constexpr,
    PAD_QH: tl.constexpr,
    PAD_KH: tl.constexpr,
    PAD_RD2: tl.constexpr,
    SEC_T: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    GLM: tl.constexpr,
    NEOX: tl.constexpr,
):
    token = tl.program_id(0)
    q_ptr = q_ptr + token * q_stride
    k_ptr = k_ptr + token * k_stride

    half = RD // 2

    # The token axis of ``positions`` carries its own stride: callers pass a
    # slice of a wider buffer, so the row stride and the element stride are
    # independent. Reading it as stride 1 silently lands on the wrong token.
    token_offset = token * positions_token_stride
    temporal = tl.load(positions_ptr + 0 * positions_stride + token_offset)
    height = tl.load(positions_ptr + 1 * positions_stride + token_offset)
    width = tl.load(positions_ptr + 2 * positions_stride + token_offset)

    t_cos_base = cos_sin_cache_ptr + temporal * RD
    h_cos_base = cos_sin_cache_ptr + height * RD
    w_cos_base = cos_sin_cache_ptr + width * RD
    t_sin_base = t_cos_base + half
    h_sin_base = h_cos_base + half
    w_sin_base = w_cos_base + half

    # Only rotary_dim // 2 lanes ever index the cache. Sizing this band by
    # head_size // 2 instead -- as the upstream kernel does -- reads
    # (head_size - rotary_dim) // 2 lanes past both the axis map and each
    # cache row. The store mask hides the result, but at the contract's own
    # geometry (head_size 256, rotary_dim 64) that is 96 wasted lanes per
    # token, four quarters of the band doing nothing.
    lanes = tl.arange(0, PAD_RD2)
    in_band = lanes < half

    if INTERLEAVED:
        if GLM:
            axes = tl.load(axis_map_ptr + lanes, mask=in_band, other=0)
            t_mask = in_band & (axes == 0)
            h_mask = in_band & (axes == 1)
            w_mask = in_band & (axes == 2)
        else:
            h_mask = in_band & ((lanes % 3) == 1) & (lanes < 3 * SEC_H)
            w_mask = in_band & ((lanes % 3) == 2) & (lanes < 3 * SEC_W)
            t_mask = in_band & ~(h_mask | w_mask)
    else:
        t_mask = in_band & (lanes < SEC_T)
        h_mask = in_band & (SEC_T <= lanes) & (lanes < SEC_T + SEC_H)
        w_mask = in_band & (SEC_T + SEC_H <= lanes) & (lanes < half)

    # Exactly one of the three masks holds for every in-band lane, so the
    # three masked loads sum to a single cos/sin row without a select chain.
    cos_row = tl.load(t_cos_base + lanes, mask=t_mask, other=0.0)
    cos_row += tl.load(h_cos_base + lanes, mask=h_mask, other=0.0)
    cos_row += tl.load(w_cos_base + lanes, mask=w_mask, other=0.0)
    sin_row = tl.load(t_sin_base + lanes, mask=t_mask, other=0.0)
    sin_row += tl.load(h_sin_base + lanes, mask=h_mask, other=0.0)
    sin_row += tl.load(w_sin_base + lanes, mask=w_mask, other=0.0)

    qh = tl.arange(0, PAD_QH)[:, None]
    kh = tl.arange(0, PAD_KH)[:, None]
    q_ok = (qh < NQH) & in_band[None, :]
    k_ok = (kh < NKH) & in_band[None, :]

    if NEOX:
        q_first = q_ptr + qh * HD + lanes[None, :]
        q_second = q_first + half
        k_first = k_ptr + kh * HD + lanes[None, :]
        k_second = k_first + half

        q1 = tl.load(q_first, mask=q_ok, other=0.0).to(cos_row.dtype)
        q2 = tl.load(q_second, mask=q_ok, other=0.0).to(cos_row.dtype)
        k1 = tl.load(k_first, mask=k_ok, other=0.0).to(cos_row.dtype)
        k2 = tl.load(k_second, mask=k_ok, other=0.0).to(cos_row.dtype)

        cos_b = cos_row[None, :]
        sin_b = sin_row[None, :]
        tl.store(q_first, q1 * cos_b - q2 * sin_b, mask=q_ok)
        tl.store(q_second, q2 * cos_b + q1 * sin_b, mask=q_ok)
        tl.store(k_first, k1 * cos_b - k2 * sin_b, mask=k_ok)
        tl.store(k_second, k2 * cos_b + k1 * sin_b, mask=k_ok)
    else:
        even = 2 * lanes
        q_even_ptr = q_ptr + qh * HD + even[None, :]
        q_odd_ptr = q_even_ptr + 1
        k_even_ptr = k_ptr + kh * HD + even[None, :]
        k_odd_ptr = k_even_ptr + 1

        q_e = tl.load(q_even_ptr, mask=q_ok, other=0.0).to(cos_row.dtype)
        q_o = tl.load(q_odd_ptr, mask=q_ok, other=0.0).to(cos_row.dtype)
        k_e = tl.load(k_even_ptr, mask=k_ok, other=0.0).to(cos_row.dtype)
        k_o = tl.load(k_odd_ptr, mask=k_ok, other=0.0).to(cos_row.dtype)

        cos_b = cos_row[None, :]
        sin_b = sin_row[None, :]
        tl.store(q_even_ptr, q_e * cos_b - q_o * sin_b, mask=q_ok)
        tl.store(q_odd_ptr, q_o * cos_b + q_e * sin_b, mask=q_ok)
        tl.store(k_even_ptr, k_e * cos_b - k_o * sin_b, mask=k_ok)
        tl.store(k_odd_ptr, k_o * cos_b + k_e * sin_b, mask=k_ok)


def sglang_fl_mrotary_embedding(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
    mrope_interleaved_glm,
    is_neox_style,
    axis_map=None,
):
    """Rotate ``q`` and ``k`` with multimodal positions, in place.

    Args:
        q: ``(num_tokens, num_q_heads * head_size)``.
        k: ``(num_tokens, num_k_heads * head_size)``.
        cos_sin_cache: ``(max_position, rotary_dim)``, cos in the first
            ``rotary_dim // 2`` columns and sin in the remaining ones.
        positions: ``(3, num_tokens)`` int64 temporal/height/width positions.
        mrope_section: Three axis quotas that must sum to
            ``rotary_dim // 2``.
        head_size: Channels per head.
        rotary_dim: Rotated channels per head, ``<= head_size``.
        mrope_interleaved: Sample the axes by striding every third lane.
        mrope_interleaved_glm: Take the axis per lane from ``axis_map``.
        is_neox_style: Pair lane ``i`` with ``i + rotary_dim // 2`` instead of
            the adjacent even/odd pair.
        axis_map: One axis index per rotary lane; required only when
            ``mrope_interleaved_glm`` is set.

    Returns:
        ``None``. Both ``q`` and ``k`` are modified in place, and the
        channels past ``rotary_dim`` are left untouched.
    """
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must be 2D (num_tokens, heads * head_size)")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same number of tokens")
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("positions must be (3, num_tokens)")
    if positions.shape[1] != q.shape[0]:
        raise ValueError("positions must have one column per token")
    if q.shape[1] % head_size != 0:
        raise ValueError("q width must be a multiple of head_size")
    if k.shape[1] % head_size != 0:
        raise ValueError("k width must be a multiple of head_size")
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2 != 0:
        raise ValueError(
            "rotary_dim must be positive, even and no larger than head_size"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("cos_sin_cache must be (max_position, rotary_dim)")
    if len(mrope_section) != 3:
        raise ValueError("mrope_section must have three entries")
    if sum(int(x) for x in mrope_section) != rotary_dim // 2:
        raise ValueError(
            "mrope_section must sum to rotary_dim // 2 " f"({rotary_dim // 2})"
        )
    if mrope_interleaved_glm:
        if axis_map is None:
            raise ValueError(
                "axis_map is required when mrope_interleaved_glm is set"
            )
        if axis_map.numel() != rotary_dim // 2:
            raise ValueError("axis_map must hold rotary_dim // 2 entries")
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("cos_sin_cache", cos_sin_cache),
        ("positions", positions),
    ):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be int32 or int64")

    num_tokens = q.shape[0]
    if num_tokens == 0:
        # No program would launch, and there is nothing to write back.
        return None

    half = rotary_dim // 2
    num_qh = q.shape[1] // head_size
    num_kh = k.shape[1] // head_size

    _mrotary_embedding_kernel[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        axis_map,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        positions.stride(1),
        num_qh,
        num_kh,
        head_size,
        rotary_dim,
        triton.next_power_of_2(num_qh),
        triton.next_power_of_2(num_kh),
        triton.next_power_of_2(half),
        int(mrope_section[0]),
        int(mrope_section[1]),
        int(mrope_section[2]),
        bool(mrope_interleaved),
        bool(mrope_interleaved_glm),
        bool(is_neox_style),
    )
    return None
