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

"""ple_prefill_short_conv_ for RDNA3 (gfx1100).

Same kernels and arithmetic as the generic op. The convolution reads the
cached state only in the time block that reaches back before the request
start (the state is strided per channel, so those reads are one element
each), and runs on wider channel blocks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

HAS_TRITON = True


@triton.jit
def _ple_prefill_dilated_conv_kernel(
    x_ptr,
    weight_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_ptr,
    conv_state_ptr,
    output_ptr,
    num_state_rows: tl.constexpr,
    channels: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_weight_channel: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_query_start: tl.constexpr,
    stride_state_index: tl.constexpr,
    stride_has_initial: tl.constexpr,
    stride_state_row: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_channel: tl.constexpr,
    null_block_id: tl.constexpr,
    kernel_width: tl.constexpr,
    dilation: tl.constexpr,
    has_state: tl.constexpr,
    block_t: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    """Compute dilated causal depthwise convolution without pack/transposes."""

    request = tl.program_id(0)
    time_block = tl.program_id(1)
    channel_block = tl.program_id(2)
    token_offsets = time_block * block_t + tl.arange(0, block_t)[:, None]
    channel_offsets = channel_block * block_c + tl.arange(0, block_c)[None, :]

    query_start = tl.load(
        query_start_loc_ptr + request * stride_query_start
    ).to(tl.int64)
    query_end = tl.load(
        query_start_loc_ptr + (request + 1) * stride_query_start
    ).to(tl.int64)
    query_len = query_end - query_start
    state_index = tl.load(state_indices_ptr + request * stride_state_index).to(
        tl.int64
    )
    valid_state = state_index != null_block_id
    if has_state:
        valid_state = (
            valid_state & (state_index >= 0) & (state_index < num_state_rows)
        )
        safe_state_index = tl.where(valid_state, state_index, 0)
        has_initial = (
            tl.load(has_initial_ptr + request * stride_has_initial).to(tl.int1)
            & valid_state
        )

    valid_tokens = token_offsets < query_len
    valid_channels = channel_offsets < channels
    accumulator = tl.zeros((block_t, block_c), dtype=tl.float32)
    history = time_block * block_t < (kernel_width - 1) * dilation

    for tap in tl.static_range(0, kernel_width):
        source_position = token_offsets - (kernel_width - 1 - tap) * dilation
        source_is_token = source_position >= 0
        token_ptrs = (
            x_ptr
            + (query_start + source_position) * stride_x_token
            + channel_offsets * stride_x_channel
        )
        token_values = tl.load(
            token_ptrs,
            mask=(
                valid_tokens
                & valid_channels
                & source_is_token
                & (source_position < query_len)
            ),
            other=0.0,
        )
        # Adding the (zero) state part keeps the generic type promotion and
        # signed-zero results.
        if has_state:
            source_values = token_values + tl.zeros(
                (block_t, block_c), dtype=conv_state_ptr.dtype.element_ty
            )
            if history:
                state_position = state_len + source_position
                state_ptrs = (
                    conv_state_ptr
                    + safe_state_index * stride_state_row
                    + channel_offsets * stride_state_channel
                    + state_position * stride_state_token
                )
                state_values = tl.load(
                    state_ptrs,
                    mask=(
                        valid_tokens
                        & valid_channels
                        & (~source_is_token)
                        & (state_position >= 0)
                        & (state_position < state_len)
                        & has_initial
                    ),
                    other=0.0,
                )
                source_values = token_values + state_values
        else:
            source_values = token_values + tl.zeros(
                (block_t, block_c), dtype=x_ptr.dtype.element_ty
            )
        conv_weight = tl.load(
            weight_ptr
            + channel_offsets * stride_weight_channel
            + tap * stride_weight_width,
            mask=valid_channels,
            other=0.0,
        )
        accumulator += source_values.to(tl.float32) * conv_weight.to(
            tl.float32
        )

    activated = accumulator * tl.sigmoid(accumulator)
    activated *= valid_state.to(tl.float32)
    output_ptrs = (
        output_ptr
        + (query_start + token_offsets) * stride_output_token
        + channel_offsets * stride_output_channel
    )
    tl.store(
        output_ptrs,
        activated,
        mask=valid_tokens & valid_channels,
    )


@triton.jit
def _ple_prefill_state_update_kernel(
    x_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_ptr,
    conv_state_ptr,
    num_state_rows: tl.constexpr,
    channels: tl.constexpr,
    state_len: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_query_start: tl.constexpr,
    stride_state_index: tl.constexpr,
    stride_has_initial: tl.constexpr,
    stride_state_row: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_token: tl.constexpr,
    null_block_id: tl.constexpr,
    block_s: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    """Update the state after the output kernel has consumed the old state."""

    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    state_offsets = tl.arange(0, block_s)[:, None]
    channel_offsets = channel_block * block_c + tl.arange(0, block_c)[None, :]
    valid_channels = channel_offsets < channels

    query_start = tl.load(
        query_start_loc_ptr + request * stride_query_start
    ).to(tl.int64)
    query_end = tl.load(
        query_start_loc_ptr + (request + 1) * stride_query_start
    ).to(tl.int64)
    query_len = query_end - query_start
    state_index = tl.load(state_indices_ptr + request * stride_state_index).to(
        tl.int64
    )
    valid_state = (
        (state_index != null_block_id)
        & (state_index >= 0)
        & (state_index < num_state_rows)
        & (query_len > 0)
    )
    safe_state_index = tl.where(valid_state, state_index, 0)
    has_initial = (
        tl.load(has_initial_ptr + request * stride_has_initial).to(tl.int1)
        & valid_state
    )

    source_position = query_len + state_offsets - state_len
    source_is_token = source_position >= 0
    token_ptrs = (
        x_ptr
        + (query_start + source_position) * stride_x_token
        + channel_offsets * stride_x_channel
    )
    token_values = tl.load(
        token_ptrs,
        mask=(
            valid_state
            & valid_channels
            & (state_offsets < state_len)
            & source_is_token
            & (source_position < query_len)
        ),
        other=0.0,
    )

    old_state_position = state_len + source_position
    old_state_ptrs = (
        conv_state_ptr
        + safe_state_index * stride_state_row
        + channel_offsets * stride_state_channel
        + old_state_position * stride_state_token
    )
    old_state_values = tl.load(
        old_state_ptrs,
        mask=(
            valid_state
            & valid_channels
            & (state_offsets < state_len)
            & (~source_is_token)
            & (old_state_position >= 0)
            & (old_state_position < state_len)
            & has_initial
        ),
        other=0.0,
    )
    next_state = token_values + old_state_values
    state_output_ptrs = (
        conv_state_ptr
        + safe_state_index * stride_state_row
        + channel_offsets * stride_state_channel
        + state_offsets * stride_state_token
    )
    tl.store(
        state_output_ptrs,
        next_state,
        mask=(valid_state & valid_channels & (state_offsets < state_len)),
    )


def _prefill_config(num_prefills, max_len, channels):
    """block_t, block_c and num_warps of the convolution."""
    if num_prefills * triton.cdiv(max_len, 8) >= 512:
        return 8, min(1024, triton.next_power_of_2(channels)), 8
    return 8, min(256, triton.next_power_of_2(channels)), 4


def can_use_ple_prefill_conv_triton(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_prefills: int,
    max_len: int,
    state_len: int,
    kernel_width: int,
    dilation: int,
    min_tokens: int = 0,
) -> bool:
    """Return whether PLE prefill can bypass pack/layout/native conv ops."""

    if not HAS_TRITON:
        return False
    if (
        x.ndim != 2
        or output.shape != x.shape
        or conv_weight.ndim != 2
        or conv_weight.shape != (x.shape[1], kernel_width)
        or x.shape[0] < min_tokens
        or num_prefills <= 0
        or max_len <= 0
        or kernel_width <= 0
        or kernel_width > 8
        or dilation <= 0
        or state_len != (kernel_width - 1) * dilation
    ):
        return False
    if query_start_loc.ndim != 1 or query_start_loc.numel() < num_prefills + 1:
        return False
    if state_indices.ndim != 1 or state_indices.numel() < num_prefills:
        return False
    if has_initial_state.ndim != 1 or has_initial_state.numel() < num_prefills:
        return False
    if conv_state.ndim != 3:
        return False
    if conv_state.shape[0] and (
        conv_state.shape[1] != x.shape[1] or conv_state.shape[2] < state_len
    ):
        return False
    device = x.device
    tensors = (
        output,
        conv_state,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
    )
    return bool(
        x.dtype in _SUPPORTED_DTYPES
        and output.dtype == x.dtype
        and conv_weight.dtype == x.dtype
        and conv_state.dtype in _SUPPORTED_DTYPES
        and device.type not in ("cpu", "meta")
        and all(tensor.device == device for tensor in tensors)
        and x.is_contiguous()
        and output.is_contiguous()
        and conv_weight.is_contiguous()
    )


def ple_prefill_short_conv_(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_prefills: int,
    max_len: int,
    state_len: int,
    kernel_width: int,
    dilation: int,
    null_block_id: int,
    min_tokens: int = 0,
    block_t: int | None = None,
    block_c: int | None = None,
) -> bool:
    """Write PLE prefill conv output/state directly; return whether fused."""

    if not can_use_ple_prefill_conv_triton(
        x,
        output,
        conv_state,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
        num_prefills=num_prefills,
        max_len=max_len,
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        min_tokens=min_tokens,
    ):
        raise NotImplementedError(
            "PLE prefill fusion received unsupported tensors or parameters"
        )
    num_warps = 8
    if block_t is None or block_c is None:
        if block_t is not None or block_c is not None:
            raise ValueError(
                "PLE prefill Triton block sizes must be set together"
            )
        block_t, block_c, num_warps = _prefill_config(
            num_prefills, max_len, x.shape[1]
        )
    if block_t <= 0 or block_c <= 0:
        raise ValueError("PLE prefill Triton block sizes must be positive")
    if block_t & (block_t - 1) or block_c & (block_c - 1):
        raise ValueError(
            "PLE prefill Triton block sizes must be powers of two"
        )

    has_state = conv_state.shape[0] > 0 and state_len > 0
    _ple_prefill_dilated_conv_kernel[
        (
            num_prefills,
            triton.cdiv(max_len, block_t),
            triton.cdiv(x.shape[1], block_c),
        )
    ](
        x,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
        conv_state,
        output,
        num_state_rows=conv_state.shape[0],
        channels=x.shape[1],
        state_len=state_len,
        stride_x_token=x.stride(0),
        stride_x_channel=x.stride(1),
        stride_weight_channel=conv_weight.stride(0),
        stride_weight_width=conv_weight.stride(1),
        stride_query_start=query_start_loc.stride(0),
        stride_state_index=state_indices.stride(0),
        stride_has_initial=has_initial_state.stride(0),
        stride_state_row=conv_state.stride(0),
        stride_state_channel=conv_state.stride(1),
        stride_state_token=conv_state.stride(2),
        stride_output_token=output.stride(0),
        stride_output_channel=output.stride(1),
        null_block_id=null_block_id,
        kernel_width=kernel_width,
        dilation=dilation,
        has_state=has_state,
        block_t=block_t,
        block_c=block_c,
        num_warps=num_warps,
    )
    # State writes must be a second launch: updating in a peer CTA of the
    # output launch would race output CTAs that still consume the old state.
    if has_state:
        block_s = triton.next_power_of_2(state_len)
        _ple_prefill_state_update_kernel[
            (num_prefills, triton.cdiv(x.shape[1], block_c))
        ](
            x,
            query_start_loc,
            state_indices,
            has_initial_state,
            conv_state,
            num_state_rows=conv_state.shape[0],
            channels=x.shape[1],
            state_len=state_len,
            stride_x_token=x.stride(0),
            stride_x_channel=x.stride(1),
            stride_query_start=query_start_loc.stride(0),
            stride_state_index=state_indices.stride(0),
            stride_has_initial=has_initial_state.stride(0),
            stride_state_row=conv_state.stride(0),
            stride_state_channel=conv_state.stride(1),
            stride_state_token=conv_state.stride(2),
            null_block_id=null_block_id,
            block_s=block_s,
            block_c=block_c,
            num_warps=8,
        )
    return True


__all__ = ["ple_prefill_short_conv_"]
