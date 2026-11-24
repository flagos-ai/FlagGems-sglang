# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flaggems_sglang import ple_prefill_short_conv_

from . import conftest as cfg

NULL_BLOCK_ID = -1
pytestmark = pytest.mark.ple_prefill_short_conv_


def _prefill_conv_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    max_len: int,
    dilation: int,
) -> torch.Tensor:
    num_reqs = state_indices.numel()
    channels, kernel_width = conv_weight.shape
    state_len = (kernel_width - 1) * dilation
    lengths = query_start_loc[1:].long() - query_start_loc[:-1].long()
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
    req_indices = torch.searchsorted(
        query_start_loc[1:].long(), positions, right=True
    )
    col_indices = positions - query_start_loc.long()[req_indices]

    packed = x.new_zeros((num_reqs, max_len, channels))
    packed[req_indices, col_indices] = x
    packed = packed.transpose(1, 2).contiguous()
    valid_state = state_indices != NULL_BLOCK_ID
    safe_indices = torch.where(
        valid_state,
        state_indices.long(),
        torch.zeros_like(state_indices.long()),
    )
    state = conv_state.index_select(0, safe_indices)[..., :state_len].to(
        x.dtype
    )
    initial = torch.where(
        (valid_state & has_initial_state).view(num_reqs, 1, 1),
        state,
        torch.zeros_like(state),
    )
    history = torch.cat((initial, packed), dim=-1)
    conv = F.conv1d(
        history,
        conv_weight.unsqueeze(1),
        groups=channels,
        dilation=dilation,
    )
    conv = F.silu(conv).transpose(1, 2).contiguous()
    valid_tokens = torch.arange(max_len, device=x.device).view(
        1, max_len
    ) < lengths.view(num_reqs, 1)
    conv.masked_fill_(
        ~(valid_tokens & valid_state.view(num_reqs, 1)).unsqueeze(-1), 0
    )
    output = conv[req_indices, col_indices].contiguous()

    state_offsets = torch.arange(state_len, device=x.device).view(
        1, 1, state_len
    )
    next_state = history.gather(
        2,
        (lengths.view(num_reqs, 1, 1) + state_offsets).expand(
            -1, channels, -1
        ),
    )
    update_mask = valid_state & (lengths > 0)
    selected = conv_state.index_select(0, safe_indices)
    selected[..., :state_len] = torch.where(
        update_mask.view(num_reqs, 1, 1),
        next_state.to(conv_state.dtype),
        selected[..., :state_len],
    )
    conv_state.index_copy_(0, safe_indices[update_mask], selected[update_mask])
    return output


@pytest.mark.skipif(
    not torch.get_device_module(cfg.device).is_available(),
    reason="CUDA is required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lengths", [[1], [15, 16, 17], [0, 32, 33]])
def test_ple_prefill_conv_boundaries_state_and_determinism(
    dtype: torch.dtype, lengths: list[int]
) -> None:
    torch.manual_seed(456 + sum(lengths))
    device = torch.device(cfg.device)
    channels = 65
    kernel_width = 4
    dilation = 3
    state_len = (kernel_width - 1) * dilation
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    total_tokens = sum(lengths)
    x = torch.randn((total_tokens, channels), device=device, dtype=dtype)
    weight = torch.randn((channels, kernel_width), device=device, dtype=dtype)
    slots = max(len(lengths), 1) + 1
    initial_state = torch.randn(
        (slots, state_len + 2, channels), device=device, dtype=dtype
    ).transpose(1, 2)
    assert not initial_state.is_contiguous()
    state_indices = torch.arange(
        len(lengths), device=device, dtype=torch.int32
    )
    if len(lengths) > 1:
        state_indices[-1] = NULL_BLOCK_ID
    has_initial = torch.tensor(
        [(index % 2) == 0 for index in range(len(lengths))],
        device=device,
        dtype=torch.bool,
    )

    reference_state = initial_state.clone()
    expected = _prefill_conv_reference(
        x,
        reference_state,
        weight,
        query_start_loc,
        state_indices,
        has_initial,
        max_len=max(lengths),
        dilation=dilation,
    )
    actual_state = initial_state.clone()
    actual = torch.empty_like(x)
    threshold_output = torch.full_like(x, 7.0)
    threshold_state = initial_state.clone()
    with pytest.raises(NotImplementedError):
        ple_prefill_short_conv_(
            x,
            threshold_output,
            threshold_state,
            weight,
            query_start_loc,
            state_indices,
            has_initial,
            num_prefills=len(lengths),
            max_len=max(lengths),
            state_len=state_len,
            kernel_width=kernel_width,
            dilation=dilation,
            null_block_id=NULL_BLOCK_ID,
            min_tokens=total_tokens + 1,
        )
    assert torch.equal(threshold_output, torch.full_like(x, 7.0))
    assert torch.equal(threshold_state, initial_state)
    assert ple_prefill_short_conv_(
        x,
        actual,
        actual_state,
        weight,
        query_start_loc,
        state_indices,
        has_initial,
        num_prefills=len(lengths),
        max_len=max(lengths),
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        null_block_id=NULL_BLOCK_ID,
        min_tokens=0,
    )
    tolerance = 2.0e-5 if dtype == torch.float32 else 4.0e-2
    torch.testing.assert_close(
        actual, expected, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_state, reference_state, atol=tolerance, rtol=tolerance
    )

    first_output = actual.clone()
    first_state = actual_state.clone()
    for _ in range(10):
        actual_state.copy_(initial_state)
        assert ple_prefill_short_conv_(
            x,
            actual,
            actual_state,
            weight,
            query_start_loc,
            state_indices,
            has_initial,
            num_prefills=len(lengths),
            max_len=max(lengths),
            state_len=state_len,
            kernel_width=kernel_width,
            dilation=dilation,
            null_block_id=NULL_BLOCK_ID,
            min_tokens=0,
        )
        assert torch.equal(actual, first_output)
        assert torch.equal(actual_state, first_state)
