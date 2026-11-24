# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch
import torch.nn.functional as F


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
    null_block_id: int,
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
    valid_state = state_indices != null_block_id
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


def reference(
    x,
    output,
    conv_state,
    conv_weight,
    query_start_loc,
    state_indices,
    has_initial_state,
    *,
    num_prefills,
    max_len,
    state_len,
    kernel_width,
    dilation,
    null_block_id,
    min_tokens=0,
    block_t=None,
    block_c=None,
):
    output.copy_(
        _prefill_conv_reference(
            x,
            conv_state,
            conv_weight,
            query_start_loc,
            state_indices,
            has_initial_state,
            max_len=max_len,
            dilation=dilation,
            null_block_id=null_block_id,
        )
    )
    return True
