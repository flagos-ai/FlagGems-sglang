# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch
import torch.nn.functional as F


def reference(
    x,
    output,
    state,
    weight,
    indices,
    *,
    dilation,
    query_start_loc=None,
    has_initial_state=None,
    accepted_tokens=None,
    max_query_len=1,
    null_block_id=-1,
):
    """Independent per-request native conv oracle, including cache mutation."""
    state_len = (weight.shape[1] - 1) * dilation
    output.zero_()
    if indices is None:
        requests = [(0, x.shape[0], None)]
    else:
        starts = (
            query_start_loc.tolist()
            if query_start_loc is not None
            else list(range(x.shape[0] + 1))
        )
        requests = [
            (starts[r], starts[r + 1], i)
            for r, i in enumerate(indices.tolist())
        ]
    for r, (start, end, index) in enumerate(requests):
        if (
            end == start
            or index == null_block_id
            or (index is not None and not 0 <= index < state.shape[0])
        ):
            continue
        initial = x.new_zeros((x.shape[1], state_len))
        rollback = (
            0 if accepted_tokens is None else int(accepted_tokens[r]) - 1
        )
        has_history = (
            index is not None
            and (has_initial_state is None or bool(has_initial_state[r]))
            and 0 <= rollback < max_query_len
        )
        if has_history:
            initial = state[index, :, rollback : rollback + state_len].to(
                x.dtype
            )
        history = torch.cat((initial, x[start:end].T), dim=1)
        conv = F.conv1d(
            history[None],
            weight[:, None],
            groups=x.shape[1],
            dilation=dilation,
        )
        output[start:end].copy_(F.silu(conv[0]).T)
        if index is not None and state_len:
            if accepted_tokens is None:
                state[index, :, :state_len].copy_(history[:, -state_len:])
            else:
                count = state_len + end - start - 1
                state[index, :, :count].copy_(history[:, 1 : count + 1])
