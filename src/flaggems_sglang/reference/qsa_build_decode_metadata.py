# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(
    seq_lens,
    req_pool_indices,
    req_to_token,
    sequence_lengths,
    row_req_pool_indices,
    prefix_lengths,
    compressed_lengths,
    write_locs,
    page_table,
    logical_positions,
    state_slots,
    ring_locs,
    compress_ratio,
    full_page_size,
    num_padding=0,
):
    rows = sequence_lengths.numel()
    lens = seq_lens[:rows].long().clone()
    requests = req_pool_indices[:rows].long().clone()
    if num_padding:
        lens[-num_padding:] = 1
        requests[-num_padding:] = 0
    current = (lens - 1).clamp_min(0)
    last = req_to_token[requests, current]
    sequence_lengths.copy_(lens)
    row_req_pool_indices.copy_(requests)
    prefix_lengths.copy_(current)
    compressed_lengths.copy_(lens // compress_ratio)
    write_locs.copy_(
        torch.where(
            (lens > 0) & (lens % compress_ratio == 0),
            last // compress_ratio,
            0,
        )
    )
    logical_positions.copy_(current)
    state_slots.copy_(requests * compress_ratio + current % compress_ratio)
    members = (
        current[:, None]
        - torch.arange(compress_ratio - 1, -1, -1, device=lens.device)
    ).clamp_min(0)
    ring_locs.copy_(
        requests[:, None] * compress_ratio + members % compress_ratio
    )
    count = min(page_table.shape[1], req_to_token.shape[1] // full_page_size)
    columns = torch.arange(count, device=lens.device) * full_page_size
    page_table[:, :count].copy_(
        (req_to_token[requests[:, None], columns] // full_page_size).clamp_min(
            0
        )
    )
