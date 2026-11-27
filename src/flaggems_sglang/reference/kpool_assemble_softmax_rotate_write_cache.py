# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent Torch oracle returning a cache without input mutation."""

import torch


def reference(
    pool,
    buf,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    n_from_tail,
    chunk_src_start,
    tail_logical_base,
    ape,
    loc,
    write_mask=None,
    round_scale=False,
):
    result = buf.clone()
    rows = req_pool_idx.numel()
    if rows == 0:
        return result
    active = (
        torch.ones(rows, dtype=torch.bool, device=buf.device)
        if write_mask is None
        else write_mask.bool()
    )
    req = req_pool_idx[active].long()
    count = n_from_tail[active].long()
    base = tail_logical_base[active].long()
    source = chunk_src_start[active].long()
    slots = torch.arange(pool.index_kpool, device=buf.device)
    from_tail = slots[None, :] < count[:, None]
    tail_pos = (base[:, None] + slots) % tail_k.shape[1]
    chunk_pos = source[:, None] + slots - count[:, None]
    # Unselected branches may carry dummy sources (all-tail uses one dummy
    # chunk row). Select valid addresses before gathering either branch.
    chunk_pos = torch.where(from_tail, 0, chunk_pos)
    tk = tail_k[req[:, None], tail_pos].float()
    ts = tail_score[req[:, None], tail_pos].float()
    k = torch.where(from_tail[..., None], tk, chunk_k[chunk_pos].float())
    scores = (
        torch.where(from_tail[..., None], ts, chunk_score[chunk_pos].float())
        + ape.float()
    )
    x = (k * torch.softmax(scores, dim=1)).sum(dim=1)
    x = x.to(torch.bfloat16).float()
    # Sylvester Hadamard matrix, independent of the kernel's butterfly code.
    h = torch.ones((1, 1), device=buf.device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    # Elementwise FP32 reduction avoids TF32/BF16 matmul policies.
    x = (x[:, None, :] * h[None, :, :]).sum(-1) * (128**-0.5)
    x = x.to(torch.bfloat16).float()
    scale = x.abs().amax(-1).clamp_min(1e-4) / 127.0
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (x / scale[:, None]).clamp(-127, 127)
    rounded = quantized.sign() * (quantized.abs() + 0.5).floor()
    encoded = rounded.to(torch.int8).view(torch.uint8)
    dest = loc[active].long()
    page, slot = dest // pool.slots_per_page, dest % pool.slots_per_page
    cols = torch.arange(128, device=buf.device)
    offset = slot[:, None] * 128 + cols
    raw = result.view(torch.uint8)
    raw[page[:, None], offset] = encoded
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(-1, 4)
    scale_offset = pool.slots_per_page * 128 + slot[:, None] * 4
    raw[page[:, None], scale_offset + torch.arange(4, device=buf.device)] = (
        scale_bytes
    )
    return result
