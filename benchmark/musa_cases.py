# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared input construction for MP31 expert GEMM validation and timing."""

import torch


def expert_inputs(shape, cur_dtype, device):
    tokens, fused, weighted = shape
    block = 16 if tokens == 4 else 64 if tokens == 16384 else 32
    hidden, width = (256, 2048) if weighted else (2048, 512)
    a = torch.randn(
        (tokens * 8 if weighted else tokens, hidden),
        dtype=cur_dtype,
        device=device,
    )
    b = torch.randn((256, width, hidden), dtype=cur_dtype, device=device)
    weights = torch.rand((tokens, 8), device=device)
    ids, eids = [], []
    for expert in range(256):
        indices = list(range(expert, tokens * 8, 256))
        indices += [tokens * 8] * ((-len(indices)) % block)
        ids.extend(indices)
        eids.extend([expert] * (len(indices) // block))
    config = dict(
        BLOCK_SIZE_M=block,
        BLOCK_SIZE_N=32 if tokens == 4 else 128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1 if tokens == 4 else 4,
        num_warps=4 if tokens == 4 else 8,
        num_stages=1,
    )
    profile = "swiglu_down" if weighted else "swiglu_up" if fused else "m4_up"
    output = torch.empty(
        (tokens, 8, width // 2 if fused else width),
        dtype=cur_dtype,
        device=device,
    )
    yield a, b, output, weights, torch.tensor(
        ids, dtype=torch.int32, device=device
    ), torch.tensor(eids, dtype=torch.int32, device=device), torch.tensor(
        [len(ids)], dtype=torch.int32, device=device
    ), dict(
        config=config,
        top_k=1 if weighted else 8,
        mul_routed_weight=bool(weighted),
        fuse_swiglu=bool(fused),
        tuning_profile=profile,
    )
