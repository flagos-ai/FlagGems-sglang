# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(residual, block, injection, weight, eps, hc_count):
    weights = 2 * torch.sigmoid(injection / hc_count)
    combined = residual.unflatten(
        -1, (hc_count, block.shape[-1])
    ) + block.unsqueeze(-2) * weights.unsqueeze(-1)
    values = combined.float()
    normalized = values * torch.rsqrt(
        values.square().mean(-1, keepdim=True) + eps
    )
    normalized = normalized * (
        1 + weight.float().reshape(hc_count, block.shape[-1])
    )
    return combined.flatten(-2), normalized.flatten(-2).to(residual.dtype)
