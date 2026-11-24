# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(injection, block, residual, hc_count):
    weights = 2 * torch.sigmoid(injection / hc_count)
    delta = block.unsqueeze(-2) * weights.unsqueeze(-1)
    return (
        residual.unflatten(-1, (hc_count, block.shape[-1])) + delta
    ).flatten(-2)
