# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(x, weight, hc_count, eps):
    values = x.float().unflatten(-1, (hc_count, x.shape[-1] // hc_count))
    norm = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (norm.flatten(-2) * (1 + weight.float())).to(x.dtype)
