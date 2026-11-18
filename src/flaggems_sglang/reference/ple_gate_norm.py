# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import math

import torch


def _grouped_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (
        x_float
        * torch.rsqrt(variance + eps)
        * (1.0 + weight.float().view(1, *x.shape[1:]))
    ).to(x.dtype)


def _gate_norm_reference(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = key.shape[-1]
    norm_key = _grouped_norm_reference(key, key_weight, eps)
    norm_query = _grouped_norm_reference(query, query_weight, eps)
    gate = (norm_key * norm_query).sum(dim=-1, keepdim=True) / math.sqrt(
        hidden_size
    )
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1.0e-6).sqrt())
    gated = gate * value.unsqueeze(1)
    normalized = _grouped_norm_reference(gated, conv_weight, eps)
    return gated, normalized


def reference(
    key, query, value, key_weight, query_weight, conv_weight, hc_count, eps
):
    return _gate_norm_reference(
        key, query, value, key_weight, query_weight, conv_weight, eps
    )
