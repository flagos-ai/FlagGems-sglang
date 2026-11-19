# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(logits, normed, hc_count):
    hidden = normed.shape[-1] // hc_count
    return (
        (torch.sigmoid(logits) * normed)
        .unflatten(-1, (hc_count, hidden))
        .mean(-2)
    )
