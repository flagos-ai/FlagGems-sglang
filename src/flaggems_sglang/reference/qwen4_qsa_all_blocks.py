# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(lengths, topk):
    columns = torch.arange(topk, device=lengths.device, dtype=torch.int32)
    return torch.where(columns[None] < lengths[:, None], columns[None], -1)
