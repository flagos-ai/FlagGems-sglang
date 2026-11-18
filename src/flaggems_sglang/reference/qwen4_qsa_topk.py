# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(logits, row_starts, row_ends, topk):
    result = torch.full(
        (logits.shape[0], topk), -1, device=logits.device, dtype=torch.int32
    )
    for row in range(logits.shape[0]):
        start, end = int(row_starts[row]), int(row_ends[row])
        selected = sorted(
            range(start, end),
            key=lambda col: float(logits[row, col]),
            reverse=True,
        )[:topk]
        result[row, : len(selected)] = torch.tensor(
            [col - start for col in selected],
            device=logits.device,
            dtype=torch.int32,
        )
    return result
