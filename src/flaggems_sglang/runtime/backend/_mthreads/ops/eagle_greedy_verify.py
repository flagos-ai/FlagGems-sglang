# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_greedy(
    Prediction,
    AcceptIndex,
    CommitCount,
    Candidates,
    Slots,
    Child,
    Sibling,
    Target,
):
    row = tl.program_id(0)
    base = row * 4
    slot = tl.load(Slots + base).to(tl.int64)
    node = tl.cast(0, tl.int64)
    count = 1
    tl.store(AcceptIndex + base, slot)

    # Match drafts before the bonus token. The caller supplies valid tree links
    # and exclusive prediction slots for each row.
    for _ in range(3):
        if node != -1:
            node = tl.load(Child + base + node).to(tl.int64)
            target = tl.load(Target + slot)
            matched = False
            for _ in range(4):
                if node != -1 and not matched:
                    candidate = tl.load(Candidates + base + node)
                    if candidate == target:
                        tl.store(Prediction + slot, target)
                        slot = tl.load(Slots + base + node).to(tl.int64)
                        tl.store(AcceptIndex + base + count, slot)
                        count += 1
                        matched = True
                    else:
                        node = tl.load(Sibling + base + node).to(tl.int64)
            if not matched:
                node = tl.cast(-1, tl.int64)

    tl.store(Prediction + slot, tl.load(Target + slot))
    tl.store(CommitCount + row, count)


def eagle_greedy_verify(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    commit_count: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Verify width-four EAGLE trees and write counts including the bonus token.

    Inputs are contiguous [batch, 4] int32 or int64 tensors on one MUSA device.
    Tree links are local columns or -1. Prediction slots address the flattened
    target predictions and belong to one row only. The caller provides three
    non-aliasing int32 output buffers, initialized as needed: [batch * 4],
    [batch, 4], and [batch], respectively. Unvisited prediction and accept-index
    entries are preserved. Writes occur on the current stream.
    """
    batch = candidates.shape[0] if candidates.ndim == 2 else 0
    width_four = (batch, 4)
    inputs = (
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
    )
    outputs = (
        (predicts, (batch * 4,)),
        (accept_index, width_four),
        (commit_count, (batch,)),
    )
    if candidates.device.type != "musa":
        raise ValueError("EAGLE greedy verification requires a MUSA device")
    if any(
        tensor.shape != width_four
        or tensor.dtype not in (torch.int32, torch.int64)
        or tensor.device != candidates.device
        or not tensor.is_contiguous()
        for tensor in inputs
    ) or any(
        tensor.shape != shape
        or tensor.dtype != torch.int32
        or tensor.device != candidates.device
        or not tensor.is_contiguous()
        for tensor, shape in outputs
    ):
        raise ValueError("Invalid EAGLE greedy verification tensor layout")
    if batch:
        _verify_greedy[(batch,)](
            predicts, accept_index, commit_count, *inputs, num_warps=4
        )


__all__ = ["eagle_greedy_verify"]
