# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def reference(
    predicts,
    accept_index,
    commit_count,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
):
    """Independent CPU traversal that preserves the untouched output entries."""
    predict = predicts.tolist()
    accept = accept_index.tolist()
    count = commit_count.tolist()
    draft = candidates.tolist()
    slots = retrieve_index.tolist()
    child = retrieve_next_token.tolist()
    siblings = retrieve_next_sibling.tolist()
    target = target_predict.flatten().tolist()
    for row, row_slots in enumerate(slots):
        slot = row_slots[0]
        accept[row][0] = slot
        node, count[row] = 0, 1
        for step in range(1, 4):
            possible = child[row][node]
            while possible != -1 and draft[row][possible] != target[slot]:
                possible = siblings[row][possible]
            if possible == -1:
                break
            predict[slot] = target[slot]
            node = possible
            slot = row_slots[node]
            accept[row][step] = slot
            count[row] += 1
        predict[slot] = target[slot]
    for out, expected in zip(
        (predicts, accept_index, commit_count), (predict, accept, count)
    ):
        out.copy_(
            torch.tensor(expected, dtype=out.dtype, device=out.device).reshape(
                out.shape
            )
        )
