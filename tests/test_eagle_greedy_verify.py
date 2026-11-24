# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

pytestmark = [
    pytest.mark.eagle_greedy_verify,
    pytest.mark.skipif(flaggems_sglang.device != "musa", reason="MUSA kernel"),
]


def _case(batch, dtype):
    # Permute slots inside each row. Include all acceptance lengths, and rows
    # where the first child is rejected but its sibling is accepted.
    drafts = torch.tensor([[0, 11, 22, 33]], dtype=dtype).repeat(batch, 1)
    slots = torch.arange(batch, dtype=dtype).unsqueeze(1) * 4
    slots = slots + torch.tensor([[2, 1, 3, 0]], dtype=dtype)
    children = torch.tensor([[1, 2, 3, -1]], dtype=dtype).repeat(batch, 1)
    siblings = torch.full_like(children, -1)
    target = torch.full((batch * 4,), 99, dtype=dtype)
    for row in range(batch):
        if row % 5 == 4:
            siblings[row, 1] = 2
            target[slots[row, 0]] = drafts[row, 2]
        else:
            for step in range(row % 5):
                target[slots[row, step]] = drafts[row, step + 1]
    inputs = (drafts, slots, children, siblings, target.view(batch, 4))
    outputs = (
        torch.full((batch * 4,), -3, dtype=torch.int32),
        torch.full((batch, 4), -4, dtype=torch.int32),
        torch.full((batch,), -5, dtype=torch.int32),
    )
    return outputs, inputs


@pytest.mark.parametrize("batch", [0, 1, 5, 31, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_eagle_greedy_verify(batch, dtype):
    op = flaggems_sglang.get_op("eagle_greedy_verify")
    assert op is flaggems_sglang.eagle_greedy_verify
    assert "._mthreads.ops." in op.__module__
    outputs, inputs = _case(batch, dtype)
    get_reference("eagle_greedy_verify")(*outputs, *inputs)
    inputs = tuple(
        torch.cat((torch.zeros(4, dtype=t.dtype), t.flatten()))
        .to(flaggems_sglang.device)[4:]
        .reshape(t.shape)
        for t in inputs
    )
    actual, _ = _case(batch, dtype)
    actual = tuple(t.to(flaggems_sglang.device) for t in actual)
    assert op(*actual, *inputs) is None
    for obtained, expected in zip(actual, outputs):
        torch.testing.assert_close(obtained.cpu(), expected, rtol=0, atol=0)


def test_eagle_greedy_verify_rejects_noncontiguous_input_before_writes():
    outputs, inputs = _case(5, torch.int32)
    outputs = tuple(t.to(flaggems_sglang.device) for t in outputs)
    inputs = [t.to(flaggems_sglang.device) for t in inputs]
    inputs[0] = inputs[0].t().contiguous().t()
    with pytest.raises(ValueError, match="layout"):
        flaggems_sglang.eagle_greedy_verify(*outputs, *inputs)
    for tensor, sentinel in zip(outputs, (-3, -4, -5)):
        assert torch.all(tensor == sentinel)
