# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.reference import get_reference

pytestmark = [
    pytest.mark.fused_shared_expert_gate_tail,
    pytest.mark.skipif(
        gems.vendor_name != "mthreads",
        reason="MUSA-specific operator; see plugin PR #125",
    ),
]


@pytest.mark.parametrize("kind", ["zero", "dyadic", "random", "offset"])
@pytest.mark.fused_shared_expert_gate_tail
def test_shared_gate_bf16_boundaries(kind):
    device = gems.device
    hidden = torch.zeros((5, 2048), dtype=torch.bfloat16, device=device)[1:]
    weight = torch.zeros((1, 2048), dtype=hidden.dtype, device=device)
    shared = torch.randn_like(hidden)
    if kind in ("dyadic", "offset"):
        hidden[:, 0] = torch.tensor([0, 0.25, -0.5, 1], device=device)
        weight[:, 0] = 0.5
    elif kind == "random":
        hidden.normal_()
        weight.normal_()
    out = torch.empty_like(shared)
    logits = torch.empty((4, 1), dtype=hidden.dtype, device=device)
    prob = torch.empty_like(logits)
    expected_logits, expected_prob = torch.empty_like(
        logits
    ), torch.empty_like(prob)
    expected = get_reference("fused_shared_expert_gate_tail")(
        hidden,
        weight,
        shared,
        gate_logits_out=expected_logits,
        gate_prob_out=expected_prob,
    )
    assert (
        gems.fused_shared_expert_gate_tail(
            hidden,
            weight,
            shared,
            out,
            gate_logits_out=logits,
            gate_prob_out=prob,
        )
        is out
    )
    for actual, ref in [
        (logits, expected_logits),
        (prob, expected_prob),
        (out, expected),
    ]:
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.fused_shared_expert_gate_tail
def test_shared_gate_refreshed_graph():
    hidden = torch.zeros((4, 2048), dtype=torch.bfloat16, device=gems.device)
    weight = torch.zeros((1, 2048), dtype=hidden.dtype, device=gems.device)
    weight[:, 0] = 1
    shared, out = torch.ones_like(hidden), torch.empty_like(hidden)
    gems.fused_shared_expert_gate_tail(hidden, weight, shared, out)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        gems.fused_shared_expert_gate_tail(hidden, weight, shared, out)
    pointers = [t.data_ptr() for t in (hidden, weight, shared, out)]
    for value in (-1, 0, 1):
        hidden[:, 0] = value
        shared.fill_(value + 2)
        expected = get_reference("fused_shared_expert_gate_tail")(
            hidden, weight, shared
        )
        out.fill_(float("nan"))
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert pointers == [t.data_ptr() for t in (hidden, weight, shared, out)]


@pytest.mark.fused_shared_expert_gate_tail
def test_shared_gate_alias_rejected_before_write():
    hidden = torch.ones((4, 2048), dtype=torch.bfloat16, device=gems.device)
    weight = torch.ones((1, 2048), dtype=hidden.dtype, device=gems.device)
    shared = torch.ones_like(hidden)
    before = hidden.clone()
    with pytest.raises(ValueError, match="alias"):
        gems.fused_shared_expert_gate_tail(hidden, weight, shared, hidden)
    assert torch.equal(hidden, before)


@pytest.mark.fused_shared_expert_gate_tail
def test_explicit_options_reach_real_launch(monkeypatch):
    import importlib

    module = importlib.import_module(
        gems.fused_shared_expert_gate_tail.__module__
    )
    original_kernel, calls = module._shared_expert_gate_tail_kernel, []

    class ObservedKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def observed(*args, **kwargs):
                calls.append(dict(kwargs))
                return launch(*args, **kwargs)

            return observed

    monkeypatch.setattr(
        module, "_shared_expert_gate_tail_kernel", ObservedKernel()
    )
    hidden = torch.zeros((4, 2048), device=gems.device, dtype=torch.bfloat16)
    weight = torch.ones((1, 2048), device=gems.device, dtype=hidden.dtype)
    shared = torch.randn_like(hidden)
    expected = get_reference("fused_shared_expert_gate_tail")(
        hidden, weight, shared
    )
    out = gems.fused_shared_expert_gate_tail(
        hidden, weight, shared, launch_options={"BLOCK_H": 512, "num_warps": 8}
    )
    assert calls[0]["BLOCK_H"] == 512 and calls[0]["num_warps"] == 8
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
