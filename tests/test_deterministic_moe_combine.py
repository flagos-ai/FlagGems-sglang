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
    pytest.mark.deterministic_moe_combine,
    pytest.mark.skipif(
        gems.vendor_name != "mthreads",
        reason="MUSA-specific operator; see plugin PR #125",
    ),
]


@pytest.mark.parametrize("tokens", [2048, 4096, 6144, 8192, 16384])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.deterministic_moe_combine
def test_combine_refreshed_graph_and_output(tokens, strided):
    shape = (tokens, 8, 4096 if strided else 2048)
    backing = torch.empty(shape, device=gems.device, dtype=torch.bfloat16)
    routed = backing[..., ::2] if strided else backing
    shared = torch.empty(
        (tokens, 2048), device=gems.device, dtype=routed.dtype
    )
    gate = torch.empty((tokens, 1), device=gems.device, dtype=routed.dtype)
    out = torch.empty_like(shared)
    routed.fill_(1)
    shared.fill_(2)
    gate.zero_()
    gems.deterministic_moe_combine(routed, gate, shared, out)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        gems.deterministic_moe_combine(routed, gate, shared, out)
    for phase in (0, 1, 0):
        routed.fill_(phase + 1)
        shared.fill_(phase + 2)
        gate.zero_()
        expected = get_reference("deterministic_moe_combine")(
            routed, gate, shared
        )
        out.fill_(float("nan"))
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
