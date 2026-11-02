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

from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    (tokens,) = shape
    routed = torch.randn((tokens, 8, 2048), dtype=cur_dtype, device=device)
    gate = torch.randn((tokens, 1), dtype=cur_dtype, device=device)
    shared = torch.randn((tokens, 2048), dtype=cur_dtype, device=device)
    yield routed, gate, shared, torch.empty_like(shared)


@pytest.mark.deterministic_moe_combine
@pytest.mark.skipif(
    gems.vendor_name != "mthreads",
    reason="MUSA-specific operator; see plugin PR #125",
)
def test_perf_deterministic_moe_combine():
    bench = OpBenchmark(
        op_name="deterministic_moe_combine",
        torch_op=get_reference("deterministic_moe_combine"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=[(t,) for t in (2048, 4096, 6144, 8192, 16384)],
        shape_desc="tokens",
    )
    bench.set_gems(gems.deterministic_moe_combine)
    bench.run()
