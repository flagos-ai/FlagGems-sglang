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

"""Benchmark for moe/gate_topk."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 256, k) for m in (1, 8, 64, 512) for k in (4, 8)]
MORE_SHAPES = [(m, 256, k) for m in (4096,) for k in (4, 8)]


def _input_fn(shape, cur_dtype, device):
    m, n, k = shape
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    g = torch.Generator(device="cpu").manual_seed(0)
    # Distinct values keep the tie-break rule out of the comparison.
    x = (
        torch.randperm(m * n, generator=g).reshape(m, n).float() / (m * n)
    ).to(device)
    yield x.contiguous(), {"k": k}


@pytest.mark.gate_topk
def test_perf_gate_topk():
    bench = OpBenchmark(
        op_name="gate_topk",
        torch_op=get_reference("gate_topk"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n, k",
    )
    bench.set_gems(flaggems_sglang.gate_topk)
    bench.run()
