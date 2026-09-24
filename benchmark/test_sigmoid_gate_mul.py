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

"""Benchmark for activation/sigmoid_gate_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (1024, 4096, 8192)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (1024, 4096, 8192)]


def _input_fn(shape, cur_dtype, device):
    bs, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    gate = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    up = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield gate, up


@pytest.mark.sigmoid_gate_mul
def test_perf_sigmoid_gate_mul():
    bench = OpBenchmark(
        op_name="sigmoid_gate_mul",
        torch_op=get_reference("sigmoid_gate_mul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, hidden",
    )
    bench.set_gems(flaggems_sglang.sigmoid_gate_mul)
    bench.run()
