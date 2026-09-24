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

"""Benchmark for diffusion/residual_gate_add."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [
    (rows, 3072, broadcast) for rows in (1, 64, 1024) for broadcast in (0, 1)
]
MORE_SHAPES = [
    (rows, 3072, broadcast) for rows in (16384,) for broadcast in (0, 1)
]


def _input_fn(shape, cur_dtype, device):
    rows, hidden, broadcast = shape
    broadcast = bool(broadcast)
    g = torch.Generator(device=device).manual_seed(0)

    def r(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    gate = r(1, hidden) if broadcast else r(rows, hidden)
    residual = r(rows, hidden)
    update = r(rows, hidden)
    yield residual, update, gate


@pytest.mark.residual_gate_add
def test_perf_residual_gate_add():
    bench = OpBenchmark(
        op_name="residual_gate_add",
        torch_op=get_reference("residual_gate_add"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, hidden, broadcast",
    )
    bench.set_gems(flaggems_sglang.residual_gate_add)
    bench.run()
