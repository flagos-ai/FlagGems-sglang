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

"""Benchmark for activation/gelu_tanh_and_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(bs, d) for bs in (1, 8, 64, 512) for d in (1024, 4096, 8192)]
MORE_SHAPES = [(bs, d) for bs in (4096,) for d in (1024, 4096, 8192)]


def _input_fn(shape, cur_dtype, device):
    bs, d = shape
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        bs, 2 * d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield (hidden_states,)


@pytest.mark.gelu_tanh_and_mul
def test_perf_gelu_tanh_and_mul():
    bench = OpBenchmark(
        op_name="gelu_tanh_and_mul",
        torch_op=get_reference("gelu_tanh_and_mul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, d",
    )
    bench.set_gems(flaggems_sglang.gelu_tanh_and_mul)
    bench.run()
