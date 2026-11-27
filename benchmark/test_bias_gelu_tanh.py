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

"""Benchmark for elementwise/bias_gelu_tanh."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

# (num_tokens, hidden) — H % 8 == 0.
SHAPES = [(1, 128), (7, 512), (128, 4096), (1024, 4096)]
MORE_SHAPES = [(33, 1024), (512, 8192), (2048, 2048)]


def _input_fn(shape, cur_dtype, device):
    g = create_generator(device, seed=0)
    x = torch.randn(
        *shape, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    bias = torch.randn(
        shape[-1], dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield dict(x=x, bias=bias)


@pytest.mark.bias_gelu_tanh
def test_perf_bias_gelu_tanh():
    gems_op = flaggems_sglang.get_op("bias_gelu_tanh")
    if gems_op is None:
        pytest.skip("elementwise/bias_gelu_tanh not implemented yet")
    bench = OpBenchmark(
        op_name="bias_gelu_tanh",
        torch_op=get_reference("bias_gelu_tanh"),
        input_fn=_input_fn,
        dtypes=[torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems(gems_op)
    bench.run()
