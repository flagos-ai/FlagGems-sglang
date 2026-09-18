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

"""Benchmark for activation_norm/gelu_and_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/gelu_and_mul.
SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 4096, 8192)]
MORE_SHAPES = [(7, 16), (83, 1024), (48, 3072), (1, 8192)]


def _input_fn(shape, cur_dtype, device):
    bs, d = shape
    # The op reads x1/x3 from a single [bs, 2 * d] tensor.
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        bs, 2 * d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield (hidden_states,)


@pytest.mark.gelu_and_mul
def test_perf_gelu_and_mul():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("gelu_and_mul")
    if gems_op is None:
        pytest.skip("activation_norm/gelu_and_mul not implemented yet")
    bench = OpBenchmark(
        op_name="gelu_and_mul",
        torch_op=get_reference("gelu_and_mul"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, d",
    )
    bench.set_gems(gems_op)
    bench.run()
