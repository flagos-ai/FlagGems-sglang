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

"""Benchmark for elementwise/add_constant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/elementwise/add_constant.
SHAPES = [(1 << 12,), (1 << 16,), (1 << 20,)]
MORE_SHAPES = [(1 << 24,)]


def _input_fn(shape, cur_dtype, device):
    (n,) = shape
    g = torch.Generator(device=device).manual_seed(9)
    src = torch.randint(
        -(2**20),
        2**20,
        (n,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield src, 42


def _case_fn(shape, cur_dtype):
    (n,) = shape
    yield BenchmarkCasePlan(
        shape={"src": (n,)},
        params={"configuration": shape, "seed": 9, "n": n},
        builder_args=(shape,),
    )


@pytest.mark.add_constant
def test_perf_add_constant():
    bench = OpBenchmark(
        op_name="add_constant",
        torch_op=get_reference("add_constant"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
