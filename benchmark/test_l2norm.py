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

"""Benchmark for activation_norm/l2norm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/l2norm.
SHAPES = [(t, d) for t in (1, 32, 512, 4096) for d in (64, 128, 256)]
MORE_SHAPES = [(1, 64), (83, 128), (256, 256), (3, 1024)]


def _input_fn(shape, cur_dtype, device):
    t, d = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    yield (x,)


def _case_fn(shape, cur_dtype):
    (t, d) = shape
    yield BenchmarkCasePlan(
        shape={"x": (t, d)},
        params={"configuration": shape, "seed": 0, "t": t, "d": d},
        builder_args=(shape,),
    )


@pytest.mark.l2norm
def test_perf_l2norm():
    bench = OpBenchmark(
        op_name="l2norm",
        torch_op=get_reference("l2norm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
