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

"""Benchmark for elementwise/add3."""

import pytest

import flaggems_sglang
from baseline import add3 as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    bs, hidden = plan.builder_args[0]
    args, kwargs = workload.build_inputs(
        {"bs": bs, "hidden": hidden}, cur_dtype, device
    )
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    num_tokens, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "a": (num_tokens, hidden),
            "b": (num_tokens, hidden),
            "c": (num_tokens, hidden),
        },
        params={
            "configuration": shape,
            "seeds": [0, 1, 2],
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.add3
def test_perf_add3():
    bench = OpBenchmark(
        op_name="add3",
        torch_op=get_reference("add3"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
