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

"""Benchmark for diffusion/one_pass_rms_norm."""

import pytest

import flaggems_sglang
from baseline import one_pass_rms_norm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

_EPS = 1e-6


def _build_inputs(plan, cur_dtype, device):
    s, d = plan.builder_args[0]
    args, kwargs = workload.build_inputs({"s": s, "d": d}, cur_dtype, device)
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    num_tokens, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (num_tokens, hidden),
            "weight": (hidden,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.one_pass_rms_norm
def test_perf_one_pass_rms_norm():
    bench = OpBenchmark(
        op_name="one_pass_rms_norm",
        torch_op=get_reference("one_pass_rms_norm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
