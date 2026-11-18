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

"""Benchmark for attention/compute_position."""

import pytest

import flaggems_sglang
from baseline import compute_position as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/compute_position.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, max_len = shape
    yield BenchmarkCasePlan(
        shape={"prefix": (bs,), "seq_lens": (bs,)},
        params={
            "configuration": shape,
            "seed": 7,
            "bs": bs,
            "max_len": max_len,
        },
        builder_args=(shape,),
    )


@pytest.mark.compute_position
def test_perf_compute_position():
    bench = OpBenchmark(
        op_name="compute_position",
        torch_op=get_reference("compute_position"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, max_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
