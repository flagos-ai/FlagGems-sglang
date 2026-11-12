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

"""Benchmark for rope/interleaved_rope."""

import pytest

import flaggems_sglang
from baseline import interleaved_rope as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/rope/interleaved_rope. The last
# entry is the mrope section split.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    s, d, mrope_section = shape
    yield BenchmarkCasePlan(
        shape={"x": (3, s, d)},
        params={
            "configuration": shape,
            "seed": 0,
            "s": s,
            "d": d,
            "mrope_section": mrope_section,
        },
        builder_args=(shape,),
    )


@pytest.mark.interleaved_rope
def test_perf_interleaved_rope():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("interleaved_rope")
        if gems_op is None:
            pytest.skip("rope/interleaved_rope not implemented yet")
    bench = OpBenchmark(
        op_name="interleaved_rope",
        torch_op=get_reference("interleaved_rope"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="s, d, mrope_section",
    )
    bench.set_gems(gems_op)
    bench.run()
