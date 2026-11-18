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

"""Benchmark for speculative/get_target_cache_loc."""

import pytest

import flaggems_sglang
from baseline import get_target_cache_loc as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/get_target_cache_loc.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, num_verify_tokens = shape
    yield BenchmarkCasePlan(
        shape={
            "num_correct_drafts": (bs,),
            "to_free_num_slots": (bs,),
            "out_cache_loc": (bs * num_verify_tokens,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "bs": bs,
            "num_verify_tokens": num_verify_tokens,
        },
        builder_args=(shape,),
    )


@pytest.mark.get_target_cache_loc
def test_perf_get_target_cache_loc():
    bench = OpBenchmark(
        op_name="get_target_cache_loc",
        torch_op=get_reference("get_target_cache_loc"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, num_verify_tokens",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
