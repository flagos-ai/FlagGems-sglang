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

"""Benchmark for moe/ep_gather."""

import pytest

import flaggems_sglang
from baseline import ep_gather as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, hidden, topk, num_src = shape
    yield BenchmarkCasePlan(
        shape={
            "recv_x": (num_src, hidden),
            "topk_ids": (num_tokens, topk),
            "out": (num_tokens, hidden),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "num_src": num_src,
        },
        builder_args=(shape,),
    )


@pytest.mark.ep_gather
def test_perf_ep_gather():
    bench = OpBenchmark(
        op_name="ep_gather",
        torch_op=get_reference("ep_gather"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden, topk, num_src",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
