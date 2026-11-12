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

"""Benchmark for utility/set_token_filter."""

import pytest

import flaggems_sglang
from baseline import set_token_filter as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/utility/set_token_filter.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, vocab, num_tokens = shape
    yield BenchmarkCasePlan(
        shape={"vocab_mask": (batch, (vocab + 31) // 32)},
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "vocab": vocab,
            "num_tokens": num_tokens,
        },
        baseline_key=workload.key_shape(
            {
                "batch": batch,
                "vocab": vocab,
                "num_tokens": num_tokens,
                **{"is_allowed": True, "reset": True},
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.set_token_filter
def test_perf_set_token_filter():
    bench = OpBenchmark(
        op_name="set_token_filter",
        torch_op=get_reference("set_token_filter"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, vocab, num_tokens",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
