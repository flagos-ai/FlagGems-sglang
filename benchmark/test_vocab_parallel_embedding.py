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

"""Benchmark for utility/vocab_parallel_embedding."""

import pytest

import flaggems_sglang
from baseline import vocab_parallel_embedding as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/utility/vocab_parallel_embedding.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, hidden = shape
    yield BenchmarkCasePlan(
        shape={
            "tokens": (num_tokens,),
            "weight": (
                workload._ORG_END
                - workload._ORG_START
                + workload._PADDING
                + workload._ADDED_END
                - workload._ADDED_START,
                hidden,
            ),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_PADDING": workload._PADDING,
            "num_tokens": num_tokens,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.vocab_parallel_embedding
def test_perf_vocab_parallel_embedding():
    bench = OpBenchmark(
        op_name="vocab_parallel_embedding",
        torch_op=get_reference("vocab_parallel_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
