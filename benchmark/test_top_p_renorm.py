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

"""Benchmark for sampling_grammar/top_p_renorm."""

import pytest

import flaggems_sglang
from baseline import top_p_renorm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/sampling_grammar/top_p_renorm.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, vocab = shape
    yield BenchmarkCasePlan(
        shape={"probs": (batch, vocab)},
        params={
            "configuration": shape,
            "seed": 9,
            "_TOP_P": workload._TOP_P,
            "batch": batch,
            "vocab": vocab,
        },
        builder_args=(shape,),
    )


@pytest.mark.top_p_renorm
def test_perf_top_p_renorm():
    bench = OpBenchmark(
        op_name="top_p_renorm",
        torch_op=get_reference("top_p_renorm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, vocab",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
