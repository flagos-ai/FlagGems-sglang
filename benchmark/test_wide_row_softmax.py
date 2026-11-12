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

"""Benchmark for speculative/wide_row_softmax."""

import pytest

import flaggems_sglang
from baseline import wide_row_softmax as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/wide_row_softmax.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, vocab = shape
    yield BenchmarkCasePlan(
        shape={"logits": (bs, vocab), "out": (bs, vocab)},
        params={"configuration": shape, "seed": 9, "bs": bs, "vocab": vocab},
        builder_args=(shape,),
    )


@pytest.mark.wide_row_softmax
def test_perf_wide_row_softmax():
    bench = OpBenchmark(
        op_name="wide_row_softmax",
        torch_op=get_reference("wide_row_softmax"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, vocab",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
