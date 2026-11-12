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

"""Benchmark for sampling_grammar/apply_token_bitmask."""

import pytest

import flaggems_sglang
from baseline import apply_token_bitmask as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/sampling_grammar/apply_token_bitmask.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    B, V = shape
    yield BenchmarkCasePlan(
        shape={"logits": (B, V), "bitmask": (B, (V + 31) // 32)},
        params={"configuration": shape, "seed": 0, "B": B, "V": V},
        builder_args=(shape,),
    )


@pytest.mark.apply_token_bitmask
def test_perf_apply_token_bitmask():
    bench = OpBenchmark(
        op_name="apply_token_bitmask",
        torch_op=get_reference("apply_token_bitmask"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        # The op masks logits in place of a softmax, so it runs in fp32.
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="B, V",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
