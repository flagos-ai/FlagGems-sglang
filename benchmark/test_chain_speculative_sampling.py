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

"""Benchmark for sampling_grammar/chain_speculative_sampling."""

import pytest

import flaggems_sglang
from baseline import chain_speculative_sampling as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/sampling_grammar/chain_speculative_sampling.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    B, S, V = shape
    yield BenchmarkCasePlan(
        shape={
            "candidates": (B, S),
            "retrive_index": (B, S),
            "uniform_samples": (B, S - 1),
            "uniform_samples_for_final_sampling": (B,),
            "target_probs": (B, S, V),
            "draft_probs": (B, S - 1, V),
        },
        params={"configuration": shape, "seed": 0, "B": B, "S": S, "V": V},
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.chain_speculative_sampling
def test_perf_chain_speculative_sampling():
    bench = OpBenchmark(
        op_name="chain_speculative_sampling",
        torch_op=get_reference("chain_speculative_sampling"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="B, S, V",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
