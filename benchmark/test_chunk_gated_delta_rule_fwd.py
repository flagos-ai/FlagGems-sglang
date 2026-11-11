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

"""Benchmark for linear-attn/chunk_gated_delta_rule_fwd."""

import pytest

import flaggems_sglang
from baseline import chunk_gated_delta_rule_fwd as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# (B, T, H, K, V).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    B, T, H, K, V = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (B, T, H, K),
            "k": (B, T, H, K),
            "v": (B, T, H, V),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "B": B,
            "T": T,
            "H": H,
            "K": K,
            "V": V,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.chunk_gated_delta_rule_fwd
def test_perf_chunk_gated_delta_rule_fwd():
    bench = OpBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=get_reference("chunk_gated_delta_rule_fwd"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="B, T, H, K, V",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
