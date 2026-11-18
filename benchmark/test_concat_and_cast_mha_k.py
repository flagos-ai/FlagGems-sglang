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

"""Benchmark for kvcache/concat_and_cast_mha_k."""

import pytest

import flaggems_sglang
from baseline import concat_and_cast_mha_k as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/concat_and_cast_mha_k.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, num_heads, nope_dim, rope_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "k": (num_tokens, num_heads, nope_dim + rope_dim),
            "k_nope": (num_tokens, num_heads, nope_dim),
            "k_rope": (num_tokens, 1, rope_dim),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.concat_and_cast_mha_k
def test_perf_concat_and_cast_mha_k():
    bench = OpBenchmark(
        op_name="concat_and_cast_mha_k",
        torch_op=get_reference("concat_and_cast_mha_k"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=workload.MUTATED_INPUTS,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, num_heads, nope_dim, rope_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
