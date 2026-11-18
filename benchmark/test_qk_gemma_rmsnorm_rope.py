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

"""Benchmark for rope/qk_gemma_rmsnorm_rope."""

import pytest

import flaggems_sglang
from baseline import qk_gemma_rmsnorm_rope as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (num_tokens, num_q_heads * head_dim),
            "k": (num_tokens, num_kv_heads * head_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.qk_gemma_rmsnorm_rope
def test_perf_qk_gemma_rmsnorm_rope():
    bench = OpBenchmark(
        op_name="qk_gemma_rmsnorm_rope",
        torch_op=get_reference("qk_gemma_rmsnorm_rope"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
