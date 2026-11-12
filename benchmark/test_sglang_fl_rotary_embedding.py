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

"""Benchmark for sglang/sglang_fl_rotary_embedding."""

import pytest

import flaggems_sglang
from baseline import sglang_fl_rotary_embedding as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# (num_tokens, num_q_heads, num_kv_heads, head_size).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, num_q_heads, num_kv_heads, head_size = shape
    yield BenchmarkCasePlan(
        shape={
            "query": (num_tokens, num_q_heads * head_size),
            "key": (num_tokens, num_kv_heads * head_size),
            "cos_sin_cache": (workload._MAX_POSITION, head_size),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
        },
        builder_args=(shape,),
        baseline_key=workload.key_shape(workload.binding_for_case(shape)),
    )


@pytest.mark.sglang_fl_rotary_embedding
def test_perf_sglang_fl_rotary_embedding():
    bench = OpBenchmark(
        op_name="sglang_fl_rotary_embedding",
        torch_op=get_reference("sglang_fl_rotary_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, num_q_heads, num_kv_heads, head_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
