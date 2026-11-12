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

"""Benchmark for attention/merge_state."""

import pytest

import flaggems_sglang
from baseline import merge_state as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    n_tokens, num_heads, head_size = plan.builder_args[0]
    args, kwargs = workload.build_inputs(
        {"n_tokens": n_tokens, "num_heads": num_heads, "head_size": head_size},
        cur_dtype,
        device,
    )
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    n_tokens, num_heads, head_size = shape
    yield BenchmarkCasePlan(
        shape={
            "prefix_output": (n_tokens, num_heads, head_size),
            "prefix_lse": (n_tokens, num_heads),
            "suffix_output": (n_tokens, num_heads, head_size),
            "suffix_lse": (n_tokens, num_heads),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "n_tokens": n_tokens,
            "num_heads": num_heads,
            "head_size": head_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.merge_state
def test_perf_merge_state():
    bench = OpBenchmark(
        op_name="merge_state",
        torch_op=get_reference("merge_state"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="n_tokens, num_heads, head_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
