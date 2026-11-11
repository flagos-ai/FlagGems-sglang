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

"""Benchmark for diffusion/rotary_embedding."""

import pytest

import flaggems_sglang
from baseline import rotary_embedding as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

INTERLEAVED = False


def _build_inputs(plan, cur_dtype, device):
    tokens, heads, head_size = plan.builder_args[0]
    args, kwargs = workload.build_inputs(
        {"tokens": tokens, "heads": heads, "head_size": head_size},
        cur_dtype,
        device,
    )
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    tokens, heads, head_size = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (tokens, heads, head_size),
            "cos": (tokens, head_size // 2),
            "sin": (tokens, head_size // 2),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "INTERLEAVED": INTERLEAVED,
            "tokens": tokens,
            "heads": heads,
            "head_size": head_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.rotary_embedding
def test_perf_rotary_embedding():
    bench = OpBenchmark(
        op_name="rotary_embedding",
        torch_op=get_reference("rotary_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="tokens, heads, head_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
