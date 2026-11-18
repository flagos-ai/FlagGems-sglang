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

import pytest

import flaggems_sglang
from baseline import fused_sigmoid_mul as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/activation_norm/fused_sigmoid_mul.
# The strided path (3D gate over a 2D attn_output) is the production shape,
# so the benchmark sweeps num_tokens with a fixed 32 x 128 head layout.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, num_heads, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "attn_output": (num_tokens, num_heads * head_dim),
            "gate": (num_tokens, num_heads, head_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_sigmoid_mul
def test_perf_fused_sigmoid_mul():
    bench = OpBenchmark(
        op_name="fused_sigmoid_mul",
        torch_op=get_reference("fused_sigmoid_mul"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
