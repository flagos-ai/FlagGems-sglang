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

"""Benchmark for diffusion/fused_qk_inv_rms."""

import pytest

import flaggems_sglang
from baseline import fused_qk_inv_rms as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    batch, num_tokens, heads, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (batch, num_tokens, heads, head_dim),
            "k": (batch, num_tokens, heads, head_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "batch": batch,
            "num_tokens": num_tokens,
            "heads": heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_qk_inv_rms
def test_perf_fused_qk_inv_rms():
    bench = OpBenchmark(
        op_name="fused_qk_inv_rms",
        torch_op=get_reference("fused_qk_inv_rms"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="batch, num_tokens, heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
