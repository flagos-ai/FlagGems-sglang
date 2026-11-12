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

"""Benchmark for moe/moe_fused_mul_sum."""

import pytest

import flaggems_sglang
from baseline import moe_fused_mul_sum as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/moe/moe_fused_mul_sum.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    (
        num_tokens,
        top_k,
        size,
        is_ep,
        use_expert_map,
        num_experts,
        routed_scaling_factor,
    ) = shape
    yield BenchmarkCasePlan(
        shape={
            "inputs": (num_tokens, top_k, size),
            "topk_weights": (num_tokens, top_k),
            "topk_ids": (
                (num_tokens, top_k) if use_expert_map or is_ep else None
            ),
            "expert_map": (num_experts,) if use_expert_map else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "size": size,
            "is_ep": is_ep,
            "use_expert_map": use_expert_map,
            "num_experts": num_experts,
            "routed_scaling_factor": routed_scaling_factor,
        },
        baseline_key=workload.key_shape(
            {
                "num_tokens": shape[0],
                "top_k": shape[1],
                "size": shape[2],
                "is_ep": shape[3],
                "use_expert_map": shape[4],
                "num_experts": shape[5],
                "routed_scaling_factor": shape[6],
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.moe_fused_mul_sum
def test_perf_moe_fused_mul_sum():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("moe_fused_mul_sum")
        if gems_op is None:
            pytest.skip("moe/moe_fused_mul_sum not implemented yet")
    bench = OpBenchmark(
        op_name="moe_fused_mul_sum",
        torch_op=get_reference("moe_fused_mul_sum"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc=(
            "num_tokens, top_k, size, is_ep, use_expert_map, num_experts, "
            "routed_scaling_factor"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
