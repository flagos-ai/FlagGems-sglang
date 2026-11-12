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

"""Benchmark for moe/moe_fused_gate."""

import pytest

import flaggems_sglang
from baseline import moe_fused_gate as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/moe/moe_fused_gate.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    (
        m,
        n,
        topk,
        scoring_func,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
        moe_softcapping,
        num_expert_group,
        topk_group,
    ) = shape
    yield BenchmarkCasePlan(
        shape={"scores": (m, n), "bias": (n,)},
        params={
            "configuration": shape,
            "seed": 0,
            "m": m,
            "n": n,
            "topk": topk,
            "scoring_func": scoring_func,
            "num_fused_shared_experts": num_fused_shared_experts,
            "renormalize": renormalize,
            "routed_scaling_factor": routed_scaling_factor,
            "apply_routed_scaling_factor_on_output": apply_routed_scaling_factor_on_output,
            "moe_softcapping": moe_softcapping,
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
        },
        baseline_key=workload.key_shape(
            {
                "m": shape[0],
                "n": shape[1],
                "topk": shape[2],
                "scoring_func": shape[3],
                "num_fused_shared_experts": shape[4],
                "renormalize": shape[5],
                "routed_scaling_factor": shape[6],
                "apply_routed_scaling_factor_on_output": shape[7],
                "moe_softcapping": shape[8],
                "num_expert_group": shape[9],
                "topk_group": shape[10],
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.moe_fused_gate
def test_perf_moe_fused_gate():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("moe_fused_gate")
        if gems_op is None:
            pytest.skip("moe/moe_fused_gate not implemented yet")
    bench = OpBenchmark(
        op_name="moe_fused_gate",
        torch_op=get_reference("moe_fused_gate"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc=(
            "m, n, topk, scoring_func, num_fused_shared_experts, "
            "renormalize, routed_scaling_factor, "
            "apply_routed_scaling_factor_on_output, moe_softcapping, "
            "num_expert_group, topk_group"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
