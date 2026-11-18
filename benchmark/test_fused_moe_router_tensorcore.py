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

"""Benchmark for moe/fused_moe_router_tensorcore."""

import pytest

import flaggems_sglang
from baseline import fused_moe_router_tensorcore as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/moe/fused_moe_router_tensorcore:
# hidden_dim is a multiple of 64 (this problem's BLOCK_SIZE_K choice) and
# topk <= 2 (the tensorcore kernel's supported range).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, hidden_dim, num_experts, topk, softcap, has_bias = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (bs, hidden_dim),
            "router_weight": (num_experts, hidden_dim),
            "bias": (num_experts,) if has_bias else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "bs": bs,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "topk": topk,
            "softcap": softcap,
            "has_bias": has_bias,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "hidden": hidden_dim,
                "num_experts": num_experts,
                "topk": topk,
                "softcap": softcap,
                "has_bias": has_bias,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.fused_moe_router_tensorcore
def test_perf_fused_moe_router_tensorcore():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("fused_moe_router_tensorcore")
        if gems_op is None:
            pytest.skip("moe/fused_moe_router_tensorcore not implemented yet")
    bench = OpBenchmark(
        op_name="fused_moe_router_tensorcore",
        torch_op=get_reference("fused_moe_router_tensorcore"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc=(
            "bs, hidden_dim, num_experts, topk, moe_softcapping, has_bias"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
