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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _shape(
    m,
    n,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    return (
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
    )


# Shapes match kernel-comp-baseline/problems/moe/moe_fused_gate.
SHAPES = [
    _shape(
        m, 256, 8, num_fused_shared_experts=1, num_expert_group=8, topk_group=4
    )
    for m in (1, 8, 64, 512, 4096)
]
MORE_SHAPES = [
    _shape(1, 8, 2),
    _shape(
        37,
        256,
        8,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.5,
        num_expert_group=8,
        topk_group=4,
    ),
    _shape(83, 64, 6, scoring_func="sqrtsoftplus"),
    _shape(
        17,
        32,
        4,
        scoring_func="softmax",
        moe_softcapping=30.0,
        renormalize=False,
    ),
    _shape(
        9,
        128,
        6,
        apply_routed_scaling_factor_on_output=True,
        routed_scaling_factor=1.7,
    ),
]


def _input_fn(shape, cur_dtype, device):
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
    g = torch.Generator(device=device).manual_seed(0)
    scores = torch.randn(
        m, n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # The bias stays fp32; the reference upcasts both operands anyway.
    bias = 0.1 * torch.randn(
        n, generator=g, device=device, dtype=torch.float32
    )
    # ``scoring_func`` is a str, which unpack_to_args_kwargs would drop from
    # the positional args, so route it (and the rest of the gate config)
    # through kwargs.
    yield scores, bias, topk, dict(
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            apply_routed_scaling_factor_on_output
        ),
        moe_softcapping=moe_softcapping,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )


@pytest.mark.moe_fused_gate
def test_perf_moe_fused_gate():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("moe_fused_gate")
    if gems_op is None:
        pytest.skip("moe/moe_fused_gate not implemented yet")
    bench = OpBenchmark(
        op_name="moe_fused_gate",
        torch_op=get_reference("moe_fused_gate"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "m, n, topk, scoring_func, num_fused_shared_experts, "
            "renormalize, routed_scaling_factor, "
            "apply_routed_scaling_factor_on_output, moe_softcapping, "
            "num_expert_group, topk_group"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
