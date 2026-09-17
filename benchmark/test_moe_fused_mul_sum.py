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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark


def _shape(
    num_tokens,
    top_k,
    size,
    is_ep=False,
    use_expert_map=False,
    num_experts=8,
    routed_scaling_factor=None,
):
    return (
        num_tokens,
        top_k,
        size,
        is_ep,
        use_expert_map,
        num_experts,
        routed_scaling_factor,
    )


# Shapes match kernel-comp-baseline/problems/moe/moe_fused_mul_sum.
SHAPES = [_shape(m, 8, 4096) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    _shape(1, 2, 128),
    _shape(37, 4, 256, routed_scaling_factor=2.5),
    _shape(83, 6, 512, is_ep=True, num_experts=16),
    _shape(64, 4, 256, use_expert_map=True, num_experts=16),
]


def _input_fn(shape, cur_dtype, device):
    (
        num_tokens,
        top_k,
        size,
        is_ep,
        use_expert_map,
        num_experts,
        routed_scaling_factor,
    ) = shape
    g = torch.Generator(device=device).manual_seed(0)
    inputs = torch.randn(
        num_tokens,
        top_k,
        size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)

    topk_ids = None
    expert_map = None
    if use_expert_map or is_ep:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            generator=g,
            device=device,
            dtype=torch.int32,
        )
        if is_ep and not use_expert_map:
            # is_ep (no expert_map): the kernel checks `id_val >= 0` directly,
            # so -1 sentinels in topk_ids are a valid "already dropped" marker.
            drop = (
                torch.rand(num_tokens, top_k, generator=g, device=device) < 0.3
            )
            topk_ids = torch.where(
                drop, torch.full_like(topk_ids, -1), topk_ids
            )
    if use_expert_map:
        # has_expert_map path indexes `expert_map[topk_ids]` with no id_val>=0
        # guard, so topk_ids must stay valid (dropping is expressed entirely
        # via expert_map's own -1 entries, never via a -1 topk_id).
        expert_map = torch.arange(
            num_experts, device=device, dtype=torch.int32
        )
        expert_map[num_experts // 2 :] = -1

    # ``is_ep`` is a bool, which unpack_to_args_kwargs would append as a
    # positional int, so route the trailing flags through kwargs.
    yield inputs, topk_weights, topk_ids, expert_map, dict(
        routed_scaling_factor=routed_scaling_factor,
        is_ep=is_ep,
    )


@pytest.mark.moe_fused_mul_sum
def test_perf_moe_fused_mul_sum():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("moe_fused_mul_sum")
    if gems_op is None:
        pytest.skip("moe/moe_fused_mul_sum not implemented yet")
    bench = OpBenchmark(
        op_name="moe_fused_mul_sum",
        torch_op=get_reference("moe_fused_mul_sum"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "num_tokens, top_k, size, is_ep, use_expert_map, num_experts, "
            "routed_scaling_factor"
        ),
    )
    bench.set_gems(gems_op)
    bench.run()
