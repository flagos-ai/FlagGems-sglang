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

"""Benchmark for moe/fused_moe."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (num_tokens, hidden, inter, num_experts, topk).
SHAPES = [
    (1, 2048, 1408, 64, 8),
    (8, 2048, 1408, 64, 8),
    (64, 2048, 1408, 64, 8),
    (512, 2048, 1408, 64, 8),
]
MORE_SHAPES = [(4096, 2048, 1408, 64, 8)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden, inter, num_experts, topk = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(num_tokens, hidden, device=device, generator=g).to(
        cur_dtype
    )
    w1 = (
        torch.randn(num_experts, 2 * inter, hidden, device=device, generator=g)
        * 0.05
    ).to(cur_dtype)
    w2 = (
        torch.randn(num_experts, hidden, inter, device=device, generator=g)
        * 0.05
    ).to(cur_dtype)
    logits = torch.randn(num_tokens, num_experts, device=device, generator=g)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_output = (
        topk_weights.float().contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        None,
    )
    yield x, w1, w2, topk_output, {
        "activation": "silu",
        "routed_scaling_factor": 1.0,
    }


def _case_fn(shape, cur_dtype):
    num_tokens, hidden, inter, num_experts, topk = shape
    yield BenchmarkCasePlan(
        shape={
            "hidden_states": (num_tokens, hidden),
            "w1": (num_experts, 2 * inter, hidden),
            "w2": (num_experts, hidden, inter),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "inter": inter,
            "num_experts": num_experts,
            "topk": topk,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_moe
def test_perf_fused_moe():
    bench = OpBenchmark(
        op_name="fused_moe",
        torch_op=get_reference("fused_moe"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden, inter, num_experts, topk",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
