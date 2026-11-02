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

"""Benchmark for moe/post_reorder_deepgemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/post_reorder_deepgemm.
SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168, 32) for t in (2048,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, topk, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    n = num_tokens * topk
    down = torch.randn(n, hidden, device=device, generator=g).to(cur_dtype)
    ids = torch.randint(
        -1,
        num_experts + 1,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    dst = (
        torch.randperm(
            n, generator=torch.Generator(device="cpu").manual_seed(9)
        )
        .to(device)
        .reshape(num_tokens, topk)
    )
    dst = torch.where(ids >= 0, dst.to(torch.int32), torch.zeros_like(ids))
    output = torch.zeros(num_tokens, hidden, dtype=cur_dtype, device=device)
    weights = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    yield (
        down,
        output,
        dst.to(torch.int32),
        ids,
        weights,
        topk,
        num_tokens,
        hidden,
        1.0,
    )


def _case_fn(shape, cur_dtype):
    (num_tokens, topk, hidden, num_experts) = shape
    yield BenchmarkCasePlan(
        shape={
            "down": (num_tokens * topk, hidden),
            "output": (num_tokens, hidden),
            "dst": (num_tokens, topk),
            "ids": (num_tokens, topk),
            "weights": (num_tokens, topk),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "num_experts": num_experts,
        },
        builder_args=(shape,),
    )


@pytest.mark.post_reorder_deepgemm
def test_perf_post_reorder_deepgemm():
    bench = OpBenchmark(
        op_name="post_reorder_deepgemm",
        torch_op=get_reference("post_reorder_deepgemm"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, topk, hidden, num_experts",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
