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

"""Benchmark for moe/topk_sigmoid."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/topk_sigmoid.
SHAPES = [(t, 256, 8) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 256, 8) for t in (4096,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_experts, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    # fp32 logits keep the top-k ordering well defined; see the test module.
    gating = torch.randn(num_tokens, num_experts, device=device, generator=g)
    topk_weights = torch.zeros(
        num_tokens, k, dtype=torch.float32, device=device
    )
    topk_ids = torch.zeros(num_tokens, k, dtype=torch.int32, device=device)
    # renormalize is a bool, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    yield topk_weights, topk_ids, gating, {
        "renormalize": True,
        "routed_scaling_factor": 2.5,
    }


def _case_fn(shape, cur_dtype):
    (num_tokens, num_experts, k) = shape
    yield BenchmarkCasePlan(
        shape={
            "topk_weights": (num_tokens, k),
            "topk_ids": (num_tokens, k),
            "gating": (num_tokens, num_experts),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "k": k,
        },
        builder_args=(shape,),
    )


@pytest.mark.topk_sigmoid
def test_perf_topk_sigmoid():
    bench = OpBenchmark(
        op_name="topk_sigmoid",
        torch_op=get_reference("topk_sigmoid"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_experts, k",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
