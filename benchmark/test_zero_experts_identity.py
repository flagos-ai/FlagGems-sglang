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

"""Benchmark for moe/zero_experts_identity."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/zero_experts_identity.
SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168, 32) for t in (4096,)]

_NUM_ZERO = 8


def _input_fn(shape, cur_dtype, device):
    num_tokens, top_k, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    idx = torch.randint(
        0,
        num_experts + _NUM_ZERO,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    scales = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device=device, generator=g
    )
    h = torch.randn(num_tokens, hidden, device=device, generator=g).to(
        cur_dtype
    )
    # zero_expert_type is a str, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    yield idx, scales, num_experts, {
        "zero_expert_type": "identity",
        "hidden_states": h,
    }


def _case_fn(shape, cur_dtype):
    (num_tokens, top_k, hidden, num_experts) = shape
    yield BenchmarkCasePlan(
        shape={
            "idx": (num_tokens, top_k),
            "scales": (num_tokens, top_k),
            "hidden_states": (num_tokens, hidden),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_NUM_ZERO": _NUM_ZERO,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "hidden": hidden,
            "num_experts": num_experts,
        },
        builder_args=(shape,),
    )


@pytest.mark.zero_experts_identity
def test_perf_zero_experts_identity():
    bench = OpBenchmark(
        op_name="zero_experts_identity",
        torch_op=get_reference("zero_experts_identity"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, top_k, hidden, num_experts",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
