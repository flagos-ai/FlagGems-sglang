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

"""Benchmark for moe/pack_topk_ids."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/pack_topk_ids.
SHAPES = [(t, 8, 256) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(t, 8, 256) for t in (16384,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, topk, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    weights = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    yield ids, weights


def _case_fn(shape, cur_dtype):
    (num_tokens, topk, num_experts) = shape
    yield BenchmarkCasePlan(
        shape={"ids": (num_tokens, topk), "weights": (num_tokens, topk)},
        params={
            "configuration": shape,
            "seed": 9,
            "num_tokens": num_tokens,
            "topk": topk,
            "num_experts": num_experts,
        },
        builder_args=(shape,),
    )


@pytest.mark.pack_topk_ids
def test_perf_pack_topk_ids():
    bench = OpBenchmark(
        op_name="pack_topk_ids",
        torch_op=get_reference("pack_topk_ids"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, topk, num_experts",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
