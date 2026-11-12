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

"""Benchmark for sampling/gemma4_fused_routing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, experts, topk). Decode-like batches stress launch overhead;
# prefill batches stress the per-row argmax chain.
SHAPES = [
    (1, 256, 8),
    (128, 128, 8),
    (2048, 256, 8),
]

MORE_SHAPES = [
    (8192, 128, 6),
    (4096, 128, 8),
    (1024, 64, 2),
]


def _input_fn(shape, cur_dtype, device):
    tokens, experts, topk = shape
    g = torch.Generator(device=device).manual_seed(0)
    gating = torch.randn(
        (tokens, experts),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    scale = (
        0.5 * torch.rand(experts, generator=g, device=device,
                         dtype=torch.float32)
        + 0.5
    ).to(cur_dtype)
    yield gating, scale, topk


def _case_fn(shape, cur_dtype):
    tokens, experts, topk = shape
    yield BenchmarkCasePlan(
        shape={"gating_output": (tokens, experts)},
        params={
            "configuration": shape,
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
        },
        builder_args=(shape,),
    )


@pytest.mark.gemma4_fused_routing
def test_perf_gemma4_fused_routing():
    bench = OpBenchmark(
        op_name="gemma4_fused_routing",
        torch_op=get_reference("gemma4_fused_routing"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, experts, topk",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
