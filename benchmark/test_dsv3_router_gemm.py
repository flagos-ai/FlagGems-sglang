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

"""Benchmark for gemm/dsv3_router_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_HIDDEN = 7168
# (num_tokens, num_experts) — num_tokens stays in [1, 16], the skinny-M
# regime this operator exists for.
SHAPES = [(m, e) for m in (1, 2, 4, 8, 16) for e in (256, 384)]
MORE_SHAPES = [(1, 128), (16, 512), (8, 64)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_experts = shape
    g = create_generator(device, seed=0)
    h = torch.randn(num_tokens, _HIDDEN, device=device, generator=g).to(
        cur_dtype
    )
    w = torch.randn(num_experts, _HIDDEN, device=device, generator=g).to(
        cur_dtype
    )
    yield dict(hidden_states=h, router_weights=w, out_dtype=torch.float32)


@pytest.mark.dsv3_router_gemm
def test_perf_dsv3_router_gemm():
    gems_op = flaggems_sglang.get_op("dsv3_router_gemm")
    if gems_op is None:
        pytest.skip("gemm/dsv3_router_gemm not implemented yet")
    bench = OpBenchmark(
        op_name="dsv3_router_gemm",
        torch_op=get_reference("dsv3_router_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_experts",
    )
    bench.set_gems(gems_op)
    bench.run()
