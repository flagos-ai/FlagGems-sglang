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

"""Benchmark for moe/fused_append_shared_experts."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 8, 1) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(16384, 8, 1)]

_N_BASE = 256


def _input_fn(shape, cur_dtype, device):
    m, k, s = shape
    g = torch.Generator(device=device).manual_seed(9)
    topk_ids = torch.randint(
        0, _N_BASE, (m, k), device=device, generator=g, dtype=torch.int32
    )
    topk_weights = torch.rand(
        m, k, dtype=torch.float32, device=device, generator=g
    )
    yield topk_ids, topk_weights, s, 0.25, _N_BASE


@pytest.mark.fused_append_shared_experts
def test_perf_fused_append_shared_experts():
    bench = OpBenchmark(
        op_name="fused_append_shared_experts",
        torch_op=get_reference("fused_append_shared_experts"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, topk, num_shared_experts",
    )
    bench.set_gems(flaggems_sglang.fused_append_shared_experts)
    bench.run()
