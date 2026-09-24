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

"""Benchmark for moe/fused_moe_dispatch_index."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(t, 8, 32, 8192) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 32, 8192) for t in (4096,)]


def _input_fn(shape, cur_dtype, device):
    total_tokens, topk, num_experts, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    expert_ids = torch.randint(
        0,
        num_experts,
        (total_tokens, topk),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    num_tokens_post_pad = torch.full(
        (num_experts,), total_tokens, dtype=torch.int64, device=device
    )
    yield expert_ids, num_tokens_post_pad


@pytest.mark.fused_moe_dispatch_index
def test_perf_fused_moe_dispatch_index():
    bench = OpBenchmark(
        op_name="fused_moe_dispatch_index",
        torch_op=get_reference("fused_moe_dispatch_index"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="total_tokens, topk, num_experts, hidden",
    )
    bench.set_gems(flaggems_sglang.fused_moe_dispatch_index)
    bench.run()
