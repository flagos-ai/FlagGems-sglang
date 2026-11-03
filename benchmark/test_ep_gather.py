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

"""Benchmark for moe/ep_gather."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(t, 4096, 8, max(t * 8, 16)) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 4096, 8, 4096 * 8)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden, topk, num_src = shape
    g = torch.Generator(device=device).manual_seed(9)
    input_tensor = torch.randn(
        num_src, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    ids = torch.randint(
        0,
        32,
        (num_tokens, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    weight = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    index = torch.randint(
        0,
        num_src,
        (num_tokens, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    out = torch.zeros(num_tokens, hidden, dtype=cur_dtype, device=device)
    yield input_tensor, ids, weight, index, out


@pytest.mark.ep_gather
def test_perf_ep_gather():
    bench = OpBenchmark(
        op_name="ep_gather",
        torch_op=get_reference("ep_gather"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden, topk, num_src",
    )
    bench.set_gems(flaggems_sglang.ep_gather)
    bench.run()
