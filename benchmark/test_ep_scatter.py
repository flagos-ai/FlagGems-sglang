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

"""Benchmark for moe/ep_scatter."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(t, 8, 2048, 8) for t in (64, 512)]
MORE_SHAPES = [(4096, 8, 2048, 8)]

_BLOCK_E = 128
_QUANT_BLOCK = 128


def _input_fn(shape, cur_dtype, device):
    t, topk, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    ids = torch.randint(
        0,
        num_experts,
        (t, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    valid = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        valid[e] = int((ids == e).sum())
    padded = ((valid + _BLOCK_E - 1) // _BLOCK_E * _BLOCK_E).to(torch.int32)
    total = int(padded.sum())
    recv_x = torch.randn(
        t, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    recv_x_scale = torch.rand(
        t,
        hidden // _QUANT_BLOCK,
        dtype=torch.float32,
        device=device,
        generator=g,
    )
    yield (
        recv_x,
        recv_x_scale,
        ids,
        padded,
        valid,
        torch.zeros(num_experts, dtype=torch.int32, device=device),
        torch.zeros((total, hidden), dtype=cur_dtype, device=device),
        torch.zeros(
            (total, hidden // _QUANT_BLOCK),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(total, dtype=torch.int32, device=device),
        torch.full((t, topk), -1, dtype=torch.int32, device=device),
    )


@pytest.mark.ep_scatter
def test_perf_ep_scatter():
    bench = OpBenchmark(
        op_name="ep_scatter",
        torch_op=get_reference("ep_scatter"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, topk, hidden, num_experts",
    )
    bench.set_gems(flaggems_sglang.ep_scatter)
    bench.run()
