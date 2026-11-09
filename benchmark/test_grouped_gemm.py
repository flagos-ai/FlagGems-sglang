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

"""Benchmark for gemm/grouped_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 512, 2048, e) for m in (128, 1024) for e in (8, 32)]
MORE_SHAPES = [(4096, 512, 2048, e) for e in (8, 32)]

_BLOCK_SIZE_M = 128


def _metadata(sorted_topk_ids, num_experts, block_size_m=_BLOCK_SIZE_M):
    """The torch equivalent of SGLang's grouped-gemm metadata; see the test
    module for the full commentary."""
    device = sorted_topk_ids.device
    num_routed_tokens = sorted_topk_ids.numel()
    offs = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    offs[0] = 0
    ids = torch.arange(num_experts, device=device, dtype=torch.int32)
    offs[1:] = torch.searchsorted(sorted_topk_ids, ids, side="right").to(
        torch.int32
    )
    counts = (offs[1:] - offs[:-1]).contiguous()
    blocks = (counts + block_size_m - 1) // block_size_m
    block_offs = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    block_offs[0] = 0
    block_offs[1:num_experts] = torch.cumsum(blocks, dim=0)[: num_experts - 1]
    block_offs[num_experts] = int(blocks.sum())
    max_blocks = (
        (num_routed_tokens + block_size_m - 1) // block_size_m
        + num_experts
        - 1
    )
    schedule = torch.full((max_blocks,), -1, device=device, dtype=torch.int32)
    for e in range(num_experts):
        nb = int(blocks[e])
        off = int(block_offs[e])
        if nb > 0:
            local = torch.arange(nb, device=device, dtype=torch.int32)
            schedule[off : off + nb] = (local << 16) + e
    return counts, offs, block_offs, schedule


def _input_fn(shape, cur_dtype, device):
    m, n, k, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    a = torch.randn(m, k, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    b = (
        0.1
        * torch.randn(
            num_experts,
            n,
            k,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).to(cur_dtype)
    ids = torch.randint(
        0, num_experts, (m,), device=device, generator=g, dtype=torch.int32
    )
    counts, offs, block_offs, schedule = _metadata(
        torch.sort(ids)[0].contiguous(), num_experts
    )
    yield a, b, num_experts, counts, offs, block_offs, schedule


@pytest.mark.grouped_gemm
def test_perf_grouped_gemm():
    bench = OpBenchmark(
        op_name="grouped_gemm",
        torch_op=get_reference("grouped_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, N, K, num_experts",
    )
    bench.set_gems(flaggems_sglang.grouped_gemm)
    bench.run()
