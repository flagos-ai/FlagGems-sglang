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

"""Benchmark for kvcache/reshape_and_cache_flash."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_HEADS = 8
_BLOCK_SIZE = 16
# (tokens, head_size) — 8 heads, block_size 16, no SWA remap and no scale.
SHAPES = [(t, 128) for t in (16, 128, 1024, 4096)]
MORE_SHAPES = [(16, 64), (333, 96), (1024, 128)]


def _input_fn(shape, cur_dtype, device):
    tokens, head_size = shape
    blocks = max(4, tokens // _BLOCK_SIZE + 1)
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*dims):
        return torch.randn(
            *dims, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    total_slots = blocks * _BLOCK_SIZE
    slots = torch.randperm(total_slots, generator=g, device=device)[
        :tokens
    ].to(torch.int64)
    yield dict(
        key=_t(tokens, _HEADS, head_size).contiguous(),
        value=_t(tokens, _HEADS, head_size).contiguous(),
        key_cache=_t(blocks, _BLOCK_SIZE, _HEADS, head_size).contiguous(),
        value_cache=_t(blocks, _BLOCK_SIZE, _HEADS, head_size).contiguous(),
        slot_mapping=slots,
        swa_slot_mapping=None,
        k_scale=None,
        v_scale=None,
    )


@pytest.mark.reshape_and_cache_flash
def test_perf_reshape_and_cache_flash():
    gems_op = flaggems_sglang.get_op("reshape_and_cache_flash")
    if gems_op is None:
        pytest.skip("kvcache/reshape_and_cache_flash not implemented yet")
    bench = OpBenchmark(
        op_name="reshape_and_cache_flash",
        torch_op=get_reference("reshape_and_cache_flash"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, head_size",
    )
    bench.set_gems(gems_op)
    bench.run()
