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

"""Benchmark for reshape_and_cache_flash."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (num_tokens, num_heads, head_size, num_blocks, block_size)
SHAPES = [
    (1024, 8, 128, 64, 16),
    (4096, 8, 128, 256, 16),
    (2048, 32, 128, 128, 16),
]

MORE_SHAPES = [
    (256, 4, 64, 8, 32),
    (8192, 8, 128, 512, 16),
    (1024, 16, 128, 64, 16),
    (512, 2, 100, 16, 32),
]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_heads, head_size, num_blocks, block_size = shape
    g = torch.Generator(device=device).manual_seed(0)

    key = torch.randn(
        (num_tokens, num_heads, head_size),
        generator=g,
        device=device,
        dtype=cur_dtype,
    )
    value = torch.randn(
        (num_tokens, num_heads, head_size),
        generator=g,
        device=device,
        dtype=cur_dtype,
    )
    key_cache = torch.zeros(
        (num_blocks, block_size, num_heads, head_size),
        device=device,
        dtype=cur_dtype,
    )
    value_cache = torch.zeros(
        (num_blocks, block_size, num_heads, head_size),
        device=device,
        dtype=cur_dtype,
    )
    slots = torch.arange(num_tokens, device=device, dtype=torch.int32) % (
        num_blocks * block_size
    )
    yield key, value, key_cache, value_cache, slots


def _case_fn(shape, cur_dtype):
    num_tokens, num_heads, head_size, num_blocks, block_size = shape
    yield BenchmarkCasePlan(
        shape={
            "key": (num_tokens, num_heads, head_size),
            "key_cache": (num_blocks, block_size, num_heads, head_size),
        },
        params={
            "configuration": shape,
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "head_size": head_size,
            "num_blocks": num_blocks,
            "block_size": block_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.reshape_and_cache_flash
def test_perf_reshape_and_cache_flash():
    bench = OpBenchmark(
        op_name="reshape_and_cache_flash",
        torch_op=get_reference("reshape_and_cache_flash"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads, head_size, blocks, block_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
