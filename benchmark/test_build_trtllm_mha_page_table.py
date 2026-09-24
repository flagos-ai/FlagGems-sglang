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

"""Benchmark for attention/build_trtllm_mha_page_table."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(bs, 64, 16384) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 64, 16384) for bs in (128,)]


def _input_fn(shape, cur_dtype, device):
    bs, block_size, max_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    num_blocks_per_seq = (max_len + block_size - 1) // block_size
    seq_lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    block_tables = torch.randint(
        0,
        10000,
        (bs, num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield seq_lens, block_tables, {"block_size": block_size}


@pytest.mark.build_trtllm_mha_page_table
def test_perf_build_trtllm_mha_page_table():
    bench = OpBenchmark(
        op_name="build_trtllm_mha_page_table",
        torch_op=get_reference("build_trtllm_mha_page_table"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, block_size, max_len",
    )
    bench.set_gems(flaggems_sglang.build_trtllm_mha_page_table)
    bench.run()
