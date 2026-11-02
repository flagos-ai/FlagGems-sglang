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

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

SHAPES = [(bs, 64, 16384) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 64, 16384) for bs in (128,)]


def _input_fn(shape, cur_dtype, device):
    bs, page_size, max_len = shape
    max_batch, max_context = 256, 131072
    g = torch.Generator(device=device).manual_seed(0)
    req_to_token = torch.randint(
        0,
        1 << 22,
        (max_batch, max_context),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    req_pool_indices = (
        torch.randperm(
            max_batch, generator=torch.Generator(device="cpu").manual_seed(0)
        )[:bs]
        .to(torch.int32)
        .to(device)
    )
    cache_seqlens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    width = (max_len + page_size - 1) // page_size
    page_table = torch.zeros(bs, width, dtype=torch.int32, device=device)
    yield req_to_token, req_pool_indices, cache_seqlens, page_table, page_size


def _case_fn(shape, cur_dtype):
    (bs, block_size, max_len) = shape
    yield BenchmarkCasePlan(
        shape={
            "seq_lens": (bs,),
            "block_tables": (bs, (max_len + block_size - 1) // block_size),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "bs": bs,
            "block_size": block_size,
            "max_len": max_len,
        },
        builder_args=(shape,),
    )


@pytest.mark.build_trtllm_mha_page_table
def test_perf_build_trtllm_mha_page_table():
    bench = OpBenchmark(
        op_name="build_trtllm_mha_page_table",
        torch_op=get_reference("build_trtllm_mha_page_table"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, page_size, max_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
