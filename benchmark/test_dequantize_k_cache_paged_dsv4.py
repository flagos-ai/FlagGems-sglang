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

"""Benchmark for kv_cache/dequantize_k_cache_paged_dsv4."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

DIM_NOPE = 448
DIM_ROPE = 64
NUM_SCALE_TILES = 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1

PAGE_SIZE = 16
SHAPES = [(tokens, PAGE_SIZE) for tokens in (1, 64, 512, 2048)]
# The cache pool actually uses page_size 64 and 256 (upstream's registered test
# says "both cache page sizes the pool uses"), so cover those token counts too
# instead of only the small-page edge cases.
MORE_SHAPES = [(8, 1), (8, 4), (2048, 64), (333, 64), (4097, 256)]


def _pack(num_pages, page_size, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * (
        NOPE_ROPE_BYTES
    )
    buf = torch.zeros(
        (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )

    q = (
        torch.randn(
            (num_pages, page_size, DIM_NOPE),
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 5.0
    ).to(torch.float8_e4m3fn)
    r = torch.randn(
        (num_pages, page_size, DIM_ROPE),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    data = torch.cat(
        [
            q.view(torch.uint8).reshape(num_pages, page_size, DIM_NOPE),
            r.view(torch.uint8).reshape(num_pages, page_size, DIM_ROPE * 2),
        ],
        dim=2,
    ).reshape(num_pages, page_size * NOPE_ROPE_BYTES)
    buf[:, : page_size * NOPE_ROPE_BYTES] = data

    scales = torch.randint(
        100,
        161,
        (num_pages, page_size, NUM_SCALE_TILES),
        generator=g,
        device=device,
        dtype=torch.uint8,
    )
    padded = torch.zeros(
        (num_pages, page_size, PADDED_SCALE_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    padded[:, :, :NUM_SCALE_TILES] = scales
    buf[:, page_size * NOPE_ROPE_BYTES :][
        :, : page_size * PADDED_SCALE_PER_TOKEN
    ] = padded.reshape(num_pages, -1)

    return buf


def _input_fn(shape, cur_dtype, device):
    num_tokens, page_size = shape
    num_pages = (num_tokens + page_size - 1) // page_size
    packed = _pack(num_pages, page_size, device)
    page_table = torch.arange(num_tokens, device=device, dtype=torch.int32)
    yield packed, {
        "page_table_1_flattened": page_table,
        "page_size": page_size,
    }


def _case_fn(shape, cur_dtype):
    num_tokens, page_size = shape
    yield BenchmarkCasePlan(
        shape={"quant_k_cache": (num_tokens, page_size)},
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "page_size": page_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.dequantize_k_cache_paged_dsv4
def test_perf_dequantize_k_cache_paged_dsv4():
    bench = OpBenchmark(
        op_name="dequantize_k_cache_paged_dsv4",
        torch_op=get_reference("dequantize_k_cache_paged_dsv4"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.uint8],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, page_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
