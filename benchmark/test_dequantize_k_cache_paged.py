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

"""Benchmark for kv_cache/dequantize_k_cache_paged."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# DeepSeek V3.2 DSA uses dim_nope=512, dim_rope=64.
SHAPES = [(tokens, 512, 64) for tokens in (1, 64, 512, 2048)]
MORE_SHAPES = [
    (8, 256, 0),
    (8, 512, 0),
    (8, 1024, 64),
    (8, 1024, 128),
]


def _pack(num_rows, dim_nope, dim_rope, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    group_size = 128
    num_tiles = dim_nope // group_size
    dim_quant = dim_nope + num_tiles * 4 + dim_rope * 2
    buf = torch.empty((num_rows, dim_quant), dtype=torch.uint8, device=device)

    q = (
        torch.randn(
            (num_rows, dim_nope),
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 50.0
    ).to(torch.float8_e4m3fn)
    buf[:, :dim_nope] = q.view(torch.uint8)

    scale = (
        torch.rand(
            (num_rows, num_tiles),
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    buf[:, dim_nope : dim_nope + num_tiles * 4] = scale.contiguous().view(
        torch.uint8
    )

    rope = torch.randn(
        (num_rows, dim_rope),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    buf[:, dim_nope + num_tiles * 4 :] = rope.view(torch.uint8)

    return buf.view(torch.float8_e4m3fn)


def _input_fn(shape, cur_dtype, device):
    num_tokens, dim_nope, dim_rope = shape
    # One cache row per token: the page table is sequential here so that the
    # benchmark measures the gather path, not a duplicate-row broadcast.
    packed = _pack(num_tokens, dim_nope, dim_rope, device)
    packed = packed.view(num_tokens, 1, -1)
    page_table = torch.arange(num_tokens, device=device, dtype=torch.int32)
    yield packed, {"page_table_1_flattened": page_table}


def _case_fn(shape, cur_dtype):
    num_tokens, dim_nope, dim_rope = shape
    yield BenchmarkCasePlan(
        shape={"quant_k_cache": (num_tokens, 1, dim_nope)},
        params={
            "configuration": shape,
            "seed": 0,
            "num_tokens": num_tokens,
            "dim_nope": dim_nope,
            "dim_rope": dim_rope,
        },
        builder_args=(shape,),
    )


@pytest.mark.dequantize_k_cache_paged
def test_perf_dequantize_k_cache_paged():
    bench = OpBenchmark(
        op_name="dequantize_k_cache_paged",
        torch_op=get_reference("dequantize_k_cache_paged"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, dim_nope, dim_rope",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
