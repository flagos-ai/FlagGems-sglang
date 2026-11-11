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

"""Benchmark for set_mla_kv_buffer_fp8_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

FP8_E4M3 = torch.float8_e4m3fn

# (num_tokens, nope_dim, rope_dim, rows)
SHAPES = [
    # Decode-sized batches, where the grid has the least work to do.
    (64, 512, 64, 256),
    (256, 512, 64, 1024),
    (1, 512, 64, 128),
]

MORE_SHAPES = [
    (1024, 512, 64, 2048),
    (32, 512, 64, 512),
    (512, 128, 64, 1024),
    # A NoPE width that is not a multiple of the block width.
    (128, 96, 32, 512),
]


def _input_fn(shape, cur_dtype, device):
    num_tokens, nope_dim, rope_dim, rows = shape
    g = torch.Generator(device=device).manual_seed(0)

    kv_buffer = torch.randint(
        0,
        256,
        (rows, nope_dim + rope_dim),
        generator=g,
        device=device,
        dtype=torch.uint8,
    )
    nope = torch.randn(
        (num_tokens, nope_dim),
        generator=g,
        device=device,
        dtype=cur_dtype,
    )
    rope = torch.randn(
        (num_tokens, rope_dim),
        generator=g,
        device=device,
        dtype=cur_dtype,
    )
    loc = torch.randint(
        0, rows, (num_tokens,), generator=g, device=device, dtype=torch.int32
    )
    yield kv_buffer, loc, nope, rope, FP8_E4M3


def _case_fn(shape, cur_dtype):
    num_tokens, nope_dim, rope_dim, rows = shape
    yield BenchmarkCasePlan(
        shape={
            "kv_buffer": (rows, nope_dim + rope_dim),
            "cache_k_nope": (num_tokens, nope_dim),
            "cache_k_rope": (num_tokens, rope_dim),
        },
        params={
            "configuration": shape,
            "num_tokens": num_tokens,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
            "rows": rows,
        },
        builder_args=(shape,),
    )


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_perf_set_mla_kv_buffer_fp8_quant():
    bench = OpBenchmark(
        op_name="set_mla_kv_buffer_fp8_quant",
        torch_op=get_reference("set_mla_kv_buffer_fp8_quant"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, nope_dim, rope_dim, rows",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
