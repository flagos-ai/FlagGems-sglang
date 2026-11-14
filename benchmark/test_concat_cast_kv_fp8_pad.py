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

"""Benchmark for kv_cache/concat_cast_kv_fp8_pad."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

NOPE = 512

# Entries are (tokens, rope, pad_rows). The first two mirror the upstream
# registered test's GLM-5.3-Flash geometry: a NoPE model with a 2176-row
# sentinel band, and a rope-64 model with a 2048-row band.
SHAPES = [
    (437, 0, 2176),
    (437, 64, 2048),
]
MORE_SHAPES = [
    (1, 0, 2176),
    (64, 0, 2176),
    (2048, 0, 2176),
    (4097, 0, 2176),
    (2048, 64, 2048),
    (4097, 64, 2048),
]


def _input_fn(shape, cur_dtype, device):
    tokens, rope, pad_rows = shape
    g = torch.Generator(device=device).manual_seed(0)
    head = NOPE + rope
    # k / k_rope arrive as [tokens, 1, width] views upstream (one KV head);
    # a [tokens, 1, head] row is sliced to its rope tail.
    k = (
        (torch.randn(tokens, NOPE, generator=g, device=device) * 0.5)
        .to(cur_dtype)
        .unsqueeze(1)
    )
    if rope > 0:
        k_rope = (
            (torch.randn(tokens, head, generator=g, device=device) * 0.5)
            .to(cur_dtype)[..., NOPE:]
            .unsqueeze(1)
        )
    else:
        k_rope = k.new_empty((tokens, 1, 0))
    # destination slice of a larger buffer, as the backend hands it over
    big = torch.empty(
        tokens + pad_rows + 50, head, dtype=torch.float8_e4m3fn, device=device
    )
    out = big[: tokens + pad_rows]
    yield out, k, k_rope, {"num_tokens": tokens}


def _case_fn(shape, cur_dtype):
    tokens, rope, pad_rows = shape
    head = NOPE + rope
    yield BenchmarkCasePlan(
        shape={
            "out": (tokens + pad_rows, head),
            "k": (tokens, 1, NOPE),
            "k_rope": (tokens, 1, rope),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "tokens": tokens,
            "rope": rope,
            "pad_rows": pad_rows,
        },
        builder_args=(shape,),
    )


@pytest.mark.concat_cast_kv_fp8_pad
def test_perf_concat_cast_kv_fp8_pad():
    bench = OpBenchmark(
        op_name="concat_cast_kv_fp8_pad",
        torch_op=get_reference("concat_cast_kv_fp8_pad"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, rope, pad_rows",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
