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

"""Benchmark for attention/mla_kv_pack_quantize_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (tokens, heads, qk_nope, qk_rope, v_head). The first two are the DeepSeek-V3
# geometry at a decode-sized and a prefill-sized token count; the third is a
# narrow-heads case where the per-head grid is still competitive.
SHAPES = [
    (1024, 64, 128, 64, 128),
    (4096, 32, 128, 64, 128),
    (64, 32, 128, 64, 128),
]

# Every launch regime in _pick_launch, so the tuned crossovers get timed too.
MORE_SHAPES = [
    (1, 32, 128, 64, 128),
    (17, 16, 128, 64, 128),
    (1600, 8, 128, 64, 128),
]


def _input_fn(shape, cur_dtype, device):
    tokens, heads, qk_nope, qk_rope, v_head = shape
    g = torch.Generator(device=device).manual_seed(0)

    k_nope = torch.randn(
        tokens, heads, qk_nope, generator=g, device=device
    ).to(cur_dtype)
    k_pe = torch.randn(tokens, 1, qk_rope, generator=g, device=device).to(
        cur_dtype
    )
    v = torch.randn(tokens, heads, v_head, generator=g, device=device).to(
        cur_dtype
    )

    yield k_nope, k_pe, v, 0.7, 1.3


def _case_fn(shape, cur_dtype):
    tokens, heads, qk_nope, qk_rope, v_head = shape
    yield BenchmarkCasePlan(
        shape={
            "k_nope": (tokens, heads, qk_nope),
            "k_pe": (tokens, 1, qk_rope),
            "v": (tokens, heads, v_head),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "tokens": tokens,
            "heads": heads,
            "qk_nope": qk_nope,
            "qk_rope": qk_rope,
            "v_head": v_head,
        },
        builder_args=(shape,),
    )


@pytest.mark.mla_kv_pack_quantize_fp8
def test_perf_mla_kv_pack_quantize_fp8():
    bench = OpBenchmark(
        op_name="mla_kv_pack_quantize_fp8",
        torch_op=get_reference("mla_kv_pack_quantize_fp8"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16, torch.float16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads, qk_nope, qk_rope, v_head",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
