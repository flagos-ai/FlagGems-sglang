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

"""Benchmark for attention/concat_mla_k."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(t, 16, 192, 512) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 16, 192, 512) for t in (4096,)]


def _input_fn(shape, cur_dtype, device):
    total_tokens, num_heads, head_dim, kv_lora_rank = shape
    g = torch.Generator(device=device).manual_seed(0)
    k_nope = torch.randn(
        total_tokens,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    k_pe = torch.randn(
        total_tokens,
        kv_lora_rank,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    yield k_nope, k_pe


@pytest.mark.concat_mla_k
def test_perf_concat_mla_k():
    bench = OpBenchmark(
        op_name="concat_mla_k",
        torch_op=get_reference("concat_mla_k"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="total_tokens, num_heads, head_dim, kv_lora_rank",
    )
    bench.set_gems(flaggems_sglang.concat_mla_k)
    bench.run()
