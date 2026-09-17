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

"""Benchmark for rope/mrope_fused."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

MROPE_SECTION = [16, 24, 24]
MAX_POSITION = 4096

# Shapes match kernel-comp-baseline/problems/rope/mrope_fused.
SHAPES = [(n, 8, 2, 128, 128) for n in (1, 128, 2048, 8192)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, n_qh, n_kh, head_size, rotary_dim = shape
    q = torch.randn(
        num_tokens, n_qh * head_size, device=device, dtype=cur_dtype
    )
    k = torch.randn(
        num_tokens, n_kh * head_size, device=device, dtype=cur_dtype
    )
    cos_sin_cache = torch.randn(
        MAX_POSITION, rotary_dim, device=device, dtype=cur_dtype
    )
    positions = torch.randint(
        0, MAX_POSITION, (3, num_tokens), device=device, dtype=torch.int64
    )
    yield q, k, cos_sin_cache, positions, MROPE_SECTION, head_size, rotary_dim


@pytest.mark.mrope_fused
def test_perf_mrope_fused():
    bench = OpBenchmark(
        op_name="mrope_fused",
        torch_op=get_reference("mrope_fused"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        shape_desc="num_tokens, n_qh, n_kh, head_size, rotary_dim",
    )
    bench.set_gems(flaggems_sglang.mrope_fused)
    bench.run()
