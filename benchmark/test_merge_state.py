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

"""Benchmark for attention/merge_state."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/merge_state.
SHAPES = [(n, 32, 128) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(7, 4, 64), (83, 16, 128), (3, 32, 512)]


def _input_fn(shape, cur_dtype, device):
    n_tokens, num_heads, head_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    prefix_output = torch.randn(
        n_tokens,
        num_heads,
        head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    suffix_output = torch.randn(
        n_tokens,
        num_heads,
        head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    # lse stays fp32; scaled up to exercise the max-subtraction path.
    prefix_lse = (
        torch.randn(n_tokens, num_heads, generator=g, device=device) * 3
    )
    suffix_lse = (
        torch.randn(n_tokens, num_heads, generator=g, device=device) * 3
    )
    yield prefix_output, prefix_lse, suffix_output, suffix_lse


@pytest.mark.merge_state
def test_perf_merge_state():
    bench = OpBenchmark(
        op_name="merge_state",
        torch_op=get_reference("merge_state"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n_tokens, num_heads, head_size",
    )
    bench.set_gems(flaggems_sglang.merge_state)
    bench.run()
