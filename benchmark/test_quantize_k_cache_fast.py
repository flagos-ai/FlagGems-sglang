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

"""Benchmark for dsa/quantize_k_cache_fast."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

_DIM_NOPE, _DIM_ROPE = 512, 64
# (num_tokens,) — group_size fixed at 128.
SHAPES = [(t,) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(37,), (256,), (4096,)]


def _input_fn(shape, cur_dtype, device):
    (num_tokens,) = shape
    g = create_generator(device, seed=0)
    k_nope = (
        2 * torch.randn(num_tokens, _DIM_NOPE, generator=g, device=device)
    ).to(cur_dtype)
    k_rope = torch.randn(num_tokens, _DIM_ROPE, generator=g, device=device).to(
        cur_dtype
    )
    yield dict(k_nope=k_nope, k_rope=k_rope, group_size=128)


@pytest.mark.quantize_k_cache_fast
def test_perf_quantize_k_cache_fast():
    gems_op = flaggems_sglang.get_op("quantize_k_cache_fast")
    if gems_op is None:
        pytest.skip("dsa/quantize_k_cache_fast not implemented yet")
    bench = OpBenchmark(
        op_name="quantize_k_cache_fast",
        torch_op=get_reference("quantize_k_cache_fast"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens",
    )
    bench.set_gems(gems_op)
    bench.run()
