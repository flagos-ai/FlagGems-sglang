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

"""Benchmark for activation_norm/gemma_qkv_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 32, 8, 128) for m in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 32, 8, 128)]

_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    m, nq, nkv, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(cols):
        return torch.randn(
            m, cols, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    # The packed-view and KV-shared (k=v=None) paths are covered by the test.
    yield (
        _t(nq * head_dim),
        _t(nkv * head_dim),
        _t(nkv * head_dim),
        _t(head_dim).reshape(-1)[:head_dim],
        _t(head_dim).reshape(-1)[:head_dim],
        nq,
        nkv,
        head_dim,
        _EPS,
    )


@pytest.mark.gemma_qkv_rmsnorm
def test_perf_gemma_qkv_rmsnorm():
    bench = OpBenchmark(
        op_name="gemma_qkv_rmsnorm",
        torch_op=get_reference("gemma_qkv_rmsnorm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_q_heads, num_kv_heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.gemma_qkv_rmsnorm)
    bench.run()
