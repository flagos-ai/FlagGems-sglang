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

"""Benchmark for speculative/fused_norm_rope_stacked."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/speculative/fused_norm_rope_stacked.
SHAPES = [(t, 8, 8, 128, 128) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 2, 2, 64, 64), (37, 3, 4, 128, 128), (129, 2, 2, 128, 64)]

_MAX_POS = 4096
_EPS = 1e-6


def _input_fn(shape, cur_dtype, device):
    t, n_layers, num_kv_heads, head_dim, rotary_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    kv_size = num_kv_heads * head_dim
    kv = torch.randn(
        t,
        n_layers,
        2 * kv_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    k_norm_weight = torch.randn(
        n_layers, head_dim, generator=g, device=device, dtype=torch.float32
    )
    eps = torch.full((n_layers,), _EPS, device=device, dtype=torch.float32)
    cos_sin_cache = torch.randn(
        _MAX_POS,
        rotary_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    positions = torch.randint(
        0, _MAX_POS, (t,), generator=g, device=device, dtype=torch.int64
    )
    yield (
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )


@pytest.mark.fused_norm_rope_stacked
def test_perf_fused_norm_rope_stacked():
    bench = OpBenchmark(
        op_name="fused_norm_rope_stacked",
        torch_op=get_reference("fused_norm_rope_stacked"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="t, n_layers, num_kv_heads, head_dim, rotary_dim",
    )
    bench.set_gems(flaggems_sglang.fused_norm_rope_stacked)
    bench.run()
