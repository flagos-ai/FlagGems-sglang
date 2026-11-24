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

"""Benchmark for rope/fused_norm_rope_inplace."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_HEAD_DIM, _ROPE_DIM = 576, 64
_EPS = 1e-6
# (m,) — head_dim 576 with a 64-wide rope tail.
SHAPES = [(m,) for m in (1, 8, 64, 512, 4096, 16384)]
MORE_SHAPES = [(7,), (333,), (1024,)]


def _input_fn(shape, cur_dtype, device):
    (m,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    kv = torch.randn(
        m, _HEAD_DIM, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight = (
        0.5
        * torch.randn(
            _HEAD_DIM, dtype=torch.float32, device=device, generator=g
        )
    ).to(cur_dtype)
    angle = (
        torch.rand(m, _ROPE_DIM // 2, device=device, generator=g) * 6.283185307
    )
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    yield dict(
        kv=kv.contiguous(),
        weight=weight,
        eps=_EPS,
        freqs_cis=freqs_cis,
        positions=None,
    )


@pytest.mark.fused_norm_rope_inplace
def test_perf_fused_norm_rope_inplace():
    gems_op = flaggems_sglang.get_op("fused_norm_rope_inplace")
    if gems_op is None:
        pytest.skip("rope/fused_norm_rope_inplace not implemented yet")
    bench = OpBenchmark(
        op_name="fused_norm_rope_inplace",
        torch_op=get_reference("fused_norm_rope_inplace"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m",
    )
    bench.set_gems(gems_op)
    bench.run()
