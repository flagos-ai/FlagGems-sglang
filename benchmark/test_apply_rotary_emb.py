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

"""Benchmark for rope/apply_rotary_emb."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_HEADS = 16
# (tokens, rope_dim) — 3-D input, forward rotation, no positions table.
SHAPES = [(t, 64) for t in (1, 8, 64, 512, 4096, 16384)]
MORE_SHAPES = [(7, 64), (333, 64), (1024, 128)]


def _input_fn(shape, cur_dtype, device):
    tokens, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        tokens,
        _HEADS,
        rope_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    angle = (
        torch.rand(tokens, rope_dim // 2, device=device, generator=g)
        * 6.283185307
    )
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    yield dict(
        x=x.contiguous(), freqs_cis=freqs_cis, positions=None, inverse=False
    )


@pytest.mark.apply_rotary_emb
def test_perf_apply_rotary_emb():
    gems_op = flaggems_sglang.get_op("apply_rotary_emb")
    if gems_op is None:
        pytest.skip("rope/apply_rotary_emb not implemented yet")
    bench = OpBenchmark(
        op_name="apply_rotary_emb",
        torch_op=get_reference("apply_rotary_emb"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, rope_dim",
    )
    bench.set_gems(gems_op)
    bench.run()
