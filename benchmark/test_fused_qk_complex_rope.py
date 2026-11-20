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

"""Benchmark for rope/fused_qk_complex_rope."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_HEADS = 16
# (tokens, head_dim) — q and k rotate together in one launch.
SHAPES = [(t, 128) for t in (1, 64, 1024, 4096, 16384)]
MORE_SHAPES = [(7, 64), (577, 72), (256, 64)]


def _input_fn(shape, cur_dtype, device):
    tokens, head_dim = shape
    g = torch.Generator(device=device).manual_seed(0)

    def _t():
        return torch.randn(
            tokens,
            _HEADS,
            head_dim,
            dtype=torch.float32,
            device=device,
            generator=g,
        ).to(cur_dtype)

    angle = (
        torch.rand(tokens, head_dim // 2, device=device, generator=g)
        * 6.283185307
    )
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    yield dict(q=_t(), k=_t(), freqs_cis=freqs_cis)


@pytest.mark.fused_qk_complex_rope
def test_perf_fused_qk_complex_rope():
    gems_op = flaggems_sglang.get_op("fused_qk_complex_rope")
    if gems_op is None:
        pytest.skip("rope/fused_qk_complex_rope not implemented yet")
    bench = OpBenchmark(
        op_name="fused_qk_complex_rope",
        torch_op=get_reference("fused_qk_complex_rope"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, head_dim",
    )
    bench.set_gems(gems_op)
    bench.run()
