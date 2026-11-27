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

"""Benchmark for kvcache/absorbed_bmm_concat_cast_q_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_K, _N, _ROPE = 128, 512, 64
# (tokens, heads) — k=128, n=512, rope=64.
SHAPES = [(t, 16) for t in (16, 128, 1024, 4096)]
MORE_SHAPES = [(16, 8), (512, 16), (64, 8)]


def _input_fn(shape, cur_dtype, device):
    tokens, heads = shape
    g = create_generator(device, seed=0)

    def _t(*dims):
        return (
            0.25
            * torch.randn(
                *dims, dtype=torch.float32, device=device, generator=g
            )
        ).to(cur_dtype)

    q_fp8_pad = torch.full(
        (tokens, heads, _N + _ROPE),
        3.0,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    yield dict(
        q_fp8_pad=q_fp8_pad,
        q_nope=_t(tokens, heads, _K).contiguous(),
        w_kc=_t(heads, _K, _N).contiguous(),
        q_rope=_t(tokens, heads, _ROPE).contiguous(),
        num_heads=heads,
    )


@pytest.mark.absorbed_bmm_concat_cast_q_fp8
def test_perf_absorbed_bmm_concat_cast_q_fp8():
    gems_op = flaggems_sglang.get_op("absorbed_bmm_concat_cast_q_fp8")
    if gems_op is None:
        pytest.skip(
            "kvcache/absorbed_bmm_concat_cast_q_fp8 not implemented yet"
        )
    bench = OpBenchmark(
        op_name="absorbed_bmm_concat_cast_q_fp8",
        torch_op=get_reference("absorbed_bmm_concat_cast_q_fp8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, heads",
    )
    bench.set_gems(gems_op)
    bench.run()
