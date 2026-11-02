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

"""Benchmark for rope/qk_gemma_rmsnorm_rope."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 32, 8, 128, 128) for m in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 32, 8, 128, 128)]

_EPS = 1e-6
_MAX_SEQ = 4096


def _input_fn(shape, cur_dtype, device):
    m, nq, nkv, head_dim, rotary_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    angle = (
        torch.rand(_MAX_SEQ, rotary_dim // 2, device=device, generator=g)
        * 6.28
    )
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(cur_dtype)
    positions = torch.randint(
        0, _MAX_SEQ, (m,), device=device, generator=g, dtype=torch.int32
    )
    # is_neox_style is a bool, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    yield (
        _t(m, nq * head_dim),
        _t(m, nkv * head_dim),
        (0.5 * _t(head_dim).float()).to(cur_dtype),
        (0.5 * _t(head_dim).float()).to(cur_dtype),
        positions,
        cos_sin_cache,
        _EPS,
        head_dim,
        rotary_dim,
        {"is_neox_style": True},
    )


@pytest.mark.qk_gemma_rmsnorm_rope
def test_perf_qk_gemma_rmsnorm_rope():
    bench = OpBenchmark(
        op_name="qk_gemma_rmsnorm_rope",
        torch_op=get_reference("qk_gemma_rmsnorm_rope"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim",
    )
    bench.set_gems(flaggems_sglang.qk_gemma_rmsnorm_rope)
    bench.run()
