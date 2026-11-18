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

"""Benchmark for attention/mla_kv_pack_quantize_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_QK_NOPE, _QK_ROPE, _V_HEAD = 512, 64, 512
# (s, heads) — the baseline picks its kernel off this pair, and these shapes
# cross that boundary.
SHAPES = [(s, h) for s in (1, 8, 64, 512, 4096) for h in (1, 16)]
MORE_SHAPES = [(7, 8), (333, 1), (256, 8)]


def _input_fn(shape, cur_dtype, device):
    s, heads = shape
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*dims):
        return torch.randn(
            *dims, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    yield dict(
        k_nope=_t(s, heads, _QK_NOPE).contiguous(),
        k_pe=_t(s, _QK_ROPE).contiguous(),
        v=_t(s, heads, _V_HEAD).contiguous(),
        k_scale_inv=1.0,
        v_scale_inv=1.0,
        fp8_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.mla_kv_pack_quantize_fp8
def test_perf_mla_kv_pack_quantize_fp8():
    gems_op = flaggems_sglang.get_op("mla_kv_pack_quantize_fp8")
    if gems_op is None:
        pytest.skip("attention/mla_kv_pack_quantize_fp8 not implemented yet")
    bench = OpBenchmark(
        op_name="mla_kv_pack_quantize_fp8",
        torch_op=get_reference("mla_kv_pack_quantize_fp8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="s, heads",
    )
    bench.set_gems(gems_op)
    bench.run()
