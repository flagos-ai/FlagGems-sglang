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

"""Benchmark for moe/dsv3_fused_a_gemm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [(m, 7168, 2112) for m in (1, 2, 4, 8)]
MORE_SHAPES = [(m, 7168, 2112) for m in (16,)]


def _input_fn(shape, cur_dtype, device):
    m, hd_in, hd_out = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        m, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # Column-major B: build the row-major [hd_out, hd_in] weight and transpose.
    w = torch.randn(
        hd_out, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield a, w.t()


@pytest.mark.dsv3_fused_a_gemm
def test_perf_dsv3_fused_a_gemm():
    bench = OpBenchmark(
        op_name="dsv3_fused_a_gemm",
        torch_op=get_reference("dsv3_fused_a_gemm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, hd_in, hd_out",
    )
    bench.set_gems(flaggems_sglang.dsv3_fused_a_gemm)
    bench.run()
