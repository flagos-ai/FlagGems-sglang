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

"""Benchmark for fla/fused_recurrent_gdn."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

OUTPUT_FINAL_STATE = True

# Shapes match kernel-comp-baseline/problems/fla/fused_recurrent_gdn.
SHAPES = [(8, 128, 8, 8, 64, 64), (32, 32, 8, 8, 64, 64)]
MORE_SHAPES = [
    (1, 4, 2, 2, 16, 16),
    (2, 8, 2, 4, 32, 32),
    (3, 6, 4, 4, 64, 32),
]


def _input_fn(shape, cur_dtype, device):
    batch, t, h, hv, k_dim, v_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        batch, t, h, k_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    k = torch.randn(
        batch, t, h, k_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    v = torch.randn(
        batch, t, hv, v_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # The gate stays negative so the recurrence decays.
    gate = (
        -torch.rand(
            batch, t, hv, generator=g, device=device, dtype=torch.float32
        )
        * 0.1
    )
    beta = torch.sigmoid(
        torch.randn(
            batch, t, hv, generator=g, device=device, dtype=torch.float32
        )
    )
    # No initial state; ``use_qk_l2norm_in_kernel`` keeps its default.
    yield q, k, v, gate, beta, k_dim**-0.5, None, OUTPUT_FINAL_STATE


@pytest.mark.fused_recurrent_gdn
def test_perf_fused_recurrent_gdn():
    # This op has a reference but no Triton implementation yet.
    gems_op = flaggems_sglang.get_op("fused_recurrent_gdn")
    if gems_op is None:
        pytest.skip("fla/fused_recurrent_gdn not implemented yet")
    bench = OpBenchmark(
        op_name="fused_recurrent_gdn",
        torch_op=get_reference("fused_recurrent_gdn"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, t, h, hv, k_dim, v_dim",
    )
    bench.set_gems(gems_op)
    bench.run()
