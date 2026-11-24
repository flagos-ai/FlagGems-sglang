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

"""Benchmark for fla/fused_recurrent_kda."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

_H, _K, _V = 8, 128, 128
_T = 64
# (batch, t) — 8 heads, K = V = 128, 64 recurrent steps.
SHAPES = [(b, _T) for b in (1, 4, 16, 64)]
MORE_SHAPES = [(2, 64), (8, 32), (32, 128)]


def _input_fn(shape, cur_dtype, device):
    b, t = shape
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*dims):
        return torch.randn(
            *dims, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    beta = torch.rand(
        b, t, _H, dtype=torch.float32, device=device, generator=g
    )
    # Log-space decay, <= 0.
    gate = -0.05 * torch.rand(
        b, t, _H, _K, dtype=torch.float32, device=device, generator=g
    )
    yield dict(
        q=_t(b, t, _H, _K).contiguous(),
        k=_t(b, t, _H, _K).contiguous(),
        v=_t(b, t, _H, _V).contiguous(),
        g=gate.contiguous(),
        beta=beta.to(cur_dtype).contiguous(),
        scale=_K**-0.5,
        initial_state=torch.zeros(
            b, _H, _V, _K, dtype=torch.float32, device=device
        ),
        use_qk_l2norm_in_kernel=False,
    )


@pytest.mark.fused_recurrent_kda
def test_perf_fused_recurrent_kda():
    gems_op = flaggems_sglang.get_op("fused_recurrent_kda")
    if gems_op is None:
        pytest.skip("fla/fused_recurrent_kda not implemented yet")
    bench = OpBenchmark(
        op_name="fused_recurrent_kda",
        torch_op=get_reference("fused_recurrent_kda"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, t",
    )
    bench.set_gems(gems_op)
    bench.run()
