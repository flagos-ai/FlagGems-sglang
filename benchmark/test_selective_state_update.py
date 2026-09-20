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

"""Benchmark for mamba/selective_state_update."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/selective_state_update.
# The trailing entries are ngroups, then the has_z / has_d / has_dt_bias /
# dt_softplus switches.
SHAPES = [
    (64, 32, 128, 128, 8, True, True, True, True),
    (256, 64, 64, 128, 8, True, True, True, True),
]
MORE_SHAPES = [
    (1, 4, 16, 8, 1, True, True, True, True),
    (5, 8, 64, 16, 2, True, True, True, True),
    (3, 16, 128, 32, 4, False, True, True, False),
    # D / z / dt_bias all absent, exercising the kernel's None guards.
    (2, 8, 32, 16, 2, False, False, False, False),
]


def _input_fn(shape, cur_dtype, device):
    (
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        has_z,
        has_d,
        has_dt_bias,
        dt_softplus,
    ) = shape
    g = torch.Generator(device=device).manual_seed(0)
    state = torch.randn(
        batch,
        nheads,
        dim,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    x = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    dt = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(
            nheads,
            dim,
            dstate,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        - 0.1
    )
    b = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    c = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # D / z / dt_bias are all optional per the signature; the kernel guards
    # them with USE_D / USE_Z / USE_DT_BIAS.
    d = None
    if has_d:
        d = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )
    z = None
    if has_z:
        z = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    dt_bias = None
    if has_dt_bias:
        dt_bias = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )
    yield state, x, dt, a, b, c, dict(
        D=d, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
    )


@pytest.mark.selective_state_update
def test_perf_selective_state_update():
    bench = OpBenchmark(
        op_name="selective_state_update",
        torch_op=get_reference("selective_state_update"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc=(
            "batch, nheads, dim, dstate, ngroups, "
            "has_z, has_d, has_dt_bias, dt_softplus"
        ),
    )
    bench.set_gems(flaggems_sglang.selective_state_update)
    bench.run()
