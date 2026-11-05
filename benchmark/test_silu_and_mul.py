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

"""Benchmark for activation_norm/silu_and_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/silu_and_mul.
SHAPES = [(bs, 4096) for bs in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 8192)]


def _case_fn(shape, cur_dtype):
    del cur_dtype
    bs, d = shape
    yield BenchmarkCasePlan(
        shape={"hidden_states": (bs, 2 * d)},
        params={"bs": bs, "d": d, "seed": 0},
        builder_args=(bs, d),
    )


def _build_inputs(plan, cur_dtype, device):
    bs, d = plan.builder_args
    # The op reads x1/x3 from a single [bs, 2 * d] tensor.
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        bs, 2 * d, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    return (hidden_states,)


@pytest.mark.silu_and_mul
def test_perf_silu_and_mul():
    bench = OpBenchmark(
        op_name="silu_and_mul",
        torch_op=get_reference("silu_and_mul"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
