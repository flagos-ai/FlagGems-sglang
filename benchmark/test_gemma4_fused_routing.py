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

"""Benchmark for moe/gemma4_fused_routing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_TOPK = 8
# (tokens, experts) — topk fixed at 8; E <= 1024.
SHAPES = [(t, e) for t in (1, 8, 64, 512, 4096) for e in (64, 128, 256)]
MORE_SHAPES = [(7, 8), (1024, 128), (64, 1024)]


def _input_fn(shape, cur_dtype, device):
    t, e = shape
    g = create_generator(device, seed=0)
    gating = torch.randn(
        t, e, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    scale = (
        0.5 + torch.rand(e, dtype=torch.float32, device=device, generator=g)
    ).to(cur_dtype)
    yield dict(gating_output=gating, per_expert_scale=scale, topk=_TOPK)


@pytest.mark.gemma4_fused_routing
def test_perf_gemma4_fused_routing():
    gems_op = flaggems_sglang.get_op("gemma4_fused_routing")
    if gems_op is None:
        pytest.skip("moe/gemma4_fused_routing not implemented yet")
    bench = OpBenchmark(
        op_name="gemma4_fused_routing",
        torch_op=get_reference("gemma4_fused_routing"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="tokens, experts",
    )
    bench.set_gems(gems_op)
    bench.run()
