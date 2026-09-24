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

"""Benchmark for norm/fused_eh_norm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (2048, 4096, 7168)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (2048, 4096, 7168)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden = shape
    eps = 1e-6
    g = torch.Generator(device=device).manual_seed(0)
    inputs_embeds = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    previous_hidden = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    enorm_weight = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    hnorm_weight = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, {
        "eps": eps
    }


@pytest.mark.fused_eh_norm
def test_perf_fused_eh_norm():
    bench = OpBenchmark(
        op_name="fused_eh_norm",
        torch_op=get_reference("fused_eh_norm"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems(flaggems_sglang.fused_eh_norm)
    bench.run()
