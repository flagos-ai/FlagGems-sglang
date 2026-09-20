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

"""Benchmark for activation_norm/softcap_inplace_logits."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

SOFTCAP_CONST = 30.0

# Shapes match kernel-comp-baseline/problems/activation_norm/
# softcap_inplace_logits.
SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (4096, 32000, 128256)]
MORE_SHAPES = [(1, 17), (37, 1024), (4, 32000)]


def _input_fn(shape, cur_dtype, device):
    m, n = shape
    g = torch.Generator(device=device).manual_seed(0)
    # Scaled up so the inputs reach the tanh saturation region.
    x = (
        torch.randn(m, n, generator=g, device=device, dtype=torch.float32) * 20
    ).to(cur_dtype)
    yield x, SOFTCAP_CONST


@pytest.mark.softcap_inplace_logits
def test_perf_softcap_inplace_logits():
    # The op caps in place, so each repeated timing call re-caps the buffer
    # left by the previous one. Values shrink toward zero across repetitions
    # but stay finite and in-range, and the work done (one read plus one
    # write per element, which is what this bandwidth-bound op costs) is
    # identical every call, so the measured latency is unaffected.
    bench = OpBenchmark(
        op_name="softcap_inplace_logits",
        torch_op=get_reference("softcap_inplace_logits"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, n",
    )
    bench.set_gems(flaggems_sglang.softcap_inplace_logits)
    bench.run()
