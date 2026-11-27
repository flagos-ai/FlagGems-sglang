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

"""Benchmark for moe/silu_mul_dynamic_tensorwise_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

# (num_tokens, intermediate) — the live region equals the buffer here; the
# scale is computed inside the operator.
SHAPES = [(t, 2048) for t in (64, 512, 4096)]
MORE_SHAPES = [(128, 512), (1024, 1024), (256, 2048)]


def _input_fn(shape, cur_dtype, device):
    t, inter = shape
    g = create_generator(device, seed=0)
    inp = torch.randn(
        t, 2 * inter, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield dict(
        input=inp.contiguous(),
        output=torch.zeros(
            (t, inter), dtype=torch.float8_e4m3fn, device=device
        ),
        scale=torch.zeros(1, dtype=torch.float32, device=device),
        num_tokens_tensor=torch.tensor([t], dtype=torch.int32, device=device),
        expected_num_tokens=t,
        intermediate_size=inter,
    )


@pytest.mark.silu_mul_dynamic_tensorwise_quant
def test_perf_silu_mul_dynamic_tensorwise_quant():
    gems_op = flaggems_sglang.get_op("silu_mul_dynamic_tensorwise_quant")
    if gems_op is None:
        pytest.skip(
            "moe/silu_mul_dynamic_tensorwise_quant not implemented yet"
        )
    bench = OpBenchmark(
        op_name="silu_mul_dynamic_tensorwise_quant",
        torch_op=get_reference("silu_mul_dynamic_tensorwise_quant"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, intermediate",
    )
    bench.set_gems(gems_op)
    bench.run()
