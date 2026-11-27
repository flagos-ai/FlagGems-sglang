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

"""Benchmark for quantization/per_tensor_quant_mla_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

# (num_head, num_seq) — head_size fixed at 512.
SHAPES = [(nh, ns) for nh in (1, 16, 128) for ns in (1, 32, 512, 4096)]
MORE_SHAPES = [(1, 7), (8, 83), (2, 3)]


def _input_fn(shape, cur_dtype, device):
    num_head, num_seq = shape
    head_size = 512
    g = create_generator(device, seed=0)
    x = (
        3
        * torch.randn(
            num_head,
            num_seq,
            head_size,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
    ).to(cur_dtype)
    yield dict(x=x, eps=1e-12)


@pytest.mark.per_tensor_quant_mla_fp8
def test_perf_per_tensor_quant_mla_fp8():
    gems_op = flaggems_sglang.get_op("per_tensor_quant_mla_fp8")
    if gems_op is None:
        pytest.skip(
            "quantization/per_tensor_quant_mla_fp8 not implemented yet"
        )
    bench = OpBenchmark(
        op_name="per_tensor_quant_mla_fp8",
        torch_op=get_reference("per_tensor_quant_mla_fp8"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_head, num_seq",
    )
    bench.set_gems(gems_op)
    bench.run()
