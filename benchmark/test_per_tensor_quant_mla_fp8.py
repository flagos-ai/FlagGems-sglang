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

"""Benchmark for per_tensor_quant_mla_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# (num_head, num_seq, head_size), in the absorbed MLA layout.
SHAPES = [
    # DeepSeek's 576-wide latent over a realistic batch.
    (128, 64, 576),
    # Single-token decode: the grid is (num_seq, num_head) = (1, 128).
    (128, 1, 576),
    (64, 32, 512),
]

MORE_SHAPES = [
    (32, 128, 128),
    (16, 256, 64),
    (8, 512, 128),
    (1, 1, 512),
]


def _input_fn(shape, cur_dtype, device):
    num_head, num_seq, head_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (num_head, num_seq, head_size), generator=g, device=device
    ).to(cur_dtype)
    yield (x, 1e-12)


def _case_fn(shape, cur_dtype):
    num_head, num_seq, head_size = shape
    yield BenchmarkCasePlan(
        shape={"x": (num_head, num_seq, head_size)},
        params={
            "configuration": shape,
            "num_head": num_head,
            "num_seq": num_seq,
            "head_size": head_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.per_tensor_quant_mla_fp8
def test_perf_per_tensor_quant_mla_fp8():
    bench = OpBenchmark(
        op_name="per_tensor_quant_mla_fp8",
        torch_op=get_reference("per_tensor_quant_mla_fp8"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_head, num_seq, head_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
