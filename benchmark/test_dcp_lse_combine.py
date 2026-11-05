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

"""Benchmark for attention/dcp_lse_combine."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/dcp_lse_combine.
SHAPES = [(n, b, 16, 128) for n in (2, 4, 8) for b in (1, 64)]
MORE_SHAPES = [(n, b, 16, 128) for n in (2, 4, 8) for b in (1024,)]


def _input_fn(shape, cur_dtype, device):
    n, b, h, d = shape
    g = torch.Generator(device=device).manual_seed(9)
    recv_output = torch.randn(n, b, h, d, device=device, generator=g).to(
        cur_dtype
    )
    # The LSE side stays fp32 regardless of the output dtype.
    recv_lse = (
        torch.randn(n, b, h, dtype=torch.float32, device=device, generator=g)
        * 4
    )
    yield recv_output.contiguous(), recv_lse.contiguous(), True, True


def _case_fn(shape, cur_dtype):
    (n, b, h, d) = shape
    yield BenchmarkCasePlan(
        shape={"recv_output": (n, b, h, d), "recv_lse": (n, b, h)},
        params={
            "configuration": shape,
            "seed": 9,
            "n": n,
            "b": b,
            "h": h,
            "d": d,
        },
        builder_args=(shape,),
    )


@pytest.mark.dcp_lse_combine
def test_perf_dcp_lse_combine():
    bench = OpBenchmark(
        op_name="dcp_lse_combine",
        torch_op=get_reference("dcp_lse_combine"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n, b, h, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
