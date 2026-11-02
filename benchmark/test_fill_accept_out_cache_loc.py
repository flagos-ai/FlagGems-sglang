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

"""Benchmark for speculative/fill_accept_out_cache_loc."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/fill_accept_out_cache_loc.
SHAPES = [(s,) for s in (8, 64, 512)]
MORE_SHAPES = [(4096,)]


def _input_fn(shape, cur_dtype, device):
    (size,) = shape
    g = torch.Generator(device=device).manual_seed(9)
    accept_index = torch.arange(size, dtype=torch.int64, device=device)
    accept_index[torch.rand(size, device=device, generator=g) > 0.5] = -1
    out_cache_loc = torch.randint(
        1,
        1 << 20,
        (size,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    yield accept_index, out_cache_loc, accept_out_cache_loc, size


def _case_fn(shape, cur_dtype):
    (size,) = shape
    yield BenchmarkCasePlan(
        shape={
            "accept_index": (size,),
            "out_cache_loc": (size,),
            "accept_out_cache_loc": (size,),
        },
        params={"configuration": shape, "seed": 9, "size": size},
        builder_args=(shape,),
    )


@pytest.mark.fill_accept_out_cache_loc
def test_perf_fill_accept_out_cache_loc():
    bench = OpBenchmark(
        op_name="fill_accept_out_cache_loc",
        torch_op=get_reference("fill_accept_out_cache_loc"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
