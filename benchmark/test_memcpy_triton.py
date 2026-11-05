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

"""Benchmark for utility/memcpy_triton."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/utility/memcpy_triton.
SHAPES = [(rows, 4096) for rows in (8, 64, 512, 4096)]
MORE_SHAPES = [(rows, 4096) for rows in (16384,)]


def _input_fn(shape, cur_dtype, device):
    rows, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)
    src = torch.randn(rows, hidden, device=device, generator=g).to(cur_dtype)
    dst = torch.zeros(rows, hidden, dtype=cur_dtype, device=device)
    # offset and size live in device memory so the copy stays capturable.
    offset = torch.tensor([0], dtype=torch.int32, device=device)
    sz = torch.tensor([rows // 2], dtype=torch.int32, device=device)
    yield dst, src, 0, offset, sz, True


def _case_fn(shape, cur_dtype):
    (rows, hidden) = shape
    yield BenchmarkCasePlan(
        shape={
            "dst": (rows, hidden),
            "src": (rows, hidden),
            "offset": (1,),
            "sz": (1,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "rows": rows,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.memcpy_triton
def test_perf_memcpy_triton():
    bench = OpBenchmark(
        op_name="memcpy_triton",
        torch_op=get_reference("memcpy_triton"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
