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

"""Benchmark for diffusion/indexed_scale_shift."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/diffusion/indexed_scale_shift.
SHAPES = [(rows, 3072, 8) for rows in (1, 64, 1024, 8192)]
MORE_SHAPES = [(rows, 3072, 8) for rows in (32768,)]


def _input_fn(shape, cur_dtype, device):
    rows, hidden, variants = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(rows, hidden, device=device, generator=g).to(cur_dtype)
    shift = torch.randn(variants, hidden, device=device, generator=g).to(
        cur_dtype
    )
    scale = torch.randn(variants, hidden, device=device, generator=g).to(
        cur_dtype
    )
    indices = torch.randint(
        0, variants, (rows,), dtype=torch.int32, device=device, generator=g
    )
    yield x, shift, scale, indices


def _case_fn(shape, cur_dtype):
    (rows, hidden, variants) = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (rows, hidden),
            "shift": (variants, hidden),
            "scale": (variants, hidden),
            "indices": (rows,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "rows": rows,
            "hidden": hidden,
            "variants": variants,
        },
        builder_args=(shape,),
    )


@pytest.mark.indexed_scale_shift
def test_perf_indexed_scale_shift():
    bench = OpBenchmark(
        op_name="indexed_scale_shift",
        torch_op=get_reference("indexed_scale_shift"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="rows, hidden, variants",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
