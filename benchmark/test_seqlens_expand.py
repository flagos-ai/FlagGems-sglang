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

"""Benchmark for attention/seqlens_expand."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

SHAPES = [(n,) for n in (1, 8, 64, 512)]
MORE_SHAPES = [(n,) for n in (4096,)]


def _input_fn(shape, cur_dtype, device):
    (n,) = shape
    max_q = 16
    g = torch.Generator(device=device).manual_seed(0)
    cu_seqlens_q = torch.zeros(n + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.randint(
        1, max_q + 1, (n,), dtype=torch.int32, device=device, generator=g
    ).cumsum(0)
    yield (cu_seqlens_q,)


def _case_fn(shape, cur_dtype):
    (n,) = shape
    yield BenchmarkCasePlan(
        shape={"cu_seqlens_q": (n + 1,)},
        params={"configuration": shape, "seed": 0, "n": n},
        builder_args=(shape,),
    )


@pytest.mark.seqlens_expand
def test_perf_seqlens_expand():
    bench = OpBenchmark(
        op_name="seqlens_expand",
        torch_op=get_reference("seqlens_expand"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
