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

"""Benchmark for activation_norm/gelu_and_mul."""

import pytest

import flaggems_sglang
from baseline import gelu_and_mul as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing


def _build_inputs(plan, cur_dtype, device):
    bs, d = plan.builder_args[0]
    args, kwargs = workload.build_inputs({"bs": bs, "d": d}, cur_dtype, device)
    return (*args, kwargs) if kwargs else args


def _case_fn(shape, cur_dtype):
    bs, d = shape
    yield BenchmarkCasePlan(
        shape={"hidden_states": (bs, 2 * d)},
        params={"configuration": shape, "seed": 0, "bs": bs, "d": d},
        builder_args=(shape,),
    )


@pytest.mark.gelu_and_mul
def test_perf_gelu_and_mul():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("gelu_and_mul")
        if gems_op is None:
            pytest.skip("activation_norm/gelu_and_mul not implemented yet")
    bench = OpBenchmark(
        op_name="gelu_and_mul",
        torch_op=get_reference("gelu_and_mul"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, d",
    )
    bench.set_gems(gems_op)
    bench.run()
