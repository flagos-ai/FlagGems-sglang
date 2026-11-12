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

"""Benchmark for activation_norm/l2norm."""

import pytest

import flaggems_sglang
from baseline import l2norm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    t, d = plan.builder_args[0]
    args, _ = workload.build_inputs({"t": t, "d": d}, cur_dtype, device)
    # The local benchmark keeps its one-argument ABI (default epsilon).
    return args[:1]


def _case_fn(shape, cur_dtype):
    t, d = shape
    yield BenchmarkCasePlan(
        shape={"x": (t, d)},
        params={"configuration": shape, "seed": 0, "t": t, "d": d},
        builder_args=(shape,),
    )


@pytest.mark.l2norm
def test_perf_l2norm():
    bench = OpBenchmark(
        op_name="l2norm",
        torch_op=get_reference("l2norm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="t, d",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
