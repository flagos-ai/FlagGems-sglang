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

"""Benchmark for activation_norm/fused_rmsnorm."""

import pytest

import flaggems_sglang
from baseline import fused_rmsnorm as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    bs, hidden = plan.builder_args[0]
    args, _ = workload.build_inputs(
        {"bs": bs, "hidden": hidden}, cur_dtype, device
    )
    # The native SGLang keyword flags are not part of the local ABI.
    return args


def _case_fn(shape, cur_dtype):
    bs, hidden = shape
    yield BenchmarkCasePlan(
        shape={"x": (bs, hidden), "weight": (hidden,)},
        params={
            "configuration": shape,
            "seed": 0,
            "EPS": workload.EPS,
            "bs": bs,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_rmsnorm
def test_perf_fused_rmsnorm():
    bench = OpBenchmark(
        op_name="fused_rmsnorm",
        torch_op=get_reference("fused_rmsnorm"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
