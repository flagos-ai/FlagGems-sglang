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

"""Benchmark for fla/layernorm_gated."""

import pytest

import flaggems_sglang
from baseline import fla_layernorm_gated as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/fla/layernorm_gated.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    t, d, activation, is_rms_norm, has_bias = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (t, d),
            "gate": (t, d),
            "weight": (d,),
            "bias": (d,) if has_bias else None,
        },
        params={
            "configuration": shape,
            "seed": 0,
            "_EPS": workload._EPS,
            "t": t,
            "d": d,
            "activation": activation,
            "is_rms_norm": is_rms_norm,
            "has_bias": has_bias,
        },
        baseline_key=workload.key_shape(
            {
                "t": t,
                "d": d,
                "activation": activation,
                "is_rms_norm": is_rms_norm,
                "has_bias": has_bias,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.fla_layernorm_gated
def test_perf_fla_layernorm_gated():
    bench = OpBenchmark(
        op_name="fla_layernorm_gated",
        torch_op=get_reference("fla_layernorm_gated"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="t, d, activation, is_rms_norm, has_bias",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
