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

"""Benchmark for diffusion/pack_qkv_destination_major."""

import pytest

import flaggems_sglang
from baseline import pack_qkv_destination_major as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    rows, global_heads, head_size, world_size = plan.builder_args[0]
    return workload.build_inputs(
        {
            "rows": rows,
            "global_heads": global_heads,
            "head_size": head_size,
            "world_size": world_size,
        },
        cur_dtype,
        device,
    )[0]


def _case_fn(shape, cur_dtype):
    rows, global_heads, head_size, world_size = shape
    yield BenchmarkCasePlan(
        shape={
            "src": (rows, global_heads, head_size),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "rows": rows,
            "global_heads": global_heads,
            "head_size": head_size,
            "world_size": world_size,
        },
        builder_args=(shape,),
    )


@pytest.mark.pack_qkv_destination_major
def test_perf_pack_qkv_destination_major():
    bench = OpBenchmark(
        op_name="pack_qkv_destination_major",
        torch_op=get_reference("pack_qkv_destination_major"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="rows, global_heads, head_size, world_size",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
