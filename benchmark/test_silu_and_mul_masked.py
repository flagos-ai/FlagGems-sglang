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

"""Benchmark for moe/silu_and_mul_masked."""

import pytest

import flaggems_sglang
from baseline import silu_and_mul_masked as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/silu_and_mul_masked. The
# last entry is the per-expert valid token count; the benchmark shapes run
# every expert full so the timing covers the whole padded buffer.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    expert_num, token_num_padded, hidden_dim, counts = shape
    yield BenchmarkCasePlan(
        shape={
            "input": (expert_num, token_num_padded, hidden_dim),
            "masked_m": (len(counts),),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "expert_num": expert_num,
            "token_num_padded": token_num_padded,
            "hidden_dim": hidden_dim,
            "counts": counts,
        },
        baseline_key=workload.key_shape(
            {
                "E": expert_num,
                "T": token_num_padded,
                "H": hidden_dim,
                "counts": counts,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.silu_and_mul_masked
def test_perf_silu_and_mul_masked():
    bench = OpBenchmark(
        op_name="silu_and_mul_masked",
        torch_op=get_reference("silu_and_mul_masked"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="expert_num, token_num_padded, hidden_dim, masked_m",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
