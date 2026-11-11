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

"""Benchmark for attention/context_attention."""

import pytest

import flaggems_sglang
from baseline import context_attention as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/context_attention;
# every sequence in a batch gets the same length.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    bs, seq_len, num_heads, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (bs * seq_len, num_heads, head_dim),
            "k": (bs * seq_len, num_heads, head_dim),
            "v": (bs * seq_len, num_heads, head_dim),
            "b_start_loc": (bs,),
            "b_seq_len": (bs,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "IS_CAUSAL": workload.IS_CAUSAL,
            "bs": bs,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        baseline_key=workload.key_shape(
            {
                "bs": bs,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.context_attention
def test_perf_context_attention():
    bench = OpBenchmark(
        op_name="context_attention",
        torch_op=get_reference("context_attention"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        # The prefill reference runs a per-sequence fp32 SDPA.
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, seq_len, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
