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

"""Benchmark for mamba/causal_conv1d_fn."""

import pytest

import flaggems_sglang
from baseline import causal_conv1d_fn as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/causal_conv1d_fn.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_seqs, seq_len, dim, width = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (dim, num_seqs * seq_len),
            "weight": (dim, width),
            "bias": (dim,),
            "query_start_loc": (num_seqs + 1,),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "num_seqs": num_seqs,
            "seq_len": seq_len,
            "dim": dim,
            "width": width,
        },
        baseline_key=workload.key_shape(
            {
                "num_seqs": num_seqs,
                "seq_len": seq_len,
                "dim": dim,
                "width": width,
                "seq_lens": [seq_len] * num_seqs,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.causal_conv1d_fn
def test_perf_causal_conv1d_fn():
    bench = OpBenchmark(
        op_name="causal_conv1d_fn",
        torch_op=get_reference("causal_conv1d_fn"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        timing_reset_args=(),
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_seqs, seq_len, dim, width",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
