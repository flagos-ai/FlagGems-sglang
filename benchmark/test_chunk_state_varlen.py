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

"""Benchmark for mamba/chunk_state_varlen."""

import pytest

import flaggems_sglang
from baseline import chunk_state_varlen as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/chunk_state_varlen. The
# first entry is the per-sequence token count; every running cumulative sum
# that ends a chunk has to land exactly on a chunk_size multiple, so no
# sequence straddles a chunk boundary (this problem's documented scope).


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    seq_lens, chunk_size, nheads, ngroups, headdim, dstate = shape
    yield BenchmarkCasePlan(
        shape={
            "b": (sum(seq_lens), ngroups, dstate),
            "x": (sum(seq_lens), nheads, headdim),
            "dt_out": (nheads, sum(seq_lens) // chunk_size, chunk_size),
            "dA_cumsum": (nheads, sum(seq_lens) // chunk_size, chunk_size),
            "cu_seqlens": (len(seq_lens) + 1,),
            "chunk_states": (
                sum(seq_lens) // chunk_size,
                nheads,
                headdim,
                dstate,
            ),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "seq_lens": seq_lens,
            "chunk_size": chunk_size,
            "nheads": nheads,
            "ngroups": ngroups,
            "headdim": headdim,
            "dstate": dstate,
        },
        builder_args=(shape,),
    )


@pytest.mark.chunk_state_varlen
def test_perf_chunk_state_varlen():
    bench = OpBenchmark(
        op_name="chunk_state_varlen",
        torch_op=get_reference("chunk_state_varlen"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="seq_lens, chunk_size, nheads, ngroups, headdim, dstate",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
