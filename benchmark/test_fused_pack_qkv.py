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

"""Benchmark for diffusion/fused_pack_qkv."""

import pytest

import flaggems_sglang
from baseline import fused_pack_qkv as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _build_inputs(plan, cur_dtype, device):
    bs, seq, num_heads, head_dim = plan.builder_args[0]
    return workload.build_inputs(
        {"bs": bs, "seq": seq, "num_heads": num_heads, "head_dim": head_dim},
        cur_dtype,
        device,
    )[0]


def _case_fn(shape, cur_dtype):
    bs, seq, num_heads, head_dim = shape
    yield BenchmarkCasePlan(
        shape={
            "q": (bs, seq, num_heads, head_dim),
            "k": (bs, seq, num_heads, head_dim),
            "v": (bs, seq, num_heads, head_dim),
            "indices": (max(1, int(bs * seq * workload.KEEP)),),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_KEEP": workload.KEEP,
            "bs": bs,
            "seq": seq,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_pack_qkv
def test_perf_fused_pack_qkv():
    bench = OpBenchmark(
        op_name="fused_pack_qkv",
        torch_op=get_reference("fused_pack_qkv"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="bs, seq, num_heads, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
