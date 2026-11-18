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

"""Benchmark for moe/hash_topk."""

import pytest

import flaggems_sglang
from baseline import hash_topk as workload
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/hash_topk.


def _build_inputs(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return workload.build_case(shape, cur_dtype, device)


def _case_fn(shape, cur_dtype):
    num_tokens, num_routed, topk_routed, shared = shape
    yield BenchmarkCasePlan(
        shape={
            "logits": (num_tokens, num_routed),
            "input_ids": (num_tokens,),
            "tid2eid": (workload._VOCAB, topk_routed),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_VOCAB": workload._VOCAB,
            "num_tokens": num_tokens,
            "num_routed": num_routed,
            "topk_routed": topk_routed,
            "shared": shared,
        },
        baseline_key=workload.key_shape(
            {
                "num_tokens": num_tokens,
                "num_routed": num_routed,
                "topk_routed": topk_routed,
                "shared": shared,
                "vocab": workload._VOCAB,
            }
        ),
        builder_args=(shape,),
    )


@pytest.mark.hash_topk
def test_perf_hash_topk():
    bench = OpBenchmark(
        op_name="hash_topk",
        torch_op=get_reference("hash_topk"),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=workload.DTYPES,
        shapes=workload.SHAPES,
        more_shapes=workload.MORE_SHAPES,
        shape_desc="num_tokens, num_routed, topk_routed, shared",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
