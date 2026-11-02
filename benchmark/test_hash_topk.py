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
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/hash_topk.
SHAPES = [(t, 256, 8, 1) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 256, 8, 1) for t in (4096,)]

_VOCAB = 32768


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_routed, topk_routed, shared = shape
    g = torch.Generator(device=device).manual_seed(9)
    logits = torch.randn(
        num_tokens, num_routed, device=device, generator=g
    ).to(cur_dtype)
    tid2eid = torch.randint(
        0,
        num_routed,
        (_VOCAB, topk_routed),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    input_ids = torch.randint(
        0, _VOCAB, (num_tokens,), dtype=torch.int32, device=device, generator=g
    )
    # scoring_func is a str, which unpack_to_args_kwargs only forwards as part
    # of a kwargs dict.
    yield logits, input_ids, tid2eid, {
        "num_fused_shared_experts": shared,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sqrtsoftplus",
    }


def _case_fn(shape, cur_dtype):
    (num_tokens, num_routed, topk_routed, shared) = shape
    yield BenchmarkCasePlan(
        shape={
            "logits": (num_tokens, num_routed),
            "input_ids": (num_tokens,),
            "tid2eid": (_VOCAB, topk_routed),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_VOCAB": _VOCAB,
            "num_tokens": num_tokens,
            "num_routed": num_routed,
            "topk_routed": topk_routed,
            "shared": shared,
        },
        builder_args=(shape,),
    )


@pytest.mark.hash_topk
def test_perf_hash_topk():
    bench = OpBenchmark(
        op_name="hash_topk",
        torch_op=get_reference("hash_topk"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_routed, topk_routed, shared",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
