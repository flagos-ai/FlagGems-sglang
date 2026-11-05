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

"""Benchmark for attention/fused_softmax_pool."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/fused_softmax_pool.
SHAPES = [(bs, k, 128) for bs in (1, 64, 1024) for k in (4, 16, 64)]
MORE_SHAPES = [(bs, k, 128) for bs in (4096,) for k in (4, 16, 64)]


def _input_fn(shape, cur_dtype, device):
    bs, k, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    # The last axis packs [kv(head_dim), score(head_dim)].
    kv_score = torch.randn(bs, k, 2 * head_dim, device=device, generator=g).to(
        cur_dtype
    )
    yield kv_score.contiguous(), head_dim


def _case_fn(shape, cur_dtype):
    (bs, k, head_dim) = shape
    yield BenchmarkCasePlan(
        shape={"kv_score": (bs, k, 2 * head_dim)},
        params={
            "configuration": shape,
            "seed": 9,
            "bs": bs,
            "k": k,
            "head_dim": head_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.fused_softmax_pool
def test_perf_fused_softmax_pool():
    bench = OpBenchmark(
        op_name="fused_softmax_pool",
        torch_op=get_reference("fused_softmax_pool"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, k, head_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
