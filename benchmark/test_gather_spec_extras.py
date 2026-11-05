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

"""Benchmark for speculative/gather_spec_extras."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/gather_spec_extras.
SHAPES = [(m, 7168) for m in (1, 8, 64, 512)]
MORE_SHAPES = [(m, 7168) for m in (2048,)]

_POOL = 4096
_TOPK = 8


def _input_fn(shape, cur_dtype, device):
    m, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    indices = torch.randperm(_POOL, device="cpu", generator=perm_g)[:m].to(
        device=device, dtype=torch.int64
    )
    topk_p_buf = torch.rand(
        _POOL, _TOPK, dtype=torch.float32, device=device, generator=g
    )
    topk_index_buf = torch.randint(
        0,
        128000,
        (_POOL, _TOPK),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    output_tokens_buf = torch.randint(
        0,
        128000,
        (_POOL,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    hidden_states_buf = torch.randn(
        _POOL, hidden, device=device, generator=g
    ).to(cur_dtype)
    yield (
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
    )


def _case_fn(shape, cur_dtype):
    (m, hidden) = shape
    yield BenchmarkCasePlan(
        shape={
            "indices": (m,),
            "topk_p_buf": (_POOL, _TOPK),
            "topk_index_buf": (_POOL, _TOPK),
            "output_tokens_buf": (_POOL,),
            "hidden_states_buf": (_POOL, hidden),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_POOL": _POOL,
            "_TOPK": _TOPK,
            "m": m,
            "hidden": hidden,
        },
        builder_args=(shape,),
    )


@pytest.mark.gather_spec_extras
def test_perf_gather_spec_extras():
    bench = OpBenchmark(
        op_name="gather_spec_extras",
        torch_op=get_reference("gather_spec_extras"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="m, hidden",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
