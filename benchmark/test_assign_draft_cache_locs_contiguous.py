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

"""Benchmark for speculative/assign_draft_cache_locs_contiguous."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/assign_draft_cache_locs_contiguous.
SHAPES = [(bs, 8, 5) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [(bs, 8, 5) for bs in (2048,)]

_MAX_BATCH = 256
_POOL_LEN = 8192


def _input_fn(shape, cur_dtype, device):
    bs, topk, steps = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (_MAX_BATCH, _POOL_LEN),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    req_pool_indices = torch.randperm(
        _MAX_BATCH, device="cpu", generator=perm_g
    )[:bs].to(device=device, dtype=torch.int32)
    seq_lens = torch.randint(
        0,
        _POOL_LEN - topk * steps,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    out_cache_loc = torch.zeros(
        bs * topk * steps, dtype=torch.int32, device=device
    )
    yield req_pool_indices, req_to_token, seq_lens, out_cache_loc, topk, steps


def _case_fn(shape, cur_dtype):
    (bs, topk, steps) = shape
    yield BenchmarkCasePlan(
        shape={
            "req_pool_indices": (min(bs, _MAX_BATCH),),
            "req_to_token": (_MAX_BATCH, _POOL_LEN),
            "seq_lens": (bs,),
            "out_cache_loc": (bs * topk * steps,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_MAX_BATCH": _MAX_BATCH,
            "_POOL_LEN": _POOL_LEN,
            "bs": bs,
            "topk": topk,
            "steps": steps,
        },
        builder_args=(shape,),
    )


@pytest.mark.assign_draft_cache_locs_contiguous
def test_perf_assign_draft_cache_locs_contiguous():
    bench = OpBenchmark(
        op_name="assign_draft_cache_locs_contiguous",
        torch_op=get_reference("assign_draft_cache_locs_contiguous"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, topk, steps",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
