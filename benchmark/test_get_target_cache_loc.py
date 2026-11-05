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

"""Benchmark for speculative/get_target_cache_loc."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/speculative/get_target_cache_loc.
SHAPES = [(bs, 8) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 8) for bs in (1024,)]


def _input_fn(shape, cur_dtype, device):
    bs, num_verify_tokens = shape
    g = torch.Generator(device=device).manual_seed(9)
    num_correct_drafts = torch.randint(
        0,
        num_verify_tokens,
        (bs,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    to_free_num_slots = torch.randint(
        0,
        num_verify_tokens // 2 + 1,
        (bs,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    out_cache_loc = torch.randint(
        1,
        1 << 20,
        (bs * num_verify_tokens,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    yield (
        num_correct_drafts,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
    )


def _case_fn(shape, cur_dtype):
    (bs, num_verify_tokens) = shape
    yield BenchmarkCasePlan(
        shape={
            "num_correct_drafts": (bs,),
            "to_free_num_slots": (bs,),
            "out_cache_loc": (bs * num_verify_tokens,),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "bs": bs,
            "num_verify_tokens": num_verify_tokens,
        },
        builder_args=(shape,),
    )


@pytest.mark.get_target_cache_loc
def test_perf_get_target_cache_loc():
    bench = OpBenchmark(
        op_name="get_target_cache_loc",
        torch_op=get_reference("get_target_cache_loc"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, num_verify_tokens",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
