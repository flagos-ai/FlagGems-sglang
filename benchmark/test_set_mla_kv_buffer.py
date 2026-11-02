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

"""Benchmark for kvcache/set_mla_kv_buffer."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/set_mla_kv_buffer.
SHAPES = [(n, 512, 64) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(n, 512, 64) for n in (16384,)]

_NUM_SLOTS = 65536


def _input_fn(shape, cur_dtype, device):
    n_loc, nope_dim, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    kv_buffer = torch.zeros(
        _NUM_SLOTS, nope_dim + rope_dim, dtype=cur_dtype, device=device
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    loc = (
        torch.randperm(
            _NUM_SLOTS, generator=torch.Generator(device="cpu").manual_seed(9)
        )[:n_loc]
        .to(torch.int64)
        .to(device)
    )
    nope = torch.randn(n_loc, nope_dim, device=device, generator=g).to(
        cur_dtype
    )
    rope = torch.randn(n_loc, rope_dim, device=device, generator=g).to(
        cur_dtype
    )
    yield kv_buffer, loc, nope, rope


def _case_fn(shape, cur_dtype):
    (n_loc, nope_dim, rope_dim) = shape
    yield BenchmarkCasePlan(
        shape={
            "kv_buffer": (_NUM_SLOTS, nope_dim + rope_dim),
            "loc": (n_loc,),
            "nope": (n_loc, nope_dim),
            "rope": (n_loc, rope_dim),
        },
        params={
            "configuration": shape,
            "seed": 9,
            "_NUM_SLOTS": _NUM_SLOTS,
            "n_loc": n_loc,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        },
        builder_args=(shape,),
    )


@pytest.mark.set_mla_kv_buffer
def test_perf_set_mla_kv_buffer():
    bench = OpBenchmark(
        op_name="set_mla_kv_buffer",
        torch_op=get_reference("set_mla_kv_buffer"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="n_loc, nope_dim, rope_dim",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
