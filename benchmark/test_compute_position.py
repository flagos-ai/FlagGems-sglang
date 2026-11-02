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

"""Benchmark for attention/compute_position."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/compute_position.
SHAPES = [(bs, 512) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 512) for bs in (1024,)]


def _input_fn(shape, cur_dtype, device):
    bs, max_len = shape
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    prefix = torch.randint(
        0, 4096, (bs,), dtype=torch.int32, device=device, generator=g
    )
    yield prefix, seq_lens, int(seq_lens.sum())


def _case_fn(shape, cur_dtype):
    (bs, max_len) = shape
    yield BenchmarkCasePlan(
        shape={"prefix": (bs,), "seq_lens": (bs,)},
        params={
            "configuration": shape,
            "seed": 7,
            "bs": bs,
            "max_len": max_len,
        },
        builder_args=(shape,),
    )


@pytest.mark.compute_position
def test_perf_compute_position():
    bench = OpBenchmark(
        op_name="compute_position",
        torch_op=get_reference("compute_position"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, max_len",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
