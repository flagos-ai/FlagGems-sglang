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

"""Benchmark for attention/fixup_zero_kv."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/fixup_zero_kv.
SHAPES = [(bs, 16, 128, 4) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [(bs, 16, 128, 4) for bs in (2048,)]

_ZERO_FRAC = 0.3


def _input_fn(shape, cur_dtype, device):
    bs, num_heads, v_head_dim, max_q = shape
    g = torch.Generator(device=device).manual_seed(11)
    q_lens = torch.randint(
        1, max_q + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    cum = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cum[1:] = torch.cumsum(q_lens, dim=0)
    total = int(cum[-1])
    kv_lens = torch.randint(
        1, 4096, (bs,), dtype=torch.int32, device=device, generator=g
    )
    kv_lens[torch.rand(bs, device=device, generator=g) < _ZERO_FRAC] = 0
    out = torch.randn(
        total, num_heads, v_head_dim, device=device, generator=g
    ).to(cur_dtype)
    lse = torch.randn(total, num_heads, device=device, generator=g)
    yield out, lse, kv_lens, cum, int(q_lens.max())


def _case_fn(shape, cur_dtype):
    (bs, num_heads, v_head_dim, max_q) = shape
    yield BenchmarkCasePlan(
        shape={
            "out": ("sum(q_lens)", num_heads, v_head_dim),
            "lse": ("sum(q_lens)", num_heads),
            "kv_lens": (bs,),
            "cum": (bs + 1,),
        },
        params={
            "configuration": shape,
            "seed": 11,
            "_ZERO_FRAC": _ZERO_FRAC,
            "bs": bs,
            "num_heads": num_heads,
            "v_head_dim": v_head_dim,
            "max_q": max_q,
        },
        builder_args=(shape,),
    )


@pytest.mark.fixup_zero_kv
def test_perf_fixup_zero_kv():
    bench = OpBenchmark(
        op_name="fixup_zero_kv",
        torch_op=get_reference("fixup_zero_kv"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, num_heads, v_head_dim, max_q",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
