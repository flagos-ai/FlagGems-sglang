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

"""Benchmark for kvcache/create_chunked_prefix_cache_kv_indices."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/kvcache/create_chunked_prefix_cache_kv_indices.
SHAPES = [(bs, 2048) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 2048) for bs in (128,)]

_MAX_BATCH = 256
_MAX_CONTEXT = 16384


def _input_fn(shape, cur_dtype, device):
    bs, chunk = shape
    g = torch.Generator(device=device).manual_seed(7)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (_MAX_BATCH, _MAX_CONTEXT),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    req_pool_indices = (
        torch.randperm(
            _MAX_BATCH, generator=torch.Generator(device="cpu").manual_seed(7)
        )[:bs]
        .to(torch.int32)
        .to(device)
    )
    seq_lens = torch.randint(
        1, chunk + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    start_idx = torch.randint(
        0,
        _MAX_CONTEXT - chunk - 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.zeros(int(cu[-1]), dtype=torch.int32, device=device)
    yield (
        req_to_token,
        req_pool_indices,
        start_idx,
        seq_lens,
        cu,
        kv_indices,
    )


def _case_fn(shape, cur_dtype):
    (bs, chunk) = shape
    yield BenchmarkCasePlan(
        shape={
            "req_to_token": (_MAX_BATCH, _MAX_CONTEXT),
            "req_pool_indices": (bs,),
            "start_idx": (bs,),
            "seq_lens": (bs,),
            "cu": (bs + 1,),
            "kv_indices": ("sum(seq_lens)",),
        },
        params={
            "configuration": shape,
            "seed": 7,
            "_MAX_BATCH": _MAX_BATCH,
            "_MAX_CONTEXT": _MAX_CONTEXT,
            "bs": bs,
            "chunk": chunk,
        },
        builder_args=(shape,),
    )


@pytest.mark.create_chunked_prefix_cache_kv_indices
def test_perf_create_chunked_prefix_cache_kv_indices():
    bench = OpBenchmark(
        op_name="create_chunked_prefix_cache_kv_indices",
        torch_op=get_reference("create_chunked_prefix_cache_kv_indices"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, chunk",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
