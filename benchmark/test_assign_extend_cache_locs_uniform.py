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

"""Benchmark for speculative/assign_extend_cache_locs_uniform."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.utils.rng_utils import create_generator

from .op_benchmark import OpBenchmark

_MAX_BATCH, _POOL_LEN = 4096, 8192
# (batch_size, draft_token_num)
SHAPES = [(b, 8) for b in (1, 8, 64, 512, 2048)]
MORE_SHAPES = [(17, 8), (128, 4), (512, 64)]


def _input_fn(shape, cur_dtype, device):
    bs, draft_token_num = shape
    g = create_generator(device, seed=0)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (_MAX_BATCH, _POOL_LEN),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    pool_idx = torch.randperm(_MAX_BATCH, device=device, generator=g)[:bs].to(
        torch.int32
    )
    start_offset = torch.randint(
        0,
        _POOL_LEN - draft_token_num,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield dict(
        req_pool_indices=pool_idx,
        req_to_token=req_to_token,
        start_offset=start_offset,
        batch_size=bs,
        draft_token_num=draft_token_num,
        device=torch.device(device),
    )


@pytest.mark.assign_extend_cache_locs_uniform
def test_perf_assign_extend_cache_locs_uniform():
    gems_op = flaggems_sglang.get_op("assign_extend_cache_locs_uniform")
    if gems_op is None:
        pytest.skip(
            "speculative/assign_extend_cache_locs_uniform not implemented yet"
        )
    bench = OpBenchmark(
        op_name="assign_extend_cache_locs_uniform",
        torch_op=get_reference("assign_extend_cache_locs_uniform"),
        input_fn=_input_fn,
        dtypes=[torch.int32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch_size, draft_token_num",
    )
    bench.set_gems(gems_op)
    bench.run()
