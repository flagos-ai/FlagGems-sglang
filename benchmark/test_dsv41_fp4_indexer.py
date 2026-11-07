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

"""Six-query/persistent index scoring against the native per-query path."""

from functools import partial

import pytest
import torch

import flaggems_sglang as gems
from tests.dsv41_helpers import fp4_serving_graph_inputs

from .op_benchmark import OpBenchmark

ROWS = [6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120]
SHAPES = [(b, 655627 // ratio, ratio) for b in ROWS for ratio in [1, 2]]


def _inputs(shape, cur_dtype, device):
    yield fp4_serving_graph_inputs(shape[0], shape[2])


def scoring(*args, query_group_size):
    ratio = args[7]
    return gems.fp4_index_logits_req_to_token(
        *args,
        candidate_block_size=8 if ratio == 1 else 0,
        write_logits=ratio != 1,
        query_group_size=query_group_size,
    )


@pytest.mark.dsv41_fp4_indexer
def test_perf_dsv41_fp4_indexer():
    baseline = partial(scoring, query_group_size=1)
    candidate = partial(scoring, query_group_size=6)
    bench = OpBenchmark(
        op_name="dsv41_fp4_indexer_native",
        torch_op=baseline,
        input_fn=_inputs,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        shape_desc="B,width,ratio",
    )
    bench.set_gems(candidate)
    bench.run()
