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

import pytest
import torch

import flaggems_sglang as gems

from .dsv41_helpers import (
    assert_exact,
    changed_graph,
    fp4_inputs,
    fp4_reference,
    fp4_serving_graph_inputs,
)

pytestmark = [
    pytest.mark.dsv41_fp4_indexer,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0),
        reason="SM90 operator",
    ),
]


@pytest.mark.parametrize(
    "b", [6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 192]
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_grouped_and_persistent_changed_graph(b, ratio):
    args = fp4_inputs(b, 65536, ratio)
    fn = lambda *x: gems.fp4_index_logits_req_to_token(
        *x, candidate_block_size=8, write_logits=False, query_group_size=6
    )
    baseline = lambda *x: gems.fp4_index_logits_req_to_token(
        *x, candidate_block_size=8, write_logits=False, query_group_size=1
    )
    assert_exact(fn(*args), baseline(*args))

    def refresh(xs, seed):
        # Mutate actual device data and per-replay request/length metadata.
        xs[0].copy_(torch.roll(xs[0], seed + 1, 1))
        xs[1].mul_(0.9375)
        xs[3].copy_(torch.roll(xs[3], 6))
        xs[4].copy_(
            torch.arange(b, device="cuda")
            .remainder(5)
            .mul(17)
            .neg()
            .add(65536)
        )

    changed_graph(fn, args, refresh, baseline)


@pytest.mark.parametrize("width", [0, 1, 63, 64, 65, 127, 129])
@pytest.mark.parametrize("ratio", [1, 2])
def test_independent_packed_page_reference(width, ratio):
    args = fp4_inputs(6, width, ratio)
    actual = gems.fp4_index_logits_req_to_token(*args, query_group_size=6)
    if width:
        expected = fp4_reference(*args)
        torch.testing.assert_close(
            actual, expected, rtol=0.008, atol=0.03125, equal_nan=True
        )
    else:
        assert actual.shape == (6, 0)


def test_per_query_fallback_odd_rows():
    args = fp4_inputs(7, 129)
    assert_exact(
        gems.fp4_index_logits_req_to_token(*args, query_group_size=6),
        gems.fp4_index_logits_req_to_token(*args),
    )


@pytest.mark.parametrize("ratio", [1, 2])
def test_compact_candidate_consumer_matches_independent_reference(ratio):
    args = fp4_inputs(6, 129, ratio)
    q, weights, mapping, req, _, table, page, _, _ = args
    blocks = torch.tensor(
        [[0, 2, 4, 6]], dtype=torch.int32, device="cuda"
    ).repeat(6, 1)
    lens = torch.tensor(
        [0, 1, 7, 15, 24, 32], dtype=torch.int64, device="cuda"
    )
    actual = gems.fp4_index_logits_candidate_blocks(
        q, weights, mapping, req, blocks, lens, table, page, ratio, 8
    )
    logical = blocks.long().repeat_interleave(8, 1) * 8 + torch.arange(
        32, device="cuda"
    ).remainder(8)
    expected = fp4_reference(*args).gather(1, logical)
    expected.masked_fill_(
        torch.arange(32, device="cuda")[None, :] >= lens[:, None], -torch.inf
    )
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.03125)


@pytest.mark.parametrize("rows", [6, 48, 120])
@pytest.mark.parametrize("ratio", [1, 2])
def test_actual_serving_graph_capacity(rows, ratio):
    args = fp4_serving_graph_inputs(rows, ratio)

    def call(group):
        return gems.fp4_index_logits_req_to_token(
            *args,
            candidate_block_size=8 if ratio == 1 else 0,
            write_logits=ratio != 1,
            query_group_size=group,
        )

    assert_exact(call(6), call(1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(6)
    for seed in range(2):
        args[0].copy_(torch.roll(args[0], 1, 1))
        args[4].sub_(3 + seed)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(out, call(1))
