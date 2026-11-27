# SPDX-License-Identifier: Apache-2.0
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

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .attri_util import QSA_DECODE_METADATA_SHAPES
from .op_benchmark import OpBenchmark


def _inputs(shape, dtype, device):
    rows, pool_rows, capacity = shape
    lengths = torch.full((rows,), 1024, dtype=torch.int32, device=device)
    requests = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    table = torch.arange(
        pool_rows * capacity, dtype=torch.int32, device=device
    ).reshape(pool_rows, capacity)
    one = lambda: torch.empty_like(lengths)
    state_slots = torch.empty(rows, dtype=torch.int64, device=device)
    yield lengths, requests, table, one(), one(), one(), one(), one(), torch.empty(
        rows, capacity // 64, dtype=torch.int32, device=device
    ), one(), state_slots, torch.empty(
        rows, 4, dtype=torch.int32, device=device
    ), 4, 64


@pytest.mark.qsa_build_decode_metadata
def test_perf_qsa_build_decode_metadata():
    bench = OpBenchmark(
        op_name="qsa_build_decode_metadata",
        torch_op=get_reference("qsa_build_decode_metadata"),
        input_fn=_inputs,
        dtypes=[torch.int32],
        shapes=QSA_DECODE_METADATA_SHAPES,
        shape_desc="decode rows, pool rows, table width",
    )
    bench.set_gems(flaggems_sglang.qsa_build_decode_metadata)
    bench.run()
