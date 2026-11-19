# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import flaggems_sglang
import pytest
import torch
from flaggems_sglang.reference import get_reference

from .attri_util import QSA_DECODE_METADATA_SHAPES
from .op_benchmark import OpBenchmark


def _inputs(shape, dtype, device):
    rows = shape[0]
    lengths = torch.full((rows,), 1024, dtype=torch.int32, device=device)
    requests = torch.arange(rows, dtype=torch.int32, device=device)
    table = torch.arange(
        (rows + 1) * 8192, dtype=torch.int32, device=device
    ).reshape(rows + 1, 8192)
    one = lambda: torch.empty_like(lengths)
    yield lengths, requests, table, one(), one(), one(), one(), one(), torch.empty(
        rows, 64, dtype=torch.int32, device=device
    ), one(), one(), torch.empty(
        rows, 8, dtype=torch.int32, device=device
    ), 8, 128


@pytest.mark.qsa_build_decode_metadata
def test_perf_qsa_build_decode_metadata():
    bench = OpBenchmark(
        op_name="qsa_build_decode_metadata",
        torch_op=get_reference("qsa_build_decode_metadata"),
        input_fn=_inputs,
        dtypes=[torch.int32],
        shapes=QSA_DECODE_METADATA_SHAPES,
        shape_desc="decode rows",
    )
    bench.set_gems(flaggems_sglang.qsa_build_decode_metadata)
    bench.run()
