# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public expert GEMM versus the independent vendor reference."""

import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.reference import get_reference

from .musa_cases import expert_inputs
from .op_benchmark import OpBenchmark


@pytest.mark.fused_moe_bf16_gemm
@pytest.mark.skipif(
    gems.vendor_name != "mthreads",
    reason="MUSA-specific operator; plugin PR #125",
)
def test_perf_expert_gemm():
    bench = OpBenchmark(
        op_name="fused_moe_bf16_gemm",
        torch_op=get_reference("fused_moe_bf16_gemm"),
        input_fn=expert_inputs,
        dtypes=[torch.bfloat16],
        shapes=[
            (4, 0, 0),
            (2048, 1, 0),
            (8192, 1, 0),
            (16384, 1, 0),
            (8192, 0, 1),
            (16384, 0, 1),
        ],
        shape_desc="tokens, fused W13, weighted down",
    )
    bench.set_gems(gems.fused_moe_bf16_gemm)
    bench.run()
