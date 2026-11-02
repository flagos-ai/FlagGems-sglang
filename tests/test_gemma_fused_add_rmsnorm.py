# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.operator_workloads import (
    build_inputs,
    case_id,
    workloads,
)

from . import conftest as cfg
from .operator_assertions import clone_inputs, validate_result

CASES = workloads("gemma_fused_add_rmsnorm", comprehensive=True)


@pytest.mark.gemma_fused_add_rmsnorm
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case_id("gemma_fused_add_rmsnorm", row) for row in CASES],
)
def test_gemma_fused_add_rmsnorm(case):
    reference = get_reference("gemma_fused_add_rmsnorm")
    try:
        candidate = get_candidate(
            "gemma_fused_add_rmsnorm", reference=reference
        )
    except ImportError:
        pytest.skip(
            "gemma_fused_add_rmsnorm: no public candidate on this backend"
        )
    inputs = build_inputs(
        "gemma_fused_add_rmsnorm", case, torch.bfloat16, cfg.device
    )
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "gemma_fused_add_rmsnorm: candidate not implemented on this backend"
        )
    validate_result(
        "gemma_fused_add_rmsnorm",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
