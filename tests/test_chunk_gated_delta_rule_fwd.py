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

CASES = workloads("chunk_gated_delta_rule_fwd", comprehensive=True)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case_id("chunk_gated_delta_rule_fwd", row) for row in CASES],
)
def test_chunk_gated_delta_rule_fwd(case):
    reference = get_reference("chunk_gated_delta_rule_fwd")
    try:
        candidate = get_candidate(
            "chunk_gated_delta_rule_fwd", reference=reference
        )
    except ImportError:
        pytest.skip(
            "chunk_gated_delta_rule_fwd: no public candidate on this backend"
        )
    inputs = build_inputs(
        "chunk_gated_delta_rule_fwd", case, torch.bfloat16, cfg.device
    )
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "chunk_gated_delta_rule_fwd: candidate not implemented on this backend"
        )
    validate_result(
        "chunk_gated_delta_rule_fwd",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
