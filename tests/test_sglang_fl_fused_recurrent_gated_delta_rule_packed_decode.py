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

CASES = workloads(
    "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode",
    comprehensive=True,
)


@pytest.mark.sglang_fl_fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        case_id(
            "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode", row
        )
        for row in CASES
    ],
)
def test_sglang_fl_fused_recurrent_gated_delta_rule_packed_decode(case):
    reference = get_reference(
        "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode"
    )
    try:
        candidate = get_candidate(
            "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode",
            reference=reference,
        )
    except ImportError:
        pytest.skip(
            "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode: no public candidate on this backend"
        )
    inputs = build_inputs(
        "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode",
        case,
        torch.bfloat16,
        cfg.device,
    )
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode: candidate not implemented on this backend"
        )
    validate_result(
        "sglang_fl_fused_recurrent_gated_delta_rule_packed_decode",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
