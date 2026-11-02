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
    "fused_recurrent_gated_delta_rule_packed_decode_plugin", comprehensive=True
)


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode_plugin
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        case_id("fused_recurrent_gated_delta_rule_packed_decode_plugin", row)
        for row in CASES
    ],
)
def test_fused_recurrent_gated_delta_rule_packed_decode_plugin(case):
    reference = get_reference(
        "fused_recurrent_gated_delta_rule_packed_decode_plugin"
    )
    try:
        candidate = get_candidate(
            "fused_recurrent_gated_delta_rule_packed_decode_plugin",
            reference=reference,
        )
    except ImportError:
        pytest.skip(
            "fused_recurrent_gated_delta_rule_packed_decode_plugin: no public candidate on this backend"
        )
    inputs = build_inputs(
        "fused_recurrent_gated_delta_rule_packed_decode_plugin",
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
            "fused_recurrent_gated_delta_rule_packed_decode_plugin: candidate not implemented on this backend"
        )
    validate_result(
        "fused_recurrent_gated_delta_rule_packed_decode_plugin",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
