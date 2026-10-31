# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 fused_recurrent_gated_delta_rule_packed_decode workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads(
    "fused_recurrent_gated_delta_rule_packed_decode", comprehensive=True
)


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
)
def test_fused_recurrent_gated_delta_rule_packed_decode(case):
    reference = get_reference("fused_recurrent_gated_delta_rule_packed_decode")
    try:
        candidate = get_candidate(
            "fused_recurrent_gated_delta_rule_packed_decode",
            reference=reference,
        )
    except ImportError:
        pytest.skip(
            "fused_recurrent_gated_delta_rule_packed_decode: no public candidate on this backend"
        )
    inputs = build_inputs(
        "fused_recurrent_gated_delta_rule_packed_decode",
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
            "fused_recurrent_gated_delta_rule_packed_decode: candidate not implemented on this backend"
        )
    validate_result(
        "fused_recurrent_gated_delta_rule_packed_decode",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
