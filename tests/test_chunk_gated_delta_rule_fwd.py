# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 chunk_gated_delta_rule_fwd workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads("chunk_gated_delta_rule_fwd", comprehensive=True)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
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
