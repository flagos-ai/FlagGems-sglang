# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 select_experts workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads("select_experts", comprehensive=True)


@pytest.mark.select_experts
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
)
def test_select_experts(case):
    reference = get_reference("select_experts")
    try:
        candidate = get_candidate("select_experts", reference=reference)
    except ImportError:
        pytest.skip("select_experts: no public candidate on this backend")
    inputs = build_inputs("select_experts", case, torch.bfloat16, cfg.device)
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "select_experts: candidate not implemented on this backend"
        )
    validate_result(
        "select_experts", actual, expected, inputs, expected_inputs, before
    )
