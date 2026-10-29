# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 gemma_rmsnorm workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads("gemma_rmsnorm", comprehensive=True)


@pytest.mark.gemma_rmsnorm
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
)
def test_gemma_rmsnorm(case):
    reference = get_reference("gemma_rmsnorm")
    try:
        candidate = get_candidate("gemma_rmsnorm", reference=reference)
    except ImportError:
        pytest.skip("gemma_rmsnorm: no public candidate on this backend")
    inputs = build_inputs("gemma_rmsnorm", case, torch.bfloat16, cfg.device)
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip("gemma_rmsnorm: candidate not implemented on this backend")
    validate_result(
        "gemma_rmsnorm", actual, expected, inputs, expected_inputs, before
    )
