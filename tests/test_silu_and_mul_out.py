# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 silu_and_mul_out workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads("silu_and_mul_out", comprehensive=True)


@pytest.mark.silu_and_mul_out
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
)
def test_silu_and_mul_out(case):
    reference = get_reference("silu_and_mul_out")
    try:
        candidate = get_candidate("silu_and_mul_out", reference=reference)
    except ImportError:
        pytest.skip("silu_and_mul_out: no public candidate on this backend")
    inputs = build_inputs("silu_and_mul_out", case, torch.bfloat16, cfg.device)
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "silu_and_mul_out: candidate not implemented on this backend"
        )
    validate_result(
        "silu_and_mul_out", actual, expected, inputs, expected_inputs, before
    )
