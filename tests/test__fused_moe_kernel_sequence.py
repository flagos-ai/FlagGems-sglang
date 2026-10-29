# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the Qwen3.6 _fused_moe_kernel_sequence workload interface."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.qwen36_workloads import build_inputs, workloads

from . import conftest as cfg
from .qwen36_utils import clone_inputs, validate_result

CASES = workloads("_fused_moe_kernel_sequence", comprehensive=True)


@pytest.mark.underscore_fused_moe_kernel_sequence
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        f"{row['model']}-{row['tokens']}-{row['role']}-{row['variant']}"
        for row in CASES
    ],
)
def test__fused_moe_kernel_sequence(case):
    reference = get_reference("_fused_moe_kernel_sequence")
    try:
        candidate = get_candidate(
            "_fused_moe_kernel_sequence", reference=reference
        )
    except ImportError:
        pytest.skip(
            "_fused_moe_kernel_sequence: no public candidate on this backend"
        )
    inputs = build_inputs(
        "_fused_moe_kernel_sequence", case, torch.bfloat16, cfg.device
    )
    before = clone_inputs(inputs)
    expected_inputs = clone_inputs(inputs, "cpu" if cfg.TO_CPU else None)
    expected = reference(**expected_inputs)
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(
            "_fused_moe_kernel_sequence: candidate not implemented on this backend"
        )
    validate_result(
        "_fused_moe_kernel_sequence",
        actual,
        expected,
        inputs,
        expected_inputs,
        before,
    )
