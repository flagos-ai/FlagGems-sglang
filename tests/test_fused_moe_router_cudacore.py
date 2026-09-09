# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness test for moe/fused_moe_router_cudacore."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_moe_router_cudacore")


# ---------------------------------------------------------------------------
# Tolerance helper (from kernel-comp-baseline/harness/correctness.py)
# ---------------------------------------------------------------------------

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/moe/fused_moe_router_cudacore/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_w, a_ids = actual
    e_w, e_ids = expected
    assert torch.equal(a_ids.long(), e_ids.long()), "topk expert ids mismatch"
    assert_close(a_w, e_w, dtype=torch.float32)


def _case(
    bs,
    hidden_dim,
    num_experts,
    topk,
    softcap=0.0,
    has_bias=False,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        bs,
        hidden_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    router_weight = torch.randn(
        num_experts,
        hidden_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    bias = None
    if has_bias:
        bias = (
            torch.randn(
                num_experts,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
            * 0.1
        )
    return dict(
        x=x,
        router_weight=router_weight,
        topk=topk,
        moe_softcapping=softcap,
        correction_bias=bias,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 64, 8, 1),
    _case(37, 256, 16, 2, softcap=30.0),
    _case(83, 512, 32, 3, has_bias=True),
]

BENCH_CASES = [_case(m, 4096, 256, 8) for m in (1, 8, 64, 512, 4096)]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_moe_router_cudacore
def test_fused_moe_router_cudacore(case_idx):
    case = CORRECTNESS_CASES[case_idx]
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        from flaggems_sglang.ops.fused_moe_router_cudacore import (
            fused_moe_router_cudacore,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_moe_router_cudacore ops module not found")
        return

    try:
        actual = fused_moe_router_cudacore(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_moe_router_cudacore not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
