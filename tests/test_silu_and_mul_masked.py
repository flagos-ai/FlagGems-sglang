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

"""Correctness test for moe/silu_and_mul_masked."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("silu_and_mul_masked")


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
# Cases (from kernel-comp-baseline/problems/moe/silu_and_mul_masked)
# ---------------------------------------------------------------------------


def _check_factory(masked_m):
    # Rows past ``masked_m[e]`` are not part of the contract, so only the
    # valid prefix of each expert is compared.
    def _check(actual, expected):
        E = masked_m.shape[0]
        for e in range(E):
            n = int(masked_m[e].item())
            if n <= 0:
                continue
            assert_close(actual[e, :n], expected[e, :n])

    return _check


def _case(
    expert_num,
    token_num_padded,
    hidden_dim,
    counts,
    dtype=torch.bfloat16,
    seed=0,
):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)
    input = torch.randn(
        expert_num,
        token_num_padded,
        hidden_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)
    return dict(input=input, masked_m=masked_m, check=_check_factory(masked_m))


CORRECTNESS_CASES = [
    _case(4, 16, 32, [0, 3, 16, 9]),
    _case(8, 128, 256, [128, 0, 64, 1, 17, 128, 5, 90]),
]

BENCH_CASES = [_case(e, 256, 4096, [256] * e) for e in (4, 8, 32)]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.silu_and_mul_masked
def test_silu_and_mul_masked(case_idx):
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
        from flaggems_sglang import silu_and_mul_masked
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/silu_and_mul_masked ops module not found")
        return

    try:
        actual = silu_and_mul_masked(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/silu_and_mul_masked not yet implemented")
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
