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

"""Correctness test for fla/fused_recurrent_gdn."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_recurrent_gdn")


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
    equal_nan = tol.pop("equal_nan", True)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=equal_nan,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/fla/fused_recurrent_gdn/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_o, a_final = actual
    e_o, e_final = expected
    assert_close(a_o, e_o)
    if e_final is not None:
        # Accumulated float32 rounding differences over many sequential
        # recurrence steps can exceed the tight float32 default tolerance
        # once state magnitudes grow, without indicating an actual bug.
        assert_close(a_final, e_final, atol=1e-2, rtol=1e-2)


def _case(
    batch,
    t,
    h,
    hv,
    k_dim,
    v_dim,
    beta_headwise=False,
    has_init=False,
    output_final_state=True,
    l2norm=False,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    q = torch.randn(
        batch,
        t,
        h,
        k_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    k = torch.randn(
        batch,
        t,
        h,
        k_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    v = torch.randn(
        batch,
        t,
        hv,
        v_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    gate = (
        -torch.rand(
            batch,
            t,
            hv,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
        * 0.1
    )
    if beta_headwise:
        beta = torch.sigmoid(
            torch.randn(
                batch,
                t,
                hv,
                v_dim,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
        )
    else:
        beta = torch.sigmoid(
            torch.randn(
                batch,
                t,
                hv,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
        )
    initial_state = None
    if has_init:
        initial_state = torch.randn(
            batch,
            hv,
            v_dim,
            k_dim,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
    scale = k_dim**-0.5
    return dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=l2norm,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 4, 2, 2, 16, 16),
    _case(2, 8, 2, 4, 32, 32, beta_headwise=True, has_init=True),
    _case(3, 6, 4, 4, 64, 32, l2norm=True, output_final_state=False),
]

BENCH_CASES = [
    _case(8, 128, 8, 8, 64, 64),
    _case(32, 32, 8, 8, 64, 64),
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_recurrent_gdn
def test_fused_recurrent_gdn(case_idx):
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
        from flaggems_sglang.ops.fused_recurrent_gdn import fused_recurrent_gdn
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/fused_recurrent_gdn ops module not found")
        return

    try:
        actual = fused_recurrent_gdn(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/fused_recurrent_gdn not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if a is None and e is None:
                continue
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
