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

"""Correctness test for speculative/wide_row_softmax."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
    # NaN at the same position is a match (both sides overflowed the same
    # way); NaN vs a real value still fails.
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
# Cases (from
# kernel-comp-baseline/problems/speculative/wide_row_softmax/cases.py)
# ---------------------------------------------------------------------------


def _case(bs, vocab, out_stride=None, temps=False, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    logits = (
        torch.randn(
            bs, vocab, dtype=torch.float32, device=cfg.device, generator=g
        )
        * 4
    )
    out_stride = out_stride or vocab
    out = torch.full(
        (bs, out_stride), -1.0, dtype=torch.float32, device=cfg.device
    )
    temperatures = (
        (
            0.5
            + torch.rand(
                bs, dtype=torch.float32, device=cfg.device, generator=g
            )
        )
        if temps
        else None
    )
    return dict(
        logits=logits.contiguous(),
        temperatures=temperatures,
        out=out,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, dtype=torch.float32, atol=1e-6, rtol=1e-5)


_CASE_FACTORIES = [
    lambda: _case(1, 4096),
    # The wide draft vocab this kernel exists for.
    lambda: _case(4, 131072),
    lambda: _case(8, 32000, temps=True),
    # out row stride wider than the vocab.
    lambda: _case(2, 8192, out_stride=8448),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.wide_row_softmax
def test_wide_row_softmax(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("wide_row_softmax")
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
        candidate = get_candidate("wide_row_softmax", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("speculative/wide_row_softmax ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("speculative/wide_row_softmax not yet implemented")
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
