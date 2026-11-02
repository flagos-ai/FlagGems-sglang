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

"""Correctness test for attention/linear_decode_forward."""

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
# kernel-comp-baseline/problems/attention/linear_decode_forward/cases.py)
# ---------------------------------------------------------------------------


def _case(
    b,
    h,
    d,
    num_slots=None,
    block=32,
    dtype=torch.float32,
    seed=0,
    pad=False,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(seed)
    num_slots = num_slots or b + 2

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    slots = torch.randperm(num_slots, device="cpu", generator=perm_g)[:b].to(
        device=cfg.device, dtype=torch.int32
    )
    if pad:
        slots[0] = -1  # a padding row: cache and output must be left alone
    return dict(
        q=_t(b, h, 1, d).contiguous(),
        k=_t(b, h, 1, d).contiguous(),
        v=_t(b, h, 1, d).contiguous(),
        kv_caches=_t(num_slots, h, d, d).contiguous(),
        slope_rate=torch.rand(
            h, dtype=torch.float32, device=cfg.device, generator=g
        ),
        slot_idx=slots,
        BLOCK_SIZE=block,
        check=_check,
    )


def _check(actual, expected):
    a_out, a_cache = actual
    e_out, e_cache = expected
    assert_close(a_cache, e_cache)
    assert_close(a_out, e_out)


_CASE_FACTORIES = [
    lambda: _case(1, 4, 64),
    lambda: _case(8, 8, 128),
    lambda: _case(4, 2, 32, block=32),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.linear_decode_forward
def test_linear_decode_forward(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("linear_decode_forward")
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
        candidate = get_candidate("linear_decode_forward", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/linear_decode_forward ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/linear_decode_forward not yet implemented")
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
