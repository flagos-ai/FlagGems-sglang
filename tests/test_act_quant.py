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

"""Correctness test for dsa/act_quant."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.utils.rng_utils import create_generator

from . import conftest as cfg

# torch_gcu aborts the whole process when casting to these dtypes
# on-device (backend CHECK failure); skip every test in this module.
pytestmark = cfg.cast_skip_mark(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Tolerance helper
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
# Cases
# ---------------------------------------------------------------------------


_FP8_DTYPE = torch.float8_e4m3fn


def _check_factory(block_size):
    def _check(actual, expected):
        aq, asc = actual
        eq, esc = expected
        assert_close(asc, esc, dtype=torch.float32)
        ratio = aq.shape[-1] // asc.shape[-1]
        a_deq = aq.to(torch.float32) * asc.repeat_interleave(ratio, dim=-1)
        e_deq = eq.to(torch.float32) * esc.repeat_interleave(ratio, dim=-1)
        atol, rtol = 2e-2, 2e-2
        mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
        assert (
            mismatch.float().mean() < 1e-2
        ), "too many fp8 rounding-boundary mismatches"

    return _check


def _case(
    m, n, block_size=128, round_scale=False, dtype=torch.bfloat16, seed=0
):
    g = create_generator(cfg.device, seed=seed)
    x = torch.randn(
        m, n, generator=g, device=cfg.device, dtype=torch.float32
    ).to(dtype)
    scale_fmt = "ue8m0" if round_scale else None
    return dict(
        x=x,
        block_size=block_size,
        scale_fmt=scale_fmt,
        check=_check_factory(block_size),
    )


_CASE_FACTORIES = [
    lambda: _case(1, 128),
    lambda: _case(37, 512),
    lambda: _case(83, 1024, round_scale=True),
] + [(lambda m: lambda: _case(m, 4096))(m) for m in (1, 8, 64, 512, 4096)]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.act_quant
def test_act_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("act_quant")
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
        candidate = get_candidate("act_quant", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("dsa/act_quant ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("dsa/act_quant not yet implemented")
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
