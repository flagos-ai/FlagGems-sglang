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

"""Correctness test for moe/swiglu_oai_mxfp8_quant."""

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


def _check(actual, expected):
    a_q, a_s = actual
    e_q, e_s = expected
    # UE8M0 exponents and a power-of-two divide: both outputs are exact.
    assert_close(a_s, e_s)
    assert_close(a_q.view(torch.uint8), e_q.view(torch.uint8))


def _case(m, n_inter, dtype=torch.bfloat16, seed=0, limit=None):
    g = create_generator(cfg.device, seed=seed)
    gate_up = (
        2.0
        * torch.randn(
            m,
            2 * n_inter,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        )
    ).to(dtype)
    return dict(
        gate_up=gate_up.contiguous(),
        alpha=1.702,
        beta=1.0,
        limit=limit,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 128),
    lambda: _case(1024, 2048),
    lambda: _case(333, 1024, limit=7.0),
    lambda: _case(256, 512),
] + [
    (lambda m, n: lambda: _case(m, n))(m, n)
    for m in (1, 8, 64, 512, 4096)
    for n in (1024, 4096, 12288)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.swiglu_oai_mxfp8_quant
def test_swiglu_oai_mxfp8_quant(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("swiglu_oai_mxfp8_quant")
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
        candidate = get_candidate(
            "swiglu_oai_mxfp8_quant", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/swiglu_oai_mxfp8_quant ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/swiglu_oai_mxfp8_quant not yet implemented")
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
