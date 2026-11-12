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

"""Correctness test for rope/fused_qk_complex_rope."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

from . import conftest as cfg

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


def check(actual, expected, **kw):
    for a, e in zip(actual, expected):
        assert_close(a, e, **kw)


def _case(t, hq, hk, d, dtype=torch.bfloat16, strided=False, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    p = d // 2
    ang = torch.rand(t, p, device=cfg.device, generator=g) * 6.2831853
    freqs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    if strided:
        bq = torch.randn(
            t, hq, 3 * d, dtype=torch.float32, device=cfg.device,
            generator=g,
        ).to(dtype)
        q = bq[..., :d]
        k = torch.randn(
            t, hk, d, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
    else:
        q = torch.randn(
            t, hq, d, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
        k = torch.randn(
            t, hk, d, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
    return dict(q=q, k=k, freqs_cis=freqs, check=check)


_CASE_FACTORIES = [
    lambda: _case(512, 8, 8, 64),
    lambda: _case(512, 32, 8, 128, dtype=torch.float16),
    lambda: _case(2048, 16, 16, 128),
    lambda: _case(33, 5, 3, 48, dtype=torch.float32),
    lambda: _case(1, 1, 1, 64),
    lambda: _case(256, 16, 4, 64),
    lambda: _case(128, 8, 8, 64, strided=True),
    lambda: _case(4096, 8, 8, 64),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_qk_complex_rope
def test_fused_qk_complex_rope(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_qk_complex_rope")
    check_fn = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import fused_qk_complex_rope
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/fused_qk_complex_rope ops module not found")
        return

    try:
        actual = fused_qk_complex_rope(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/fused_qk_complex_rope not yet implemented")
        return

    if check_fn is not None:
        check_fn(actual, expected)
    else:
        check(actual, expected)
    if check_fn is not None:
        case["check"] = check_fn
