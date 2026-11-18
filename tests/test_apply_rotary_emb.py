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

"""Correctness test for embedding/apply_rotary_emb."""

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


def _case(t, h, d, dtype=torch.bfloat16, pos=True, inverse=False,
          table_extra=False, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    p = d // 2
    s = t + 64 if table_extra else t
    freqs = torch.zeros(s, p, 2, device=cfg.device)
    ang = torch.rand(s, p, device=cfg.device, generator=g) * 6.2831853
    freqs[..., 0] = torch.cos(ang)
    freqs[..., 1] = torch.sin(ang)
    x = torch.randn(
        t, h, d, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    kwargs = dict(x=x, freqs_cis=freqs, inverse=inverse)
    if pos:
        kwargs["positions"] = (
            torch.randperm(s, device=cfg.device, generator=g)[:t]
            if table_extra
            else torch.arange(t, device=cfg.device)
        )
    else:
        kwargs["positions"] = None
    return dict(**kwargs, check=assert_close)


_CASE_FACTORIES = [
    lambda: _case(512, 8, 64),
    lambda: _case(512, 8, 64, inverse=True),
    lambda: _case(128, 32, 128, dtype=torch.float16),
    lambda: _case(2048, 8, 128, table_extra=True),
    lambda: _case(64, 1, 32, dtype=torch.float32),
    lambda: _case(33, 5, 12),
    lambda: _case(1, 4, 8, pos=False),
    lambda: _case(1024, 16, 64, pos=False),
    lambda: _case(256, 8, 64, dtype=torch.float32, inverse=True),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.apply_rotary_emb
def test_apply_rotary_emb(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("apply_rotary_emb")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import apply_rotary_emb
    except (ImportError, ModuleNotFoundError):
        pytest.skip("embedding/apply_rotary_emb ops module not found")
        return

    try:
        actual = apply_rotary_emb(**kwargs)
    except NotImplementedError:
        pytest.skip("embedding/apply_rotary_emb not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    if check is not None:
        case["check"] = check
