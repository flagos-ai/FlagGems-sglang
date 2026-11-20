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

"""Correctness test for norm/fused_norm_rope_inplace."""

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


def _case(t, h, d, rot, dtype=torch.bfloat16, weight=True, pos=True,
          eps=1e-6, strided=False, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    npairs = rot // 2
    freqs = torch.zeros(t, npairs, 2, device=cfg.device)
    ang = torch.rand(t, npairs, device=cfg.device, generator=g) * 6.2831853
    freqs[..., 0] = torch.cos(ang)
    freqs[..., 1] = torch.sin(ang)
    if strided:
        base = torch.randn(
            t, h, 3 * d, dtype=torch.float32, device=cfg.device,
            generator=g,
        ).to(dtype)
        kv = base[..., :d]
    else:
        kv = torch.randn(
            t, h, d, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)
    w = (
        torch.randn(d, dtype=torch.float32, device=cfg.device, generator=g)
        .to(dtype)
        if weight
        else None
    )
    positions = (
        torch.arange(t, device=cfg.device) if pos else None
    )
    return dict(
        kv=kv,
        weight=w,
        eps=eps,
        freqs_cis=freqs,
        positions=positions,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(512, 8, 128, 64),
    lambda: _case(512, 8, 128, 64, weight=False),
    lambda: _case(512, 8, 128, 64, pos=False),
    lambda: _case(64, 40, 64, 64),
    lambda: _case(33, 5, 96, 48, dtype=torch.float32),
    lambda: _case(256, 8, 128, 64, strided=True),
    lambda: _case(1024, 8, 64, 32, dtype=torch.float16, eps=1e-5),
    lambda: _case(1, 1, 64, 64),
    lambda: _case(4, 4, 64, 2, dtype=torch.float32),
    lambda: _case(2048, 8, 128, 128),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_norm_rope_inplace
def test_fused_norm_rope_inplace(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_norm_rope_inplace")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        from flaggems_sglang import fused_norm_rope_inplace
    except (ImportError, ModuleNotFoundError):
        pytest.skip("norm/fused_norm_rope_inplace ops module not found")
        return

    try:
        actual = fused_norm_rope_inplace(**kwargs)
    except NotImplementedError:
        pytest.skip("norm/fused_norm_rope_inplace not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    if check is not None:
        case["check"] = check
