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

"""Correctness test for quantization/store_sf_interleaved."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

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
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        equal_nan=True,
        **tol,
    )


def _case(num_tokens, nheads=8, num_pages=256, page_size=128, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    sf_in = torch.randint(
        0,
        256,
        (num_tokens, nheads, 4),
        dtype=torch.uint8,
        device=flaggems_sglang.device,
        generator=g,
    )
    sf_out = torch.zeros(
        num_pages,
        nheads,
        32,
        4,
        4,
        dtype=torch.uint8,
        device=flaggems_sglang.device,
    )
    loc = torch.randperm(
        num_pages * page_size, device=flaggems_sglang.device, generator=g
    )[:num_tokens].to(torch.int64)
    return dict(
        sf_in=sf_in,
        sf_out=sf_out,
        loc=loc,
        page_size=page_size,
        check=assert_close,
    )


_CORRECTNESS_CASES = [
    lambda: _case(1),
    lambda: _case(37, seed=1),
    lambda: _case(512, nheads=4, seed=2),
    lambda: _case(4096, seed=3),
]
_CORRECTNESS_CASES += [
    (lambda t=t: _case(t, seed=9)) for t in (1, 8, 64, 512, 4096, 16384)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.store_sf_interleaved
def test_store_sf_interleaved(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("store_sf_interleaved")(**kwargs)

    try:
        from flaggems_sglang.ops.store_sf_interleaved import store_sf_interleaved
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/store_sf_interleaved ops module not found")
        return

    try:
        actual = store_sf_interleaved(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/store_sf_interleaved not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check
