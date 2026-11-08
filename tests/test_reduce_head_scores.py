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

"""Correctness test for kv_cache/reduce_head_scores."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)

NUM_HEADS = 32


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype,
            _DEFAULT_TOLERANCE,
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


def _scores(rows, width, seed=0, negative=False, strided=False):
    """Build scores, optionally all-negative or with a non-unit row stride."""
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    if strided:
        # One extra column so the slice is a real non-contiguous view.
        base = torch.randn(
            rows, NUM_HEADS, width + 1, generator=g, device=dev
        ).to(torch.bfloat16)
        scores = base[:, :, :width]
    else:
        scores = torch.randn(
            rows, NUM_HEADS, width, generator=g, device=dev
        ).to(torch.bfloat16)
    if negative:
        scores = -scores.abs()
    return scores


def _weights(rows, seed=0):
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    return torch.randn(rows, NUM_HEADS, generator=g, device=dev).to(
        torch.bfloat16
    )


def _row_major(rows, width, seed=0):
    return {
        "scores": _scores(rows, width, seed),
        "weights": _weights(rows, seed),
    }


def _negative(rows, width, seed=0):
    return {
        "scores": _scores(rows, width, seed, negative=True),
        "weights": _weights(rows, seed),
    }


def _strided(rows, width, seed=0):
    return {
        "scores": _scores(rows, width, seed, strided=True),
        "weights": _weights(rows, seed),
    }


_CASE_FACTORIES = [
    lambda: _row_major(64, 2048),
    lambda: _row_major(512, 2048),
    lambda: _row_major(2048, 2048),
    lambda: _row_major(1, 256),
    # width straddles the 256-column block, so the tail mask is exercised
    lambda: _row_major(3, 100),
    # a single column is the narrowest legal width
    lambda: _row_major(128, 1),
    # every score is negative, so relu zeroes the whole product
    lambda: _negative(64, 512),
    # non-contiguous view: row stride is width + 1, not width
    lambda: _strided(64, 512),
    # an empty row set must return an empty result, not launch a grid
    lambda: _row_major(0, 512),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.reduce_head_scores
def test_reduce_head_scores(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("reduce_head_scores")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        candidate = get_candidate("reduce_head_scores", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kv_cache/reduce_head_scores ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kv_cache/reduce_head_scores not yet implemented")
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
