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


@pytest.mark.reduce_head_scores
def test_bf16_halfway_rounding():
    """Replicates upstream's halfway case, which only the right order passes.

    Row 6 sums 2^24, -2^24, 1.0 and 2^-8 over the 32 heads. The two large
    terms cancel, leaving 1.0 + 2^-8 = 1.00390625, which rounds to 1.0 in
    bf16. Summing 2^24 + 1.0 first would lose the 1.0 entirely, so the
    expected values below cannot be produced by an accumulation that skips
    the per-head bf16 cast or reorders the 32 heads.
    """
    dev = flaggems_sglang.device
    scores = torch.ones(8, NUM_HEADS, 257, device=dev, dtype=torch.bfloat16)
    weights = torch.zeros(8, NUM_HEADS, device=dev, dtype=torch.bfloat16)
    weights[0, :2] = torch.tensor([1.0, 2.0**-8], device=dev)
    weights[1, :2] = torch.tensor([1.0 + 2.0**-7, 2.0**-8], device=dev)
    weights[2] = -weights[0]
    weights[3] = -weights[1]
    scores[4:6, 0] = 1.0 + 2.0**-7
    weights[4, :2] = torch.tensor([1.5, -(2.0**-9)], device=dev)
    weights[5] = -weights[4]
    weights[6, :4] = torch.tensor(
        [2.0**24, -(2.0**24), 1.0, 2.0**-8], device=dev
    )
    weights[7] = -weights[6]

    expected = torch.tensor(
        [1.0, 1.015625, -1.0, -1.015625, 1.515625, -1.515625, 1.0, -1.0],
        device=dev,
    )

    candidate = get_candidate("reduce_head_scores")
    actual = candidate(scores, weights)
    assert actual.dtype == torch.float32
    # Scores are all ones except rows 4-5 at column 0, so every column of the
    # output holds the value its row's row-6 pattern produces; upstream
    # compares column 0, which is the column the special scores sit in.
    assert torch.equal(actual[:, 0], expected), (
        f"halfway values differ: got {actual[:, 0].tolist()}, "
        f"expected {expected.tolist()}"
    )


@pytest.mark.parametrize("width", [65537, 2049])
@pytest.mark.reduce_head_scores
def test_wide_widths(width):
    """Widths past the upstream grid's largest point, over the 256-col block.

    65537 is the widest upstream tests; it is one column past a whole
    multiple of 65536, so it lands on the far tail of the mask.
    """
    scores = _scores(3, width, seed=width)
    weights = torch.randn(
        3, NUM_HEADS, generator=None, device=scores.device
    ).to(torch.bfloat16)
    candidate = get_candidate("reduce_head_scores")
    reference = get_reference("reduce_head_scores")
    actual = candidate(scores, weights)
    expected = reference(scores, weights)
    assert torch.equal(actual, expected)


@pytest.mark.reduce_head_scores
def test_head_contiguous_and_transposed_weights():
    """Strides the upstream fallback list assumes this kernel cannot read.

    Upstream falls back to torch for head-contiguous scores and for a
    transposed 2-D weight. Score and weight strides are passed to the kernel
    as runtime scalars, so both should read correctly; assert bit equality
    rather than closeness, because no arithmetic differs.
    """
    dev = flaggems_sglang.device
    rows, heads, width = 5, NUM_HEADS, 257

    head_major = torch.randn(heads, rows, width, device=dev).to(torch.bfloat16)
    scores = head_major.permute(1, 0, 2)
    assert scores.stride() == (width, rows * width, 1)

    weights = torch.randn(rows, heads, device=dev).to(torch.bfloat16)
    candidate = get_candidate("reduce_head_scores")
    reference = get_reference("reduce_head_scores")
    actual = candidate(scores, weights)
    expected = reference(scores, weights)
    assert torch.equal(actual, expected)

    w2 = torch.randn(heads, rows, device=dev).to(torch.bfloat16)
    transposed = w2.t()
    assert transposed.stride() == (1, rows)
    scores2 = _scores(rows, width, seed=11)
    actual2 = candidate(scores2, transposed)
    expected2 = reference(scores2, transposed)
    assert torch.equal(actual2, expected2)
