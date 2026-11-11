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

"""Correctness test for per_token_group_quant_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "per_token_group_quant_fp8"

# Recovered values can differ by one E4M3 quantum, and that quantum is 32x the
# scale near the top of the range (3 mantissa bits), so an upper bound written
# in units of the scale is wrong. 0.2 is the bound sglang's own manual check
# uses for its E4M3 round-trip.
_Q_RTOL = 0.2
_Q_ATOL = 0.2
# The scales themselves are `amax / 448` on both sides, but the compiled kernel
# folds that division into a multiply by the reciprocal, and 1/448 is not
# exactly representable, so the two can differ by one ulp. A relative bound of
# 1e-6 covers that and still catches a reciprocal written the wrong way, which
# is off by 448 squared.
_S_RTOL = 1.0e-6
_S_ATOL = 0.0


def _make(shape, group_size, device, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype)


def _check(actual, expected):
    aq, as_ = actual
    eq, es = expected

    assert (
        aq.shape == eq.shape
    ), f"quantized shape {tuple(aq.shape)} != {tuple(eq.shape)}"
    assert (
        as_.shape == es.shape
    ), f"scale shape {tuple(as_.shape)} != {tuple(es.shape)}"
    assert aq.dtype == torch.float8_e4m3fn
    assert as_.dtype == torch.float32
    assert es.dtype == torch.float32

    # Dequantize both sides before comparing: fp8 bit patterns are only
    # meaningful through their real value, and a one-code difference at the
    # top of the range is a legitimate 32-scale gap.
    torch.testing.assert_close(
        aq.float(), eq.float(), rtol=_Q_RTOL, atol=_Q_ATOL
    )
    torch.testing.assert_close(as_, es, rtol=_S_RTOL, atol=_S_ATOL)


_CASES = [
    # The production geometry.
    lambda: ((1, 7168), 128),
    lambda: ((32, 7168), 128),
    lambda: ((256, 2048), 128),
    # Every supported group size.
    lambda: ((16, 1024), 16),
    lambda: ((16, 1024), 32),
    lambda: ((16, 1024), 64),
    lambda: ((16, 1024), 256),
    # A single group, and a group covering the whole row.
    lambda: ((1, 128), 128),
    lambda: ((4, 512), 512),
    # Higher rank inputs keep the same last-dimension grouping.
    lambda: ((2, 8, 1024), 128),
    # A group size that is not a power of two takes the masked branch.
    lambda: ((8, 96), 48),
]

CASE_IDS = [f"{s}x{g}" for s, g in (c() for c in _CASES)]


@pytest.mark.parametrize(
    "shape,group_size", [c() for c in _CASES], ids=CASE_IDS
)
@pytest.mark.per_token_group_quant_fp8
def test_per_token_group_quant_fp8(shape, group_size):
    device = flaggems_sglang.device
    x = _make(shape, group_size, device)

    reference = get_reference(OP)
    expected = reference(x, group_size)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("per_token_group_quant_fp8 ops module not found")
        return

    try:
        actual = candidate(x, group_size)
    except NotImplementedError:
        pytest.skip("per_token_group_quant_fp8 not yet implemented")
        return

    _check(actual, expected)


@pytest.mark.per_token_group_quant_fp8
def test_scales_are_amax_over_448_exactly():
    """The emitted scale is the small value ``amax / 448``, not its reciprocal."""
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = torch.zeros((4, 256), dtype=torch.float32, device=device)
    x[0, :128] = 448.0
    x[1, :128] = 2.0
    x[2, :128] = 1.0e-3
    x[3, :128] = 0.0
    x[:, 128:] = 0.0

    q, s = op(x, 128)

    assert s.shape == (4, 2)
    torch.testing.assert_close(
        s[0, 0], torch.tensor(1.0, device=device), rtol=_S_RTOL, atol=_S_ATOL
    )
    torch.testing.assert_close(
        s[1, 0],
        torch.tensor(2.0 / 448.0, device=device),
        rtol=_S_RTOL,
        atol=_S_ATOL,
    )
    torch.testing.assert_close(
        s[2, 0],
        torch.tensor(1.0e-3 / 448.0, device=device),
        rtol=_S_RTOL,
        atol=_S_ATOL,
    )
    # An all-zero group floors the maximum, so the scale stays finite and
    # positive instead of collapsing to zero or nan.
    torch.testing.assert_close(
        s[3, 0],
        torch.tensor(1.0e-10 / 448.0, device=device),
        rtol=_S_RTOL,
        atol=_S_ATOL,
    )
    # The all-zero group's factor is still applied to zeros, which quantize to
    # zeros, so no saturating constant appears there.
    assert torch.all(q[3].float() == 0)


@pytest.mark.per_token_group_quant_fp8
def test_saturated_group_hits_the_finite_maximum():
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = torch.full((2, 128), 448.0, dtype=torch.float32, device=device)
    q, s = op(x, 128)

    torch.testing.assert_close(
        s[:, 0], torch.ones(2, device=device), rtol=0, atol=0
    )
    assert torch.all(q.float() == 448.0)


@pytest.mark.per_token_group_quant_fp8
def test_zero_group_does_not_divide_by_zero():
    """Without the absmax floor this becomes nan; the floor keeps it at zero."""
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = torch.zeros((8, 512), dtype=torch.float32, device=device)
    q, s = op(x, 128)

    assert torch.all(q.float() == 0)
    assert torch.all(torch.isfinite(s))
    assert torch.all(s > 0)


@pytest.mark.per_token_group_quant_fp8
def test_scale_count_matches_group_count():
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = _make((3, 640), 128, device)
    q, s = op(x, 128)
    assert q.shape == (3, 640)
    assert s.shape == (3, 5)


@pytest.mark.per_token_group_quant_fp8
def test_one_dimensional_input():
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = _make((512,), 128, device)
    q, s = op(x, 128)

    assert q.shape == (512,)
    assert s.shape == (4,)

    reference = get_reference(OP)
    _check((q, s), reference(x, 128))


@pytest.mark.per_token_group_quant_fp8
def test_empty_group_is_handled_without_launching():
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = torch.empty((0, 128), dtype=torch.float32, device=device)
    q, s = op(x, 128)
    assert q.shape == (0, 128)
    assert s.shape == (0, 1)


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize(
    "mutation", ["not_divisible", "bad_group", "bad_dtype"]
)
def test_per_token_group_quant_fp8_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.per_token_group_quant_fp8

    x = torch.randn(4, 100, dtype=torch.float32, device=device)

    if mutation == "not_divisible":
        with pytest.raises(ValueError):
            op(x, 128)
    elif mutation == "bad_group":
        with pytest.raises(ValueError):
            op(x, 0)
    else:
        with pytest.raises(ValueError):
            op(x, 100, torch.int8)
