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

"""Correctness test for per_tensor_quant_mla_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "per_tensor_quant_mla_fp8"

FP8_E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0

# The scale is produced by one division on each side. Both are IEEE divisions,
# so a byte-exact payload should come out; the scale itself is only compared to
# within a relative epsilon because a reciprocal-multiply rewrite anywhere in
# the chain costs one ulp.
_SCALE_RTOL = 1.0e-6


def _pair():
    ref = get_reference(OP)
    try:
        cand = get_candidate(OP)
    except (ImportError, NotImplementedError):
        pytest.skip(f"no candidate implementation for {OP}")
    return ref, cand


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _check(actual, expected):
    got_q, got_s = actual
    want_q, want_s = expected

    assert got_q.shape == want_q.shape
    assert got_s.shape == want_s.shape
    assert got_s.dtype == torch.float32
    # The payload is eight-bit, so it is compared on the encoded bytes and
    # needs no tolerance at all.
    torch.testing.assert_close(
        _bytes(got_q), _bytes(want_q), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(got_s, want_s, rtol=_SCALE_RTOL, atol=0.0)


SHAPES = [
    # The ordinary absorbed case.
    lambda: (16, 8, 64),
    # A single decoded token still gets one scale over all heads.
    lambda: (32, 1, 512),
    # DeepSeek's 576-wide latent, not a power of two, so the tile is padded.
    lambda: (8, 4, 576),
    # Odd head width leaves most of the tile masked out.
    lambda: (4, 3, 100),
    # Two tiles worth of head width.
    lambda: (64, 16, 128),
    # The degenerate single element still has a well-defined scale.
    lambda: (1, 1, 1),
]


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("shape_fn", SHAPES)
def test_per_tensor_quant_mla_fp8(shape_fn):
    ref, cand = _pair()
    shape = shape_fn()
    device = flaggems_sglang.device
    x = torch.randn(shape, device=device, dtype=torch.bfloat16) * 3.0

    expected = ref(x, 1e-12)
    actual = cand(x, 1e-12)
    _check(actual, expected)


@pytest.mark.per_tensor_quant_mla_fp8
def test_scale_is_the_tensor_absmax_over_448():
    """One scale for the whole tensor, derived from the largest magnitude."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 2, 64), device=device, dtype=torch.bfloat16)
    x[1, 0, 7] = 100.0

    got_q, got_s = cand(x, 1e-12)
    amax = x.float().abs().amax()
    torch.testing.assert_close(
        got_s[0], amax / FP8_MAX, rtol=_SCALE_RTOL, atol=0.0
    )
    # The peak is the number that maps onto the FP8 maximum.
    assert (
        _bytes(got_q)[1, 0, 7].item()
        == _bytes(torch.tensor([FP8_MAX], dtype=FP8_E4M3)).item()
    )
    # Every other element was zero, so the payload is that one byte and zeros.
    assert got_q.float().max().item() == FP8_MAX
    assert got_q.float().abs().sum().item() == FP8_MAX


@pytest.mark.per_tensor_quant_mla_fp8
def test_negative_peak_sets_the_scale():
    """The scale follows the magnitude, so a large negative value counts."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 2, 64), device=device, dtype=torch.bfloat16)
    x[2, 1, 13] = -250.0

    expected = ref(x, 1e-12)
    actual = cand(x, 1e-12)
    _check(actual, expected)
    torch.testing.assert_close(
        actual[1][0],
        torch.tensor(250.0, device=device) / FP8_MAX,
        rtol=_SCALE_RTOL,
        atol=0.0,
    )


@pytest.mark.per_tensor_quant_mla_fp8
def test_zeros_use_the_eps_floor():
    """All-zero input would give a zero scale, so the floor takes over."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    eps = 1e-12
    x = torch.zeros((2, 3, 32), device=device, dtype=torch.bfloat16)

    got_q, got_s = cand(x, eps)
    torch.testing.assert_close(
        got_s[0], torch.tensor(eps / FP8_MAX, device=device), rtol=0.0, atol=0.0
    )
    assert got_q.float().eq(0).all()


@pytest.mark.per_tensor_quant_mla_fp8
def test_values_above_the_fp8_range_saturate():
    """The peak maps to 448 by construction; nothing can exceed it."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    # The epsilon floor is what normally keeps the scale above the peak, but an
    # exponent difference of more than 448 has to clamp rather than wrap.
    x = torch.full((2, 2, 16), 1e30, device=device, dtype=torch.float32)
    x[0, 0, 0] = 1e-30

    got_q, _ = cand(x, 1e-12)
    finite = got_q.float()
    assert torch.isfinite(finite).all()
    assert finite.abs().max() <= FP8_MAX


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_input_dtypes(dtype):
    ref, cand = _pair()
    device = flaggems_sglang.device
    x = torch.randn((8, 5, 128), device=device, dtype=dtype) * 2.0

    _check(cand(x, 1e-12), ref(x, 1e-12))


@pytest.mark.per_tensor_quant_mla_fp8
def test_non_contiguous_input():
    """Strides are honoured for the read; the payload is written contiguous."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    x = torch.randn((8, 4, 96), device=device, dtype=torch.bfloat16)
    x = x.transpose(0, 1).contiguous().transpose(0, 1)
    assert not x.is_contiguous()

    got_q, got_s = cand(x, 1e-12)
    assert got_q.is_contiguous()
    _check((got_q, got_s), ref(x, 1e-12))


@pytest.mark.per_tensor_quant_mla_fp8
def test_input_is_not_mutated():
    ref, cand = _pair()
    device = flaggems_sglang.device
    x = torch.randn((4, 2, 64), device=device, dtype=torch.bfloat16)
    before = x.clone()

    cand(x, 1e-12)
    assert torch.equal(x, before)


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("shape", [(0, 4, 128), (3, 0, 64), (5, 2, 0)])
def test_empty_dimension(shape):
    """An empty tensor still answers with a shape and a floored scale."""
    ref, cand = _pair()
    device = flaggems_sglang.device
    eps = 1e-12
    x = torch.empty(shape, device=device, dtype=torch.bfloat16)

    got_q, got_s = cand(x, eps)
    assert got_q.shape == shape
    assert got_q.numel() == 0
    torch.testing.assert_close(
        got_s[0], torch.tensor(eps / FP8_MAX, device=device), rtol=0.0, atol=0.0
    )


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_rejects_a_non_positive_floor(eps):
    """A floor of zero divides a zero row by zero, so it cannot be allowed."""
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((2, 2, 8), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cand(x, eps)


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("shape", [(16, 64), (2, 2, 2, 16)])
def test_rejects_a_non_3d_tensor(shape):
    """The absorbed MLA layout is three-dimensional by definition."""
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cand(x, 1e-12)


@pytest.mark.per_tensor_quant_mla_fp8
@pytest.mark.parametrize("dtype", [torch.int8, torch.float64])
def test_rejects_an_unsupported_dtype(dtype):
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((2, 2, 8), device=device, dtype=dtype)
    with pytest.raises(ValueError):
        cand(x, 1e-12)


@pytest.mark.per_tensor_quant_mla_fp8
def test_rejects_a_cpu_tensor():
    _, cand = _pair()
    x = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cand(x, 1e-12)
