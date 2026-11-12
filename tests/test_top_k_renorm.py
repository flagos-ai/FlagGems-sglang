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

"""Correctness test for sampling/top_k_renorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "top_k_renorm"

# Both sides find the same cutoff bit-exactly, so the only difference left is
# the order the retained mass is summed in. That is a few ulp, not something a
# loose bound would hide.
_RTOL = 1e-4
_ATOL = 1e-6


def _probs(batch, vocab, device, seed=0, dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    raw = torch.rand((batch, vocab), generator=g, device=device)
    normalized = raw / raw.sum(dim=-1, keepdim=True)
    return normalized.to(dtype)


def _check(actual, expected):
    assert (
        actual.shape == expected.shape
    ), f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype
    ), f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


_CASES = [
    # A whole row inside one block, at both ends of the useful k range.
    lambda: (1, 111, 10),
    lambda: (1, 111, 100),
    # Several rows on a mid-sized vocabulary, spanning k below, at and above a
    # realistic sampling width.
    lambda: (4, 1024, 10),
    lambda: (37, 1024, 100),
    lambda: (99, 1024, 500),
    # The real sampling geometry, where the vocabulary spans many blocks.
    lambda: (1, 32000, 10),
    lambda: (4, 32000, 100),
    lambda: (1, 128256, 100),
    # Corner values of k.
    lambda: (8, 1024, 1),
    lambda: (8, 1024, 1024),
]

CASE_IDS = [f"{b}x{v}x{k}" for b, v, k in (c() for c in _CASES)]


@pytest.mark.parametrize("batch,vocab,k", [c() for c in _CASES], ids=CASE_IDS)
@pytest.mark.top_k_renorm
def test_top_k_renorm(batch, vocab, k):
    device = flaggems_sglang.device
    probs = _probs(batch, vocab, device)

    reference = get_reference(OP)
    expected = reference(probs, k)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("sampling/top_k_renorm ops module not found")
        return

    try:
        actual = candidate(probs, k)
    except NotImplementedError:
        pytest.skip("sampling/top_k_renorm not yet implemented")
        return

    _check(actual, expected)


@pytest.mark.top_k_renorm
def test_top_k_renorm_retains_ties_at_the_cutoff():
    """A row with repeated values keeps every entry equal to the cutoff."""
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    # One dominant value, then nine values tied at 0.05, then noise below them.
    vocab = 64
    row = torch.full((1, vocab), 0.001, dtype=torch.float32, device=device)
    row[0, 0] = 0.5
    row[0, 1:10] = 0.05
    probs = row / row.sum(dim=-1, keepdim=True)

    out = op(probs, 5)

    kept = (out[0] > 0).sum().item()
    # k=5 would keep one entry under rank semantics; the cutoff is the 5th
    # largest, which is 0.05, and all nine copies of it are retained.
    assert kept == 10, f"expected 10 retained entries, got {kept}"
    assert torch.all(out[0, 1:10] > 0), "a tied entry was dropped"
    assert torch.all(out[0, 10:] == 0), "an entry below the cutoff survived"
    torch.testing.assert_close(
        out.sum(dim=-1), torch.ones(1, device=device), rtol=1e-6, atol=1e-6
    )


@pytest.mark.top_k_renorm
def test_top_k_renorm_rows_sum_to_one():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = _probs(32, 4096, device, seed=3)
    out = op(probs, 200)

    torch.testing.assert_close(
        out.sum(dim=-1),
        torch.ones(32, device=device),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.top_k_renorm
def test_top_k_renorm_below_cutoff_is_exactly_zero():
    """The masked entries are written as zeros, not as denormal noise."""
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = _probs(8, 2048, device, seed=4)
    out = op(probs, 50)

    zeros = (out == 0).sum(dim=-1)
    assert torch.all(zeros >= 2048 - 50)


@pytest.mark.top_k_renorm
def test_top_k_renorm_k_above_vocab_keeps_the_whole_row():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = _probs(4, 111, device, seed=5)
    out = op(probs, 2000)

    expected = probs.float() / probs.float().sum(dim=-1, keepdim=True)
    torch.testing.assert_close(out, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.top_k_renorm
def test_top_k_renorm_k_zero_keeps_only_the_maximum():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = _probs(4, 512, device, seed=6)
    out = op(probs, 0)

    row_max = probs.max(dim=-1).values
    kept = (out > 0).sum(dim=-1)
    assert torch.all(kept >= 1)
    for r in range(4):
        assert torch.all(out[r][out[r] > 0] > 0)
        assert out[r].sum().item() > 0
        # everything retained must equal the row maximum
        mask = out[r] > 0
        assert torch.all(probs[r][mask] == row_max[r])


@pytest.mark.top_k_renorm
def test_top_k_renorm_per_row_tensor():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = _probs(8, 1024, device, seed=7)
    top_k = torch.tensor([1, 10, 100, 500, 1024, 3, 37, 99], device=device)
    out = op(probs, top_k)

    reference = get_reference(OP)
    _check(out, reference(probs, top_k))

    # A one-element tensor broadcasts across every row.
    out_bcast = op(probs, torch.tensor([100], device=device))
    _check(out_bcast, reference(probs, 100))


@pytest.mark.top_k_renorm
def test_top_k_renorm_mixed_precision_input_still_returns_fp32():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    for dtype in (torch.bfloat16, torch.float16):
        probs = _probs(4, 1024, device, seed=8, dtype=dtype)
        out = op(probs, 100)
        assert out.dtype == torch.float32
        reference = get_reference(OP)
        _check(out, reference(probs, 100))


@pytest.mark.top_k_renorm
def test_top_k_renorm_empty_batch_and_vocab():
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    empty_batch = torch.empty(0, 512, dtype=torch.float32, device=device)
    assert op(empty_batch, 10).shape == (0, 512)

    empty_vocab = torch.empty(4, 0, dtype=torch.float32, device=device)
    assert op(empty_vocab, 10).shape == (4, 0)


@pytest.mark.top_k_renorm
@pytest.mark.parametrize("mutation", ["ndim", "top_k_count", "top_k_type"])
def test_top_k_renorm_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.top_k_renorm

    probs = torch.rand(4, 512, dtype=torch.float32, device=device)

    if mutation == "ndim":
        with pytest.raises(ValueError):
            op(probs[0], 10)
    elif mutation == "top_k_count":
        with pytest.raises(ValueError):
            op(probs, torch.tensor([10, 20], device=device))
    else:
        with pytest.raises(ValueError):
            op(probs, 10.0)
