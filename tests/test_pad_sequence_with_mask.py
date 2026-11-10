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

"""Correctness test for attention/pad_sequence_with_mask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "pad_sequence_with_mask"


def _case(batch, max_len, hidden, lengths, seed=0, dtype=torch.float32):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)

    starts = []
    total = 0
    for n in lengths:
        starts.append(total)
        total += n

    offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

    if dtype == torch.float8_e4m3fn:
        # Integers exactly representable in e4m3, so byte equality is the
        # right assertion rather than a tolerance.
        payload = torch.randint(
            -8,
            9,
            (total, hidden),
            dtype=torch.int32,
            generator=g,
            device=device,
        )
        input_emb = payload.to(dtype)
    else:
        input_emb = torch.randn(
            total, hidden, generator=g, device=device, dtype=torch.float32
        ).to(dtype)

    return dict(
        input_emb=input_emb,
        offsets=offsets,
        lengths=lengths_t,
        max_len=max_len,
        check=_check,
    )


def _check(actual, expected):
    a_bs, a_pad, a_mask = actual
    e_bs, e_pad, e_mask = expected
    assert a_bs == e_bs, f"batch_size {a_bs} != {e_bs}"
    # Gather-and-copy, no arithmetic: the padded rows must match byte for byte,
    # and the pad band must be exactly zero rather than merely close.
    assert torch.equal(a_pad, e_pad), "padded mismatch"
    assert torch.equal(a_mask, e_mask), "valid_mask mismatch"


_CASE_FACTORIES = [
    # The geometry from aiter_backend: 4096 = 32 heads x 128, dp=2048.
    lambda: _case(8, 2048, 4096, [2048] * 8, seed=0),
    lambda: _case(8, 2048, 4096, [1, 2, 3, 0, 2048, 17, 700, 1500], seed=1),
    # A short-sequence batch with one long row: pad band dominates.
    lambda: _case(4, 512, 1024, [3, 0, 512, 9], seed=2),
    # hidden not a multiple of the 128 hidden tile (100 -> one partial tile).
    lambda: _case(3, 70, 100, [5, 12, 30], seed=3),
    # max_len not a multiple of the 64 row tile; lengths exactly max_len.
    lambda: _case(2, 65, 3584, [65, 65], seed=4),
    # A sequence longer than the slot: must truncate, not overflow.
    lambda: _case(3, 16, 128, [40, 7, 16], seed=5),
    # All-empty batch: every row is pad band, mask all false.
    lambda: _case(5, 33, 256, [0, 0, 0, 0, 0], seed=6),
    # Single row, single token.
    lambda: _case(1, 1, 128, [1], seed=7),
]

_DTYPE_FACTORIES = [
    lambda: _case(4, 128, 512, [3, 17, 0, 64], seed=8, dtype=torch.bfloat16),
    lambda: _case(4, 128, 512, [3, 17, 0, 64], seed=9, dtype=torch.float16),
    lambda: _case(
        4, 128, 512, [3, 17, 0, 64], seed=10, dtype=torch.float8_e4m3fn
    ),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES) + len(_DTYPE_FACTORIES)))


def _build(idx):
    if idx < len(_CASE_FACTORIES):
        return _CASE_FACTORIES[idx]()
    return _DTYPE_FACTORIES[idx - len(_CASE_FACTORIES)]()


@pytest.mark.parametrize("case_idx", CORRECTNESS_CASES)
@pytest.mark.pad_sequence_with_mask
def test_pad_sequence_with_mask(case_idx):
    case = _build(case_idx)
    reference = get_reference(OP)
    check = case.pop("check", None)
    kwargs = case

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/pad_sequence_with_mask ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/pad_sequence_with_mask not yet implemented")
        return

    check(actual, expected)


@pytest.mark.pad_sequence_with_mask
def test_pad_sequence_with_mask_empty_dims():
    """A zero-length grid dimension must not launch and must not crash."""
    device = flaggems_sglang.device
    op = flaggems_sglang.pad_sequence_with_mask
    reference = get_reference(OP)

    cases = [
        (0, 8, 64),
        (3, 0, 64),
        (3, 8, 0),
    ]
    for batch, max_len, hidden in cases:
        offsets = torch.zeros(batch, dtype=torch.int32, device=device)
        lengths = torch.zeros(batch, dtype=torch.int32, device=device)
        input_emb = torch.zeros(0, hidden, dtype=torch.float32, device=device)

        actual = op(input_emb, offsets, lengths, max_len)
        expected = reference(input_emb, offsets, lengths, max_len)
        _check(actual, expected)


@pytest.mark.pad_sequence_with_mask
def test_pad_sequence_with_mask_int64_indices():
    device = flaggems_sglang.device
    op = flaggems_sglang.pad_sequence_with_mask
    reference = get_reference(OP)

    offsets = torch.tensor([0, 5, 5], dtype=torch.int64, device=device)
    lengths = torch.tensor([5, 0, 9], dtype=torch.int64, device=device)
    input_emb = torch.arange(
        14 * 256, dtype=torch.float32, device=device
    ).reshape(14, 256)

    actual = op(input_emb, offsets, lengths, 32)
    expected = reference(input_emb, offsets, lengths, 32)
    _check(actual, expected)


@pytest.mark.pad_sequence_with_mask
@pytest.mark.parametrize(
    "mutation",
    [
        "bad_ndim",
        "mismatched_lengths",
        "bad_offsets_dtype",
        "negative_max_len",
    ],
)
def test_pad_sequence_with_mask_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.pad_sequence_with_mask

    input_emb = torch.zeros(4, 8, dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
    lengths = torch.tensor([2, 2], dtype=torch.int32, device=device)
    max_len = 4

    if mutation == "bad_ndim":
        with pytest.raises(ValueError):
            op(input_emb[None], offsets, lengths, max_len)
    elif mutation == "mismatched_lengths":
        with pytest.raises(ValueError):
            op(
                input_emb,
                offsets,
                torch.tensor([2], dtype=torch.int32),
                max_len,
            )
    elif mutation == "bad_offsets_dtype":
        with pytest.raises(ValueError):
            op(input_emb, offsets.to(torch.float32), lengths, max_len)
    else:
        with pytest.raises(ValueError):
            op(input_emb, offsets, lengths, -1)
