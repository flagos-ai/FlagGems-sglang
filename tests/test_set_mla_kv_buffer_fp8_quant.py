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

"""Correctness test for set_mla_kv_buffer_fp8_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "set_mla_kv_buffer_fp8_quant"

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2

# e4m3 has no infinity and saturates at 448, which is what the implicit store
# cast in the kernel produces. e5m2 does have an infinity, so the two casts
# only agree for in-range inputs and that is all this file asserts.
_E4M3_MAX = 448.0


def _build(shape, device, seed=0, fp8_dtype=FP8_E4M3):
    """Return (kv_buffer, loc, cache_k_nope, cache_k_rope) for a shape."""
    num_tokens, nope_dim, rope_dim, rows = shape
    assert (
        num_tokens <= rows
    ), "duplicate target slots are racy, keep the slots unique"
    g = torch.Generator(device=device).manual_seed(seed)

    kv_buffer = torch.randint(
        0,
        256,
        (rows, nope_dim + rope_dim),
        generator=g,
        device=device,
        dtype=torch.uint8,
    )
    nope = torch.randn(
        (num_tokens, nope_dim),
        generator=g,
        device=device,
        dtype=torch.bfloat16,
    )
    rope = torch.randn(
        (num_tokens, rope_dim),
        generator=g,
        device=device,
        dtype=torch.bfloat16,
    )
    loc = torch.arange(num_tokens, device=device, dtype=torch.int32) % rows
    return kv_buffer, loc, nope, rope


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # The payload is eight-bit, so the comparison is on the encoded bytes and
    # needs no tolerance at all.
    torch.testing.assert_close(
        actual.reshape(-1).view(torch.uint8),
        expected.reshape(-1).view(torch.uint8),
        rtol=0.0,
        atol=0.0,
    )


def _run(kv_buffer, loc, nope, rope, fp8_dtype=FP8_E4M3):
    reference = get_reference(OP)
    expected = reference(kv_buffer, loc, nope, rope, fp8_dtype)
    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(f"{OP} ops module not found")
        return None, expected
    try:
        actual = candidate(kv_buffer, loc, nope, rope, fp8_dtype)
    except NotImplementedError:
        pytest.skip(f"{OP} not yet implemented")
        return None, expected
    return actual, expected


# (num_tokens, nope_dim, rope_dim, rows)
_CASES = [
    # The 512 + 64 split is what DeepSeek-style MLA caches use, and 512 is an
    # exact multiple of the block width so no tile straddles the seam.
    lambda: (8, 512, 64, 8),
    lambda: (64, 512, 64, 64),
    # A single token is the decode step, where the grid has almost no work.
    lambda: (1, 512, 64, 1),
    # 96 is not a multiple of the block width, so one tile straddles NoPE/RoPE.
    lambda: (8, 96, 32, 8),
    lambda: (16, 128, 64, 16),
    # Odd, tiny dimensions leave most of the tile masked out.
    lambda: (3, 17, 5, 4),
]

CASE_IDS = [f"{c()[0]}t-{c()[1]}n-{c()[2]}r" for c in _CASES]


@pytest.mark.parametrize("shape", [c() for c in _CASES], ids=CASE_IDS)
@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_set_mla_kv_buffer_fp8_quant(shape):
    device = flaggems_sglang.device
    kv_buffer, loc, nope, rope = _build(shape, device)
    actual, expected = _run(kv_buffer, loc, nope, rope)
    if actual is None:
        return
    _check(actual, expected)


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_inputs_are_not_mutated():
    """The contract declares mutates: [], so the caller's buffer must survive."""
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    kv_buffer, loc, nope, rope = _build((8, 512, 64, 8), device)
    before = kv_buffer.clone()

    out = op(kv_buffer, loc, nope, rope, FP8_E4M3)

    assert torch.equal(kv_buffer, before), "kv_buffer was mutated"
    assert out.data_ptr() != kv_buffer.data_ptr()
    assert out.dtype == kv_buffer.dtype


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_rows_land_at_the_requested_slots():
    """Slots are arbitrary, not sequential, and must be honoured exactly."""
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    rows, nope_dim, rope_dim = 16, 64, 32
    kv_buffer = torch.zeros(
        (rows, nope_dim + rope_dim), device=device, dtype=torch.uint8
    )
    count = 4
    loc = torch.tensor([11, 3, 15, 0], device=device, dtype=torch.int64)
    nope = torch.ones((count, nope_dim), device=device, dtype=torch.bfloat16)
    rope = torch.ones((count, rope_dim), device=device, dtype=torch.bfloat16)

    out = op(kv_buffer, loc, nope, rope, FP8_E4M3)
    buffer = out.view(FP8_E4M3).float()

    # Slot 0 is written, unlike the upstream kernel's default that treats 0 as
    # a reserved skip index and leaves it alone.
    for slot in loc.tolist():
        assert torch.equal(
            buffer[slot], torch.ones(nope_dim + rope_dim, device=device)
        ), f"slot {slot} was not written"
    touched = torch.nonzero(buffer.abs().sum(dim=1), as_tuple=False).flatten()
    assert sorted(touched.tolist()) == sorted(loc.tolist())


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_out_of_range_slots_are_skipped():
    """Negative and too-large slots leave the copied row untouched."""
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    kv_buffer, _, nope, rope = _build((8, 64, 32, 8), device)
    before = kv_buffer.clone()
    loc = torch.tensor([-1, -7, 8, 99], device=device, dtype=torch.int64)
    tail_nope = nope[:4].clone()
    tail_rope = rope[:4].clone()

    out = op(kv_buffer, loc, tail_nope, tail_rope, FP8_E4M3)
    assert torch.equal(out, before)


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_overflow_saturates_in_e4m3():
    """e4m3 has no infinity, so the store cast must clamp at 448."""
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    nope_dim, rope_dim = 64, 32
    kv_buffer = torch.zeros(
        (1, nope_dim + rope_dim), device=device, dtype=torch.uint8
    )
    nope = torch.full(
        (1, nope_dim), 1000.0, device=device, dtype=torch.bfloat16
    )
    rope = torch.full(
        (1, rope_dim), -1000.0, device=device, dtype=torch.bfloat16
    )
    loc = torch.zeros(1, device=device, dtype=torch.int32)

    out = op(kv_buffer, loc, nope, rope, FP8_E4M3).view(FP8_E4M3).float()
    assert out[0, :nope_dim].max() == _E4M3_MAX
    assert out[0, nope_dim:].min() == -_E4M3_MAX


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_e5m2_in_range_is_exact():
    """e5m2 differs from the torch cast only past its finite range."""
    device = flaggems_sglang.device
    kv_buffer, loc, nope, rope = _build((8, 64, 32, 8), device)
    nope = nope / 4.0
    rope = rope / 4.0
    actual, expected = _run(kv_buffer, loc, nope, rope, FP8_E5M2)
    if actual is None:
        return
    _check(actual, expected)


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_accepts_an_already_fp8_buffer():
    """A buffer that is already FP8 needs no reinterpretation."""
    device = flaggems_sglang.device
    kv_buffer, loc, nope, rope = _build(
        (8, 64, 32, 8), device, fp8_dtype=FP8_E4M3
    )
    kv_buffer = kv_buffer.view(FP8_E4M3)
    actual, expected = _run(kv_buffer, loc, nope, rope)
    if actual is None:
        return
    _check(actual, expected)


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_wider_buffer_keeps_its_tail():
    """Rows may be wider than the concatenated key; the extra columns stay."""
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    nope_dim, rope_dim, extra = 64, 32, 16
    width = nope_dim + rope_dim + extra
    kv_buffer = torch.zeros((4, width), device=device, dtype=torch.uint8)
    kv_buffer[:, nope_dim + rope_dim :] = 0x7F
    loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
    nope = torch.ones((2, nope_dim), device=device, dtype=torch.bfloat16)
    rope = torch.ones((2, rope_dim), device=device, dtype=torch.bfloat16)

    out = op(kv_buffer, loc, nope, rope, FP8_E4M3)
    assert torch.equal(
        out[:, nope_dim + rope_dim :], kv_buffer[:, nope_dim + rope_dim :]
    )
    # And the rows no slot points at keep a zero key, tail excluded.
    assert out[2:, : nope_dim + rope_dim].eq(0).all()


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_empty_batch_returns_an_unchanged_copy():
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant

    kv_buffer, _, nope, rope = _build((8, 64, 32, 8), device)
    loc = torch.empty(0, device=device, dtype=torch.int32)
    out = op(kv_buffer, loc, nope[:0], rope[:0], FP8_E4M3)
    assert torch.equal(out, kv_buffer)
    assert out.data_ptr() != kv_buffer.data_ptr()


@pytest.mark.set_mla_kv_buffer_fp8_quant
def test_rejects_bad_arguments():
    device = flaggems_sglang.device
    op = flaggems_sglang.set_mla_kv_buffer_fp8_quant
    kv_buffer, loc, nope, rope = _build((8, 64, 32, 8), device)

    with pytest.raises(TypeError):
        op(kv_buffer, loc, nope, rope, torch.bfloat16)

    with pytest.raises(ValueError):
        op(kv_buffer.unsqueeze(0), loc, nope, rope, FP8_E4M3)

    with pytest.raises(TypeError):
        op(kv_buffer.to(torch.bfloat16), loc, nope, rope, FP8_E4M3)

    with pytest.raises(TypeError):
        op(kv_buffer, loc.to(torch.float32), nope, rope, FP8_E4M3)

    with pytest.raises(ValueError):
        op(kv_buffer, loc[:4], nope, rope, FP8_E4M3)

    with pytest.raises(ValueError):
        op(kv_buffer.transpose(0, 1), loc, nope, rope, FP8_E4M3)

    with pytest.raises(ValueError):
        op(kv_buffer, loc, nope.t(), rope, FP8_E4M3)

    narrow = torch.zeros((8, 32), device=device, dtype=torch.uint8)
    with pytest.raises(ValueError):
        op(narrow, loc, nope, rope, FP8_E4M3)
