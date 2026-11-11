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

"""Correctness test for kv_cache/quantize_k_cache_fast_separate."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "quantize_k_cache_fast_separate"

# The kernel forms the quotient as x * (1/scale) while the reference divides,
# so individual fp8 codes can land on different rounding sides. What must hold
# exactly is the scale and the copied rotary bytes; the quantized values are
# judged through the round trip at the tolerance upstream's own manual check
# (`check_dsa_quant_k_cache.py`) uses for this quantizer: atol=0.2, rtol=0.2.
_FP8_MAX = 448.0
_RTOL = 0.2
_ATOL = 0.2


def _randn(shape, device, generator):
    return torch.randn(
        shape, dtype=torch.float32, generator=generator, device=device
    ).to(torch.bfloat16)


def _split(nope_part, rope_part, dim_nope, dim_rope, group_size):
    """Split the two byte buffers into typed segments."""
    nope_rows = nope_part.reshape(-1, nope_part.shape[-1])
    nope_q = nope_rows[:, :dim_nope].view(torch.float8_e4m3fn).float()
    nope_s = nope_rows[:, dim_nope:].view(torch.float32)
    if dim_rope > 0:
        rope = rope_part.reshape(-1, rope_part.shape[-1]).view(torch.bfloat16)
    else:
        rope = torch.empty(
            (nope_rows.shape[0], 0),
            dtype=torch.bfloat16,
            device=nope_part.device,
        )
    return nope_q, nope_s, rope


def _dequant(nope_part, rope_part, dim_nope, dim_rope, group_size):
    """Undo the quantization, so the result can be judged against the input."""
    nope_q, nope_s, rope = _split(
        nope_part, rope_part, dim_nope, dim_rope, group_size
    )
    scales = nope_s.repeat_interleave(group_size, dim=-1)
    nope = nope_q * scales
    return torch.cat([nope, rope.float()], dim=-1)


def _check(actual, expected, k_nope, k_rope, dim_nope, dim_rope, group_size):
    a_nope, a_rope = actual
    e_nope, e_rope = expected

    assert (
        a_nope.shape == e_nope.shape
    ), f"nonrotary shape {tuple(a_nope.shape)} != {tuple(e_nope.shape)}"
    assert (
        a_rope.shape == e_rope.shape
    ), f"rotary shape {tuple(a_rope.shape)} != {tuple(e_rope.shape)}"
    assert a_nope.dtype == torch.uint8
    assert a_rope.dtype == torch.uint8

    # Both halves carry the singleton head axis whatever the input rank was.
    assert a_nope.ndim == 3 and a_rope.ndim == 3
    assert a_nope.shape[-2] == 1 and a_rope.shape[-2] == 1

    num_tiles = dim_nope // group_size
    assert a_nope.shape[-1] == dim_nope + num_tiles * 4
    assert a_rope.shape[-1] == dim_rope * 2

    # The scales come from the same amax/448 expression, so they must match bit
    # for bit; any difference there is a real bug, not rounding.
    a_q, a_s, a_r = _split(a_nope, a_rope, dim_nope, dim_rope, group_size)
    _, e_s, e_r = _split(e_nope, e_rope, dim_nope, dim_rope, group_size)
    torch.testing.assert_close(a_s, e_s, rtol=0, atol=0)

    # The RoPE segment is copied, never arithmetically touched.
    assert torch.equal(a_r, e_r), "rope bytes differ from the reference copy"

    # The quantized NoPE segment has to survive a round trip back to bf16.
    recovered = _dequant(a_nope, a_rope, dim_nope, dim_rope, group_size)
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    ref_vals = torch.cat([k_nope_2d.float(), k_rope_2d.float()], dim=-1)
    torch.testing.assert_close(recovered, ref_vals, rtol=_RTOL, atol=_ATOL)


def _make(tokens, dim_nope, dim_rope, group_size=128, seed=0, ndim=2):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)

    k_nope = _randn((tokens, dim_nope), device, g)
    k_rope = _randn((tokens, dim_rope), device, g)
    if ndim == 3:
        k_nope = k_nope.unsqueeze(1)
        k_rope = k_rope.unsqueeze(1)

    return dict(
        case=dict(k_nope=k_nope, k_rope=k_rope, group_size=group_size),
        dim_nope=dim_nope,
        dim_rope=dim_rope,
        group_size=group_size,
    )


_CASES = [
    # The DSA geometry the operator ships for.
    lambda: _make(1, 512, 64, seed=0),
    lambda: _make(37, 512, 64, seed=1),
    lambda: _make(512, 512, 64, seed=2),
    lambda: _make(2048, 512, 64, seed=3),
    lambda: _make(129, 512, 64, seed=4),
    # Other NoPE widths the layout allows.
    lambda: _make(8, 128, 64, seed=5),
    lambda: _make(8, 256, 0, seed=6),
    lambda: _make(8, 512, 0, seed=7),
    lambda: _make(8, 1024, 64, seed=8),
    lambda: _make(8, 1024, 128, seed=9),
    # A coarser and a finer tile size, and the singleton-head 3-D input.
    lambda: _make(16, 512, 64, group_size=64, seed=10),
    lambda: _make(16, 512, 64, group_size=256, seed=11),
    lambda: _make(8, 512, 64, seed=12, ndim=3),
]

CASE_IDS = list(range(len(_CASES)))


@pytest.mark.parametrize("case_idx", CASE_IDS)
@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate(case_idx):
    built = _CASES[case_idx]()
    kw = built["case"]
    reference = get_reference(OP)

    expected = reference(**kw)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "kv_cache/quantize_k_cache_fast_separate ops module not found"
        )
        return

    try:
        actual = candidate(**kw)
    except NotImplementedError:
        pytest.skip(
            "kv_cache/quantize_k_cache_fast_separate not yet implemented"
        )
        return

    _check(
        actual,
        expected,
        kw["k_nope"],
        kw["k_rope"],
        built["dim_nope"],
        built["dim_rope"],
        built["group_size"],
    )


@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate_scale_is_amax_over_448():
    """The stored scale is exactly the tile amax divided by 448."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    g = torch.Generator(device=device).manual_seed(20)
    k_nope = _randn((6, 512), device, g)
    # Flatten the whole first tile, then plant one known maximum, so 3.5 is
    # guaranteed to be the tile amax rather than a random draw beating it.
    first_tile = k_nope[:, :128].float()
    first_tile.clamp_(-1.0, 1.0)
    first_tile[0, 3] = 3.5
    k_nope[:, :128] = first_tile.to(torch.bfloat16)
    k_rope = _randn((6, 64), device, g)

    nope_part, rope_part = op(k_nope, k_rope)
    _, scales, _ = _split(nope_part, rope_part, 512, 64, 128)

    expected_first = k_nope[:, :128].float().abs().amax(dim=-1) / _FP8_MAX
    torch.testing.assert_close(scales[:, 0], expected_first, rtol=0, atol=0)
    assert scales[0, 0].item() == 3.5 / 448.0


@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate_rope_is_byte_exact():
    """The rotary columns are copied as bytes, not re-derived from fp8."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    g = torch.Generator(device=device).manual_seed(21)
    k_nope = _randn((4, 512), device, g)
    k_rope = _randn((4, 64), device, g)

    nope_part, rope_part = op(k_nope, k_rope)
    rope = rope_part.view(torch.bfloat16).reshape(4, 64)
    assert rope.dtype == torch.bfloat16
    assert torch.equal(rope, k_rope), "rotary bytes were altered"


@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate_byte_widths_match_the_packed_row():
    """The two halves add up to the width the packed variant would emit."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    dim_nope, dim_rope, group_size = 512, 64, 128
    num_tiles = dim_nope // group_size
    expected_nope_bytes = dim_nope + num_tiles * 4
    expected_rope_bytes = dim_rope * 2

    g = torch.Generator(device=device).manual_seed(22)
    k_nope = _randn((5, dim_nope), device, g)
    k_rope = _randn((5, dim_rope), device, g)

    nope_part, rope_part = op(k_nope, k_rope)
    assert nope_part.shape == (5, 1, expected_nope_bytes)
    assert rope_part.shape == (5, 1, expected_rope_bytes)
    assert nope_part.stride(0) == expected_nope_bytes
    assert rope_part.stride(0) == expected_rope_bytes
    # 528 + 128 = 656 bytes, the width of the row the reader asserts on.
    assert expected_nope_bytes == 528
    assert expected_nope_bytes + expected_rope_bytes == 656


@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate_empty_tokens():
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    k_nope = torch.empty(0, 512, dtype=torch.bfloat16, device=device)
    k_rope = torch.empty(0, 64, dtype=torch.bfloat16, device=device)
    nope_part, rope_part = op(k_nope, k_rope)
    assert nope_part.shape == (0, 1, 528)
    assert rope_part.shape == (0, 1, 128)


@pytest.mark.quantize_k_cache_fast_separate
def test_quantize_k_cache_fast_separate_always_three_dimensional():
    """A 2-D input still comes back with the head axis the writer expects."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    g = torch.Generator(device=device).manual_seed(23)
    k_nope_2d = _randn((6, 512), device, g)
    k_rope_2d = _randn((6, 64), device, g)

    flat = op(k_nope_2d, k_rope_2d)
    with_head = op(k_nope_2d.unsqueeze(1), k_rope_2d.unsqueeze(1))

    for got, want in zip(flat, with_head):
        assert got.shape == want.shape
        assert torch.equal(got, want)


@pytest.mark.quantize_k_cache_fast_separate
@pytest.mark.parametrize(
    "mutation",
    [
        "k_nope_ndim",
        "k_rope_ndim",
        "dtype",
        "token_mismatch",
        "dim_not_multiple",
        "group_not_power_of_two",
        "group_not_int",
        "head_axis_not_one",
        "non_unit_stride",
    ],
)
def test_quantize_k_cache_fast_separate_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast_separate

    k_nope = torch.zeros(4, 512, dtype=torch.bfloat16, device=device)
    k_rope = torch.zeros(4, 64, dtype=torch.bfloat16, device=device)

    if mutation == "k_nope_ndim":
        with pytest.raises(ValueError):
            op(k_nope[0], k_rope)
    elif mutation == "k_rope_ndim":
        with pytest.raises(ValueError):
            op(k_nope, k_rope[0, 0])
    elif mutation == "dtype":
        with pytest.raises(ValueError):
            op(k_nope, k_rope.to(torch.float16))
    elif mutation == "token_mismatch":
        with pytest.raises(ValueError):
            op(k_nope, torch.zeros(5, 64, dtype=torch.bfloat16, device=device))
    elif mutation == "dim_not_multiple":
        with pytest.raises(ValueError):
            op(
                torch.zeros(4, 500, dtype=torch.bfloat16, device=device),
                k_rope,
            )
    elif mutation == "group_not_power_of_two":
        with pytest.raises(ValueError):
            op(k_nope, k_rope, group_size=100)
    elif mutation == "group_not_int":
        with pytest.raises(ValueError):
            op(k_nope, k_rope, group_size=128.0)
    elif mutation == "head_axis_not_one":
        with pytest.raises(ValueError):
            op(k_nope.view(4, 1, 512).expand(4, 2, 512), k_rope)
    else:
        # A stepped inner slice is the only inner-dim view that breaks the
        # unit-stride requirement; a plain prefix slice keeps stride 1.
        wide = torch.zeros(4, 1024, dtype=torch.bfloat16, device=device)
        stepped = wide[..., ::2]
        assert stepped.stride(1) == 2
        with pytest.raises(ValueError):
            op(stepped, k_rope)
