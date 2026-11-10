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

"""Correctness test for kv_cache/quantize_k_cache_fast."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "quantize_k_cache_fast"

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


def _parts(packed, dim_nope, dim_rope, group_size):
    """Split a packed row into its three typed segments."""
    rows = packed.reshape(-1, packed.shape[-1])
    num_tiles = dim_nope // group_size
    nope_q = rows[:, :dim_nope].float()
    nope_s = rows[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope = rows[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    return nope_q, nope_s, rope


def _dequant(packed, dim_nope, dim_rope, group_size):
    """Undo the quantization, so the result can be judged against the input."""
    nope_q, nope_s, rope = _parts(packed, dim_nope, dim_rope, group_size)
    scales = nope_s.repeat_interleave(group_size, dim=-1)
    nope = nope_q * scales
    out = torch.cat([nope, rope.float()], dim=-1)
    return out.view(*packed.shape[:-1], dim_nope + dim_rope)


def _check(actual, expected, k_nope, k_rope, dim_nope, dim_rope, group_size):
    assert (
        actual.shape == expected.shape
    ), f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype
    ), f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.dtype == torch.float8_e4m3fn

    # The scales come from the same amax/448 expression, so they must match bit
    # for bit; any difference there is a real bug, not rounding.
    a_q, a_s, a_r = _parts(actual, dim_nope, dim_rope, group_size)
    _, e_s, e_r = _parts(expected, dim_nope, dim_rope, group_size)
    torch.testing.assert_close(a_s, e_s, rtol=0, atol=0)

    # The RoPE segment is copied, never arithmetically touched.
    assert torch.equal(a_r, e_r), "rope bytes differ from the reference copy"

    # The quantized NoPE segment has to survive a round trip back to bf16.
    recovered = _dequant(actual, dim_nope, dim_rope, group_size)
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    ref_vals = torch.cat(
        [k_nope_2d.float(), k_rope_2d.float()], dim=-1
    ).reshape(recovered.shape)
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
@pytest.mark.quantize_k_cache_fast
def test_quantize_k_cache_fast(case_idx):
    built = _CASES[case_idx]()
    kw = built["case"]
    reference = get_reference(OP)

    expected = reference(**kw)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kv_cache/quantize_k_cache_fast ops module not found")
        return

    try:
        actual = candidate(**kw)
    except NotImplementedError:
        pytest.skip("kv_cache/quantize_k_cache_fast not yet implemented")
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


@pytest.mark.quantize_k_cache_fast
def test_quantize_k_cache_fast_scale_is_amax_over_448():
    """The stored scale is exactly the tile amax divided by 448."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast

    g = torch.Generator(device=device).manual_seed(20)
    k_nope = _randn((6, 512), device, g)
    # Flatten the whole first tile, then plant one known maximum, so 3.5 is
    # guaranteed to be the tile amax rather than a random draw beating it.
    first_tile = k_nope[:, :128].float()
    first_tile.clamp_(-1.0, 1.0)
    first_tile[0, 3] = 3.5
    k_nope[:, :128] = first_tile.to(torch.bfloat16)
    k_rope = _randn((6, 64), device, g)

    packed = op(k_nope, k_rope)
    _, scales, _ = _parts(packed, 512, 64, 128)

    expected_first = k_nope[:, :128].float().abs().amax(dim=-1) / _FP8_MAX
    torch.testing.assert_close(scales[:, 0], expected_first, rtol=0, atol=0)
    assert scales[0, 0].item() == 3.5 / 448.0


@pytest.mark.quantize_k_cache_fast
def test_quantize_k_cache_fast_rope_is_byte_exact():
    """The rotary columns are copied as bytes, not re-derived from fp8."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast

    g = torch.Generator(device=device).manual_seed(21)
    k_nope = _randn((4, 512), device, g)
    k_rope = _randn((4, 64), device, g)

    packed = op(k_nope, k_rope)
    _, _, rope = _parts(packed, 512, 64, 128)
    assert rope.dtype == torch.bfloat16
    assert torch.equal(rope, k_rope), "rotary bytes were altered"


@pytest.mark.quantize_k_cache_fast
def test_quantize_k_cache_fast_matches_dequantize_layout():
    """The row width and segment offsets are the reader's, not a new layout."""
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast

    dim_nope, dim_rope, group_size = 512, 64, 128
    num_tiles = dim_nope // group_size
    expected_width = dim_nope + num_tiles * 4 + dim_rope * 2

    g = torch.Generator(device=device).manual_seed(22)
    k_nope = _randn((5, dim_nope), device, g)
    k_rope = _randn((5, dim_rope), device, g)

    packed = op(k_nope, k_rope)
    assert packed.shape == (
        5,
        expected_width,
    ), f"row width {packed.shape[-1]} != {expected_width}"
    assert packed.dtype == torch.float8_e4m3fn
    # 512 + 4*4 + 128 = 656 bytes, the width the reader asserts on.
    assert expected_width == 656


@pytest.mark.quantize_k_cache_fast
def test_quantize_k_cache_fast_empty_tokens():
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast

    k_nope = torch.empty(0, 512, dtype=torch.bfloat16, device=device)
    k_rope = torch.empty(0, 64, dtype=torch.bfloat16, device=device)
    packed = op(k_nope, k_rope)
    assert packed.shape == (0, 656)


@pytest.mark.quantize_k_cache_fast
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
def test_quantize_k_cache_fast_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.quantize_k_cache_fast

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
