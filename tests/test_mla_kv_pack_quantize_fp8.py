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

"""Correctness test for attention/mla_kv_pack_quantize_fp8."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "mla_kv_pack_quantize_fp8"

# fp8 carries a wide value range on a coarse mantissa, so the comparison runs
# on the widened values at the upstream tolerance. The default per-dtype table
# is keyed on input dtype and has no fp8 entry, hence the explicit constants.
RTOL = 1e-2
ATOL = 0.5


def _randn(shape, dtype, device, generator):
    return torch.randn(
        shape, dtype=torch.float32, generator=generator, device=device
    ).to(dtype)


def _case(
    tokens,
    heads,
    qk_nope,
    qk_rope,
    v_head,
    seed=0,
    dtype=torch.bfloat16,
    k_scale_inv=0.7,
    v_scale_inv=1.3,
    fp8_dtype=torch.float8_e4m3fn,
):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)
    return dict(
        k_nope=_randn((tokens, heads, qk_nope), dtype, device, g),
        k_pe=_randn((tokens, 1, qk_rope), dtype, device, g),
        v=_randn((tokens, heads, v_head), dtype, device, g),
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        fp8_dtype=fp8_dtype,
        check=_check,
    )


def _check(actual, expected):
    a_key, a_val = actual
    e_key, e_val = expected
    assert (
        a_key.shape == e_key.shape
    ), f"key shape {tuple(a_key.shape)} != {tuple(e_key.shape)}"
    assert (
        a_val.shape == e_val.shape
    ), f"value shape {tuple(a_val.shape)} != {tuple(e_val.shape)}"
    assert (
        a_key.dtype == e_key.dtype
    ), f"key dtype {a_key.dtype} != {e_key.dtype}"
    assert (
        a_val.dtype == e_val.dtype
    ), f"value dtype {a_val.dtype} != {e_val.dtype}"
    torch.testing.assert_close(
        a_key.float(), e_key.float(), rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        a_val.float(), e_val.float(), rtol=RTOL, atol=ATOL
    )


_CASE_FACTORIES = [
    # DeepSeek-V3 geometry: 128 non-rotary + 64 rotary, 128-wide values.
    lambda: _case(1024, 64, 128, 64, 128, seed=0),
    lambda: _case(64, 32, 128, 64, 128, seed=1),
    # The narrow geometry from the upstream parameterisation.
    lambda: _case(257, 16, 64, 32, 64, seed=2),
    # Every launch regime in _pick_launch, at odd token counts.
    lambda: _case(1, 8, 128, 64, 128, seed=3),
    lambda: _case(3, 8, 128, 64, 128, seed=4),
    lambda: _case(17, 8, 128, 64, 128, seed=5),
    lambda: _case(200, 8, 128, 64, 128, seed=6),
    lambda: _case(1600, 8, 128, 64, 128, seed=7),
    # Widths that are not powers of two, so the in-kernel masks carry the tail.
    lambda: _case(64, 8, 192, 96, 160, seed=8),
    # A different source dtype, and the other fp8 flavour.
    lambda: _case(128, 16, 128, 64, 128, seed=9, dtype=torch.float16),
    lambda: _case(128, 16, 128, 64, 128, seed=10, fp8_dtype=torch.float8_e5m2),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


def _build(idx):
    return _CASE_FACTORIES[idx]()


@pytest.mark.parametrize("case_idx", CORRECTNESS_CASES)
@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8(case_idx):
    case = _build(case_idx)
    reference = get_reference(OP)
    check = case.pop("check")
    kwargs = case

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/mla_kv_pack_quantize_fp8 ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/mla_kv_pack_quantize_fp8 not yet implemented")
        return

    check(actual, expected)


@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8_strided_k_nope():
    """A non-contiguous outer layout is accepted as long as the inner dim is."""
    device = flaggems_sglang.device
    op = flaggems_sglang.mla_kv_pack_quantize_fp8
    reference = get_reference(OP)

    tokens, heads = 16, 32
    qk_nope, qk_rope, v_head = 128, 64, 128
    g = torch.Generator(device=device).manual_seed(11)

    full = torch.randn(
        (tokens, heads, qk_nope * 2), generator=g, device=device
    ).to(torch.bfloat16)
    k_nope = full[..., qk_nope:]
    assert k_nope.stride(-1) == 1

    k_pe = _randn((tokens, 1, qk_rope), torch.bfloat16, device, g)
    v = _randn((tokens, heads, v_head), torch.bfloat16, device, g)

    actual = op(k_nope, k_pe, v)
    expected = reference(k_nope, k_pe, v)
    _check(actual, expected)


@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8_kpe_2d():
    """A 2D rotary key is the same object as its singleton-head 3D form."""
    device = flaggems_sglang.device
    op = flaggems_sglang.mla_kv_pack_quantize_fp8
    reference = get_reference(OP)

    tokens, heads = 8, 16
    qk_nope, qk_rope, v_head = 128, 64, 128
    g = torch.Generator(device=device).manual_seed(12)

    k_nope = _randn((tokens, heads, qk_nope), torch.bfloat16, device, g)
    k_pe_2d = _randn((tokens, qk_rope), torch.bfloat16, device, g)
    v = _randn((tokens, heads, v_head), torch.bfloat16, device, g)

    actual = op(k_nope, k_pe_2d, v)
    expected = reference(k_nope, k_pe_2d, v)
    _check(actual, expected)


@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8_default_scales_and_no_mutation():
    """The scales default to 1.0 and no input is written in place."""
    device = flaggems_sglang.device
    op = flaggems_sglang.mla_kv_pack_quantize_fp8
    reference = get_reference(OP)

    tokens, heads = 32, 16
    qk_nope, qk_rope, v_head = 128, 64, 128
    g = torch.Generator(device=device).manual_seed(13)

    k_nope = _randn((tokens, heads, qk_nope), torch.bfloat16, device, g)
    k_pe = _randn((tokens, 1, qk_rope), torch.bfloat16, device, g)
    v = _randn((tokens, heads, v_head), torch.bfloat16, device, g)

    before = (k_nope.clone(), k_pe.clone(), v.clone())
    actual = op(k_nope, k_pe, v)
    expected = reference(k_nope, k_pe, v)
    _check(actual, expected)

    for original, current in zip(before, (k_nope, k_pe, v), strict=True):
        assert torch.equal(original, current), "an input was mutated"
    # The outputs must be separate allocations, not views over the inputs.
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[0].data_ptr() != k_nope.data_ptr()
    assert actual[1].data_ptr() != v.data_ptr()


@pytest.mark.mla_kv_pack_quantize_fp8
def test_mla_kv_pack_quantize_fp8_empty_tokens():
    """Zero tokens must return shape-correct empty outputs, not launch."""
    device = flaggems_sglang.device
    op = flaggems_sglang.mla_kv_pack_quantize_fp8
    reference = get_reference(OP)

    qk_nope, qk_rope, v_head = 128, 64, 128
    k_nope = torch.empty(0, 8, qk_nope, dtype=torch.bfloat16, device=device)
    k_pe = torch.empty(0, 1, qk_rope, dtype=torch.bfloat16, device=device)
    v = torch.empty(0, 8, v_head, dtype=torch.bfloat16, device=device)

    actual = op(k_nope, k_pe, v)
    expected = reference(k_nope, k_pe, v)
    _check(actual, expected)


@pytest.mark.mla_kv_pack_quantize_fp8
@pytest.mark.parametrize(
    "mutation",
    [
        "k_nope_ndim",
        "v_ndim",
        "kpe_ndim",
        "dtype_mismatch",
        "bad_fp8_dtype",
        "token_mismatch",
        "head_mismatch",
        "kpe_head_not_one",
        "non_unit_stride",
    ],
)
def test_mla_kv_pack_quantize_fp8_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.mla_kv_pack_quantize_fp8

    tokens, heads = 8, 4
    qk_nope, qk_rope, v_head = 128, 64, 128
    dtype = torch.bfloat16

    k_nope = torch.zeros(tokens, heads, qk_nope, dtype=dtype, device=device)
    k_pe = torch.zeros(tokens, 1, qk_rope, dtype=dtype, device=device)
    v = torch.zeros(tokens, heads, v_head, dtype=dtype, device=device)

    if mutation == "k_nope_ndim":
        with pytest.raises(ValueError):
            op(k_nope[0], k_pe, v)
    elif mutation == "v_ndim":
        with pytest.raises(ValueError):
            op(k_nope, k_pe, v[0])
    elif mutation == "kpe_ndim":
        with pytest.raises(ValueError):
            op(k_nope, k_pe[0, 0], v)
    elif mutation == "dtype_mismatch":
        with pytest.raises(ValueError):
            op(k_nope, k_pe.to(torch.float16), v)
    elif mutation == "bad_fp8_dtype":
        with pytest.raises(ValueError):
            op(k_nope, k_pe, v, fp8_dtype=torch.bfloat16)
    elif mutation == "token_mismatch":
        with pytest.raises(ValueError):
            op(
                k_nope,
                torch.zeros(
                    tokens + 1, 1, qk_rope, dtype=dtype, device=device
                ),
                v,
            )
    elif mutation == "head_mismatch":
        with pytest.raises(ValueError):
            op(
                k_nope,
                k_pe,
                torch.zeros(
                    tokens, heads + 1, v_head, dtype=dtype, device=device
                ),
            )
    elif mutation == "kpe_head_not_one":
        with pytest.raises(ValueError):
            op(
                k_nope,
                torch.zeros(tokens, 2, qk_rope, dtype=dtype, device=device),
                v,
            )
    else:
        # A stepped inner slice is the only inner-dim view that breaks
        # stride(-1) == 1; a plain prefix slice keeps unit stride and is
        # accepted, which is the behaviour the strided-input test relies on.
        wide = torch.zeros(
            tokens, heads, qk_nope * 2, dtype=dtype, device=device
        )
        stepped = wide[..., ::2]
        assert stepped.stride(-1) == 2
        with pytest.raises(ValueError):
            op(stepped, k_pe, v)
