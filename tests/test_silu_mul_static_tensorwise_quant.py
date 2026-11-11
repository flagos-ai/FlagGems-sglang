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

"""Correctness test for silu_mul_static_tensorwise_quant."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "silu_mul_static_tensorwise_quant"

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2


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
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # The payload is eight-bit, so it is compared on the encoded bytes and
    # needs no tolerance at all.
    torch.testing.assert_close(
        _bytes(actual), _bytes(expected), rtol=0.0, atol=0.0
    )


def _inputs(rows, intermediate_size, active, dtype=torch.bfloat16, seed=0):
    """Build the six arguments, with an output buffer of non-zero contents."""
    device = flaggems_sglang.device
    generator = torch.Generator(device=device).manual_seed(seed)
    width = 2 * intermediate_size
    x = (
        torch.randn(
            (rows, width),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(3.0)
        .to(dtype)
    )
    # A buffer that is not zero-filled: whatever the kernel does not touch has
    # to come back unchanged, and a zero tail would hide a bug that zeroes it.
    initial = torch.randn(
        (rows, intermediate_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(FP8_E4M3)
    scale = torch.tensor([2.0], device=device, dtype=torch.float32)
    num_tokens = torch.tensor([active], device=device, dtype=torch.int32)
    return x, initial, scale, num_tokens, rows, intermediate_size


SHAPES = [
    # A full batch: every row is active.
    (8, 512, 8),
    # A decode-sized prefix in a prefill-sized buffer.
    (8, 512, 3),
    # Odd width, so the tail tile is masked on almost every program.
    (33, 576, 33),
    # Nothing active: the buffer has to come back exactly as it went in.
    (33, 576, 0),
    # One row, one token.
    (1, 128, 1),
    # Two tiles worth of width.
    (5, 2048, 5),
]


@pytest.mark.silu_mul_static_tensorwise_quant
@pytest.mark.parametrize("rows,intermediate_size,active", SHAPES)
def test_silu_mul_static_tensorwise_quant(rows, intermediate_size, active):
    ref, cand = _pair()
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )

    expected = ref(x, initial, scale, num_tokens, _rows, _width)
    actual = cand(x, initial, scale, num_tokens, _rows, _width)
    _check(actual, expected)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_rows_beyond_the_token_count_keep_the_caller_bytes():
    """The active prefix is replaced; the rest of the buffer is a pure copy."""
    ref, cand = _pair()
    rows, intermediate_size, active = 8, 512, 2
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )
    before = _bytes(initial).clone()

    actual = cand(x, initial, scale, num_tokens, _rows, _width)
    _check(actual, ref(x, initial, scale, num_tokens, _rows, _width))

    # The tail is byte-identical and the prefix is not.
    assert torch.equal(_bytes(actual)[active:], before[active:])
    assert not torch.equal(_bytes(actual)[:active], before[:active])


@pytest.mark.silu_mul_static_tensorwise_quant
def test_zero_tokens_returns_an_untouched_copy():
    ref, cand = _pair()
    rows, intermediate_size, active = 6, 256, 0
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )

    actual = cand(x, initial, scale, num_tokens, _rows, _width)
    assert torch.equal(_bytes(actual), _bytes(initial))
    # A copy, not the caller's allocation handed back.
    assert actual.data_ptr() != initial.data_ptr()


@pytest.mark.silu_mul_static_tensorwise_quant
def test_an_overstated_token_count_is_clamped():
    """A count past the capacity fills the buffer instead of running off it."""
    ref, cand = _pair()
    rows, intermediate_size = 4, 512
    x, initial, scale, _num, _rows, _width = _inputs(
        rows, intermediate_size, rows
    )
    num_tokens = torch.tensor(
        [rows + 5], device=scale.device, dtype=torch.int32
    )

    actual = cand(x, initial, scale, num_tokens, _rows, _width)
    _check(actual, ref(x, initial, scale, num_tokens, _rows, _width))
    assert not torch.equal(_bytes(actual)[-1], _bytes(initial)[-1])


@pytest.mark.silu_mul_static_tensorwise_quant
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_input_dtypes(dtype):
    ref, cand = _pair()
    rows, intermediate_size, active = 7, 384, 5
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active, dtype=dtype
    )

    _check(
        cand(x, initial, scale, num_tokens, _rows, _width),
        ref(x, initial, scale, num_tokens, _rows, _width),
    )


@pytest.mark.silu_mul_static_tensorwise_quant
def test_second_fp8_dtype_within_range():
    """e5m2 is a distinct encoding, so the bytes are checked on it too."""
    ref, cand = _pair()
    rows, intermediate_size, active = 5, 128, 5
    device = flaggems_sglang.device
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )
    # A scale wide enough to keep every value inside the finite range, where
    # the kernel's saturating store and torch's cast agree.
    wide = torch.tensor([64.0], device=device, dtype=torch.float32)
    buffer = initial.float().to(FP8_E5M2)

    _check(
        cand(x, buffer, wide, num_tokens, _rows, _width),
        ref(x, buffer, wide, num_tokens, _rows, _width),
    )


@pytest.mark.silu_mul_static_tensorwise_quant
def test_neither_buffer_is_mutated():
    ref, cand = _pair()
    rows, intermediate_size, active = 4, 256, 4
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )
    x_before = x.clone()
    buffer_before = initial.clone()

    cand(x, initial, scale, num_tokens, _rows, _width)
    assert torch.equal(x, x_before)
    assert torch.equal(initial, buffer_before)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_padded_row_stride():
    """Extra columns between rows are read past, not folded into the width."""
    ref, cand = _pair()
    rows, intermediate_size, active = 5, 256, 3
    device = flaggems_sglang.device
    width = 2 * intermediate_size
    generator = torch.Generator(device=device).manual_seed(3)
    storage = (
        torch.randn(
            (rows, 2 * width),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(3.0)
        .to(torch.bfloat16)
    )
    x = storage[:, :width]
    assert x.stride(0) == 2 * width
    assert x.stride(1) == 1

    _, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )
    _check(
        cand(x, initial, scale, num_tokens, _rows, _width),
        ref(x, initial, scale, num_tokens, _rows, _width),
    )


@pytest.mark.silu_mul_static_tensorwise_quant
def test_more_rows_than_tokens_still_uses_the_head_of_the_buffer():
    """The active rows are the head, whatever the buffer capacity is."""
    ref, cand = _pair()
    rows, intermediate_size, active = 16, 1024, 1
    x, initial, scale, num_tokens, _rows, _width = _inputs(
        rows, intermediate_size, active
    )

    actual = cand(x, initial, scale, num_tokens, _rows, _width)
    _check(actual, ref(x, initial, scale, num_tokens, _rows, _width))
    # Only row 0 changed.
    assert torch.equal(_bytes(actual)[1:], _bytes(initial)[1:])


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_mismatched_last_dimension_is_rejected():
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 96), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((4, 64), device=device, dtype=FP8_E4M3)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_non_fp8_output_is_rejected():
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((4, 64), device=device, dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_mismatched_output_shape_is_rejected():
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((8, 64), device=device, dtype=FP8_E4M3)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_vector_scale_is_rejected():
    """One static scale for the tensor, not one per row."""
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((4, 64), device=device, dtype=FP8_E4M3)
    scale = torch.ones((4,), device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_a_non_int32_token_count_is_rejected(dtype):
    """The kernel reads the count as int32, so the width has to match."""
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((4, 64), device=device, dtype=FP8_E4M3)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=dtype)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
@pytest.mark.parametrize("dtype", [torch.int8, torch.float64])
def test_an_unsupported_input_dtype_is_rejected(dtype):
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=dtype)
    buffer = torch.zeros((4, 64), device=device, dtype=FP8_E4M3)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_cpu_tensor_is_rejected():
    _, cand = _pair()
    x = torch.zeros((4, 128), dtype=torch.bfloat16)
    buffer = torch.zeros((4, 64), dtype=FP8_E4M3)
    scale = torch.tensor([1.0])
    num_tokens = torch.tensor([4], dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 4, 64)


@pytest.mark.silu_mul_static_tensorwise_quant
def test_a_capacity_beyond_the_input_rows_is_rejected():
    _, cand = _pair()
    device = flaggems_sglang.device
    x = torch.zeros((4, 128), device=device, dtype=torch.bfloat16)
    buffer = torch.zeros((8, 64), device=device, dtype=FP8_E4M3)
    scale = torch.tensor([1.0], device=device)
    num_tokens = torch.tensor([4], device=device, dtype=torch.int32)
    with pytest.raises(ValueError):
        cand(x, buffer, scale, num_tokens, 8, 64)
