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

"""Correctness test for reshape_and_cache_flash."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

OP = "reshape_and_cache_flash"


def _build(shape, device, seed=0, dtype=torch.float16):
    """Return key, value, caches, slot_mapping for a shape quadruple."""
    num_tokens, num_heads, head_size, num_blocks, block_size = shape
    g = torch.Generator(device=device).manual_seed(seed)

    key = torch.randn(
        (num_tokens, num_heads, head_size),
        generator=g,
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        (num_tokens, num_heads, head_size),
        generator=g,
        device=device,
        dtype=dtype,
    )
    key_cache = torch.zeros(
        (num_blocks, block_size, num_heads, head_size),
        device=device,
        dtype=dtype,
    )
    value_cache = torch.zeros(
        (num_blocks, block_size, num_heads, head_size),
        device=device,
        dtype=dtype,
    )

    capacity = num_blocks * block_size
    # Ascending unique slots keep the mapping legal; some callers only ever
    # fill the front of a block, but the kernel must not care either way.
    slots = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) % capacity
    )
    return key, value, key_cache, value_cache, slots


def _check(actual, expected):
    ak, av = actual
    ek, ev = expected

    assert ak.shape == ek.shape
    assert av.shape == ev.shape
    assert ak.dtype == ek.dtype
    assert av.dtype == ev.dtype

    torch.testing.assert_close(ak, ek, rtol=0.0, atol=0.0)
    torch.testing.assert_close(av, ev, rtol=0.0, atol=0.0)


# (num_tokens, num_heads, head_size, num_blocks, block_size)
_CASES = [
    lambda: (8, 4, 128, 2, 16),
    lambda: (32, 8, 64, 4, 8),
    lambda: (1, 1, 32, 1, 4),
    lambda: (64, 16, 128, 8, 8),
    # head_size that is not a power of two exercises the padded BLOCK_D path.
    lambda: (4, 2, 100, 2, 8),
    # Fewer heads than HEAD_BLOCK leaves a partially masked second tile.
    lambda: (16, 1, 64, 2, 8),
]

CASE_IDS = [f"{c()[0]}t{c()[1]}h{c()[2]}d" for c in _CASES]


@pytest.mark.parametrize("shape", [c() for c in _CASES], ids=CASE_IDS)
@pytest.mark.reshape_and_cache_flash
def test_reshape_and_cache_flash(shape):
    device = flaggems_sglang.device
    key, value, key_cache, value_cache, slots = _build(shape, device)

    reference = get_reference(OP)
    expected = reference(key, value, key_cache, value_cache, slots)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(f"{OP} ops module not found")
        return

    try:
        actual = candidate(key, value, key_cache, value_cache, slots)
    except NotImplementedError:
        pytest.skip(f"{OP} not yet implemented")
        return

    _check(actual, expected)


@pytest.mark.reshape_and_cache_flash
def test_inputs_are_not_mutated():
    """The contract declares mutates: [], so the caller's caches must survive."""
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    key, value, key_cache, value_cache, slots = _build(
        (8, 4, 128, 2, 16), device
    )
    key_cache_before = key_cache.clone()
    value_cache_before = value_cache.clone()

    out_k, out_v = op(key, value, key_cache, value_cache, slots)

    assert torch.equal(key_cache, key_cache_before), "key_cache was mutated"
    assert torch.equal(
        value_cache, value_cache_before
    ), "value_cache was mutated"
    # And the returned tensors must be separate storage, not the inputs.
    assert out_k.data_ptr() != key_cache.data_ptr()
    assert out_v.data_ptr() != value_cache.data_ptr()


@pytest.mark.reshape_and_cache_flash
def test_negative_slots_are_skipped():
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    key, value, key_cache, value_cache, _ = _build((8, 4, 128, 2, 16), device)
    # Every token is skipped, so the caches must come back exactly as given.
    slots = torch.full((8,), -1, device=device, dtype=torch.int32)

    out_k, out_v = op(key, value, key_cache, value_cache, slots)

    assert torch.equal(out_k, key_cache)
    assert torch.equal(out_v, value_cache)


@pytest.mark.reshape_and_cache_flash
def test_swa_remap_is_applied():
    """A remap table redirects slots; nothing may land at the original index."""
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    shape = (8, 4, 128, 2, 16)
    key, value, key_cache, value_cache, slots = _build(shape, device)

    capacity = 2 * 16
    # Shift every slot by one position, wrapping inside the cache.
    swa = torch.tensor(
        [(i + 1) % capacity for i in range(capacity)],
        device=device,
        dtype=torch.int32,
    )

    reference = get_reference(OP)
    expected = reference(key, value, key_cache, value_cache, slots, swa)
    actual = op(key, value, key_cache, value_cache, slots, swa)
    _check(actual, expected)


@pytest.mark.reshape_and_cache_flash
def test_swa_remap_to_negative_is_also_skipped():
    """The contract says negative *resolved* slots are ignored, not just raw ones."""
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    shape = (8, 4, 128, 2, 16)
    key, value, key_cache, value_cache, slots = _build(shape, device)

    capacity = 2 * 16
    swa = torch.full((capacity,), -1, device=device, dtype=torch.int32)

    out_k, out_v = op(key, value, key_cache, value_cache, slots, swa)
    assert torch.equal(out_k, key_cache)
    assert torch.equal(out_v, value_cache)

    reference = get_reference(OP)
    _check(
        (out_k, out_v),
        reference(key, value, key_cache, value_cache, slots, swa),
    )


@pytest.mark.reshape_and_cache_flash
def test_scales_divide_before_the_write():
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    shape = (8, 4, 128, 2, 16)
    key, value, key_cache, value_cache, slots = _build(shape, device)
    k_scale = torch.tensor(2.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(4.0, device=device, dtype=torch.float32)

    reference = get_reference(OP)
    expected = reference(
        key, value, key_cache, value_cache, slots, None, k_scale, v_scale
    )
    actual = op(
        key, value, key_cache, value_cache, slots, None, k_scale, v_scale
    )
    _check(actual, expected)


@pytest.mark.reshape_and_cache_flash
def test_empty_token_batch():
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    key = torch.empty((0, 4, 128), device=device, dtype=torch.float16)
    value = torch.empty((0, 4, 128), device=device, dtype=torch.float16)
    key_cache = torch.zeros(
        (2, 16, 4, 128), device=device, dtype=torch.float16
    )
    value_cache = torch.zeros(
        (2, 16, 4, 128), device=device, dtype=torch.float16
    )
    slots = torch.empty((0,), device=device, dtype=torch.int32)

    out_k, out_v = op(key, value, key_cache, value_cache, slots)
    assert torch.equal(out_k, key_cache)
    assert torch.equal(out_v, value_cache)


@pytest.mark.reshape_and_cache_flash
@pytest.mark.parametrize(
    "mutation",
    ["bad_key_rank", "shape_mismatch", "bad_slots_len", "only_one_scale"],
)
def test_reshape_and_cache_flash_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.reshape_and_cache_flash

    key = torch.randn((8, 4, 128), device=device, dtype=torch.float16)
    value = torch.randn((8, 4, 128), device=device, dtype=torch.float16)
    key_cache = torch.zeros(
        (2, 16, 4, 128), device=device, dtype=torch.float16
    )
    value_cache = torch.zeros(
        (2, 16, 4, 128), device=device, dtype=torch.float16
    )
    slots = torch.arange(8, device=device, dtype=torch.int32)

    if mutation == "bad_key_rank":
        with pytest.raises(ValueError):
            op(key[0], value[0], key_cache, value_cache, slots)
    elif mutation == "shape_mismatch":
        with pytest.raises(ValueError):
            op(key, value[:4], key_cache, value_cache, slots)
    elif mutation == "bad_slots_len":
        with pytest.raises(ValueError):
            op(key, value, key_cache, value_cache, slots[:4])
    else:
        with pytest.raises(ValueError):
            op(
                key,
                value,
                key_cache,
                value_cache,
                slots,
                None,
                torch.tensor(2.0, device=device),
                None,
            )
