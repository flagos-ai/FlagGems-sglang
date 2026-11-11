# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write token-major K/V into paged FlashAttention caches.

Source layout is ``[num_tokens, num_heads, head_size]`` and the caches are
``[num_blocks, block_size, num_heads, head_size]``. One program handles one
token and one block of heads: it resolves the token's slot, optionally remaps
it through a sliding-window table, skips negative slots, and scatters the
head rows into the block the slot falls in.

Two details differ from the upstream kernel this mirrors, both driven by the
operator contract rather than by preference:

- the contract declares ``mutates: []`` and returns ``key_cache_out`` /
  ``value_cache_out``, so this writes into copies and leaves the inputs
  untouched. Upstream writes in place.
- the contract asks to ignore negative *resolved* slots, so the check runs
  both before and after the SWA remap. Upstream only checks after, which means
  a negative slot reaches the remap table first and reads out of bounds.
"""

import triton
import triton.language as tl

# Heads processed per program. Matches the upstream launch configuration; it
# keeps a two-dimensional tile in registers without exceeding the minimum
# grid the kernel needs.
HEAD_BLOCK = 4


@triton.jit
def _reshape_and_cache_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    value_block_stride,
    cache_token_stride,
    cache_head_stride,
    value_cache_token_stride,
    value_cache_head_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)
        if slot_idx < 0:
            return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_idx = tl.arange(0, BLOCK_D)

    offs = head_idx[:, None] * head_size + dim_idx[None, :]
    mask = (head_idx[:, None] < num_heads) & (dim_idx[None, :] < head_size)

    k = tl.load(key_ptr + token_idx * key_stride + offs, mask=mask)
    v = tl.load(value_ptr + token_idx * value_stride + offs, mask=mask)

    if USE_SCALE:
        # The contract specifies division, which is what the upstream kernel
        # does too; taking a reciprocal here would not be equivalent.
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        k = k / k_scale
        v = v / v_scale

    tgt_k = (
        block_idx * block_stride
        + block_offset * cache_token_stride
        + head_idx[:, None] * cache_head_stride
        + dim_idx[None, :]
    )
    tgt_v = (
        block_idx * value_block_stride
        + block_offset * value_cache_token_stride
        + head_idx[:, None] * value_cache_head_stride
        + dim_idx[None, :]
    )
    tl.store(key_cache_ptr + tgt_k, k, mask=mask)
    tl.store(value_cache_ptr + tgt_v, v, mask=mask)


def reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
):
    """Scatter tokens into paged caches and return the updated copies.

    Args:
        key: ``[num_tokens, num_heads, head_size]`` source keys.
        value: same shape as ``key``, source values.
        key_cache: ``[num_blocks, block_size, num_heads, head_size]``.
        value_cache: same shape as ``key_cache``.
        slot_mapping: ``[num_tokens]``; negative means the token is skipped.
        swa_slot_mapping: optional ``[...]`` remap applied to resolved slots.
        k_scale: optional scalar dividing keys before the write.
        v_scale: optional scalar dividing values before the write.

    Returns:
        ``(key_cache_out, value_cache_out)``, copies of the caches.
    """
    if key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            f"key and value must be 3D, got {key.dim()}D and {value.dim()}D"
        )
    if key.shape != value.shape:
        raise ValueError(
            f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}"
        )
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            f"key_cache and value_cache must be 4D, got "
            f"{key_cache.dim()}D and {value_cache.dim()}D"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache shape {tuple(key_cache.shape)} != "
            f"value_cache shape {tuple(value_cache.shape)}"
        )
    if slot_mapping.dim() != 1:
        raise ValueError(f"slot_mapping must be 1D, got {slot_mapping.dim()}D")
    if slot_mapping.shape[0] != key.shape[0]:
        raise ValueError(
            f"slot_mapping has {slot_mapping.shape[0]} entries but key has "
            f"{key.shape[0]} tokens"
        )
    if (k_scale is None) != (v_scale is None):
        raise ValueError("k_scale and v_scale must be given together")

    num_tokens, num_heads, head_size = key.shape
    block_size = key_cache.shape[1]

    # The contract returns the caches rather than mutating them, so the write
    # target is a copy. This is the difference from upstream, and it costs a
    # full cache copy per call.
    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()

    if num_tokens == 0:
        return key_cache_out, value_cache_out

    grid = (num_tokens, triton.cdiv(num_heads, HEAD_BLOCK))
    _reshape_and_cache_flash[grid](
        key,
        value,
        key_cache_out,
        value_cache_out,
        slot_mapping,
        swa_slot_mapping if swa_slot_mapping is not None else slot_mapping,
        # With USE_SCALE off the kernel never reads these, so reusing
        # slot_mapping keeps a null-free signature. Upstream does the same.
        k_scale if k_scale is not None else slot_mapping,
        v_scale if v_scale is not None else slot_mapping,
        key_cache_out.stride(0),
        value_cache_out.stride(0),
        key_cache_out.stride(1),
        key_cache_out.stride(2),
        value_cache_out.stride(1),
        value_cache_out.stride(2),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        block_size,
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=triton.next_power_of_2(head_size),
        HAS_SWA=swa_slot_mapping is not None,
        USE_SCALE=k_scale is not None,
    )
    return key_cache_out, value_cache_out


__all__ = ["reshape_and_cache_flash"]
