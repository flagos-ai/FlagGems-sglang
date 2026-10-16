# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1

_FUSED_MAX_TOKENS = 2048


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["N", "MODE"],
)
@triton.jit
def _gather_split_kernel(
    kv_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    N,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    MODE: tl.constexpr,
    WIDE: tl.constexpr,
    stride_kv,
    stride_nope,
    stride_rope,
):
    pid = tl.program_id(0)
    row = tl.load(loc_ptr + pid)
    if WIDE:
        idx = pid.to(tl.int64)
        row = row.to(tl.int64)
    else:
        idx = pid
        row = row.to(tl.int32)
    src = kv_ptr + row * stride_kv

    if MODE == 0:
        offs_n = tl.arange(0, BLOCK_NOPE)
        offs_r = tl.arange(0, BLOCK_ROPE)
        vn = tl.load(src + offs_n, mask=offs_n < NOPE, other=0)
        vr = tl.load(src + NOPE + offs_r, mask=offs_r < ROPE, other=0)
        tl.store(
            nope_ptr + idx * stride_nope + offs_n,
            vn.to(nope_ptr.dtype.element_ty),
            mask=offs_n < NOPE,
        )
        tl.store(
            rope_ptr + idx * stride_rope + offs_r,
            vr.to(rope_ptr.dtype.element_ty),
            mask=offs_r < ROPE,
        )
    else:
        offs = tl.arange(0, BLOCK_ROW)
        mask = offs < NOPE + ROPE
        vals = tl.load(src + offs, mask=mask, other=0)

        tl.store(
            nope_ptr + idx * stride_nope + offs,
            vals.to(nope_ptr.dtype.element_ty),
            mask=offs < NOPE,
        )
        tl.store(
            rope_ptr + idx * stride_rope + (offs - NOPE),
            vals.to(rope_ptr.dtype.element_ty),
            mask=(offs >= NOPE) & mask,
        )


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    n = loc.shape[0]

    if n == 0 or nope_dim + rope_dim == 0:
        return cache_k_nope, cache_k_rope

    wide = (
        (n - 1) * kv_buffer.stride(0) + nope_dim + rope_dim > _INT32_MAX
        or (n - 1) * cache_k_nope.stride(0) + nope_dim > _INT32_MAX
        or (n - 1) * cache_k_rope.stride(0) + rope_dim > _INT32_MAX
    )

    _gather_split_kernel[(n,)](
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        n,
        nope_dim,
        rope_dim,
        max(triton.next_power_of_2(nope_dim), 1),
        max(triton.next_power_of_2(rope_dim), 1),
        triton.next_power_of_2(nope_dim + rope_dim),
        stride_kv=kv_buffer.stride(0),
        stride_nope=cache_k_nope.stride(0),
        stride_rope=cache_k_rope.stride(0),
        MODE=1 if n <= _FUSED_MAX_TOKENS else 0,
        WIDE=wide,
    )
    return cache_k_nope, cache_k_rope


__all__ = ["get_mla_kv_buffer"]
