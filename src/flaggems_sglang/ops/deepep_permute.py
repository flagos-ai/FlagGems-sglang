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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _permute_kernel(
    in_ptr,
    out_ptr,
    slot_ptr,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    CACHE: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    if EVEN:
        src = tl.load(
            in_ptr + token * HIDDEN + offs,
            cache_modifier=".cg" if CACHE else "",
        ).to(out_dtype)
    else:
        src = tl.load(
            in_ptr + token * HIDDEN + offs,
            mask=mask,
            other=0,
            cache_modifier=".cg" if CACHE else "",
        ).to(out_dtype)
    base = slot_ptr + token * TOPK
    for i in tl.static_range(TOPK):
        dst = tl.load(base + i)
        if WIDE:
            dst = dst.to(tl.int64)
        if dst >= 0:
            if EVEN:
                tl.store(
                    out_ptr + dst * HIDDEN + offs,
                    src,
                    cache_modifier=".cs" if CACHE else "",
                )
            else:
                tl.store(
                    out_ptr + dst * HIDDEN + offs,
                    src,
                    mask=mask,
                    cache_modifier=".cs" if CACHE else "",
                )


def _block(hidden):
    if hidden % 4096 == 0:
        return 4096
    if hidden % 2048 == 0:
        return 2048
    if hidden % 1024 == 0:
        return 1024
    if hidden % 512 == 0:
        return 512
    if hidden % 256 == 0:
        return 256
    return 128


def _fallback(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    source = input.contiguous()
    target = gateup_input.contiguous()
    slots = src2dst
    if not slots.is_contiguous():
        slots = slots.contiguous()
    rows = target.shape[0]
    hidden = source.shape[1]
    count = slots.numel()
    tokens = min(source.shape[0], count // topk)
    block = _block(hidden)
    wide = rows * hidden >= 2**31
    _permute_kernel[max(tokens, 1), max(triton.cdiv(hidden, block), 1)](
        source,
        target,
        slots,
        hidden,
        topk,
        block,
        hidden % block == 0,
        wide,
        num_warps=4,
        CACHE=tokens >= 64,
    )
    return target


@triton.jit
def _route(
    X,
    OUT,
    INDICES,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    MODE: tl.constexpr,
    P: tl.constexpr,
    CACHE: tl.constexpr,
):
    col = tl.program_id(1) * B + tl.arange(0, B)
    for token in range(tl.program_id(0), N, P):
        value = tl.load(
            X + token * H + col,
            col < H,
            0,
            cache_modifier=".cg" if CACHE else "",
        ).to(OUT.dtype.element_ty)
        if MODE == 2:
            slot = tl.program_id(2)
            dst = tl.load(INDICES + token * K + slot).to(tl.int64)
            tl.store(
                OUT + dst * H + col,
                value,
                (dst >= 0) & (col < H),
                cache_modifier=".cs" if CACHE else "",
            )
        else:
            for slot in tl.static_range(K):
                dst = tl.load(INDICES + token * K + slot).to(tl.int64)
                if MODE == 0:
                    if dst >= 0:
                        tl.store(
                            OUT + dst * H + col,
                            value,
                            col < H,
                            cache_modifier=".cs" if CACHE else "",
                        )
                else:
                    tl.store(
                        OUT + dst * H + col,
                        value,
                        (dst >= 0) & (col < H),
                        cache_modifier=".cs" if CACHE else "",
                    )


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    n, h = input.shape
    if not (n <= 32 or (n >= 512 and h >= 4096)):
        return _fallback(
            input, gateup_input, src2dst, topk_ids, topk, hidden_size
        )
    source = input.contiguous()
    target = gateup_input.contiguous()
    slots = src2dst.contiguous()
    n = min(n, slots.numel() // topk) if topk else 0
    if not n or not h:
        return target
    written = target
    mode = 2 if n <= 32 else 1
    block = 2048 if n <= 32 else 4096
    if (
        n >= 512
        and source.dtype == target.dtype
        and (source.element_size() == 2)
        and (h % 2 == 0)
    ):
        source = source.view(torch.int32)
        written = target.view(torch.int32)
        h //= 2
    block = min(block, triton.next_power_of_2(h))
    _route.run(
        source,
        written,
        slots,
        n,
        h,
        topk,
        block,
        mode,
        n,
        grid=(n, triton.cdiv(h, block), topk if mode == 2 else 1),
        warmup=False,
        num_warps=4,
        num_stages=1,
        CACHE=n >= 64,
    )
    return target


__all__ = ["deepep_permute"]
