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

import triton
import triton.language as tl


@triton.jit
def _permute_kernel(
    in_ptr,
    out_ptr,
    slot_ptr,
    HIDDEN: tl.constexpr,
    TOKENS,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    for token in tl.range(tl.program_id(0), TOKENS, tl.num_programs(0)):
        if EVEN:
            src = tl.load(in_ptr + token * HIDDEN + offs).to(out_dtype)
        else:
            src = tl.load(
                in_ptr + token * HIDDEN + offs, mask=mask, other=0
            ).to(out_dtype)
        base = slot_ptr + token * TOPK
        for i in tl.static_range(TOPK):
            dst = tl.load(base + i)
            if dst >= 0:
                if EVEN:
                    tl.store(out_ptr + dst * HIDDEN + offs, src)
                else:
                    tl.store(out_ptr + dst * HIDDEN + offs, src, mask=mask)


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    source = input.contiguous()
    target = gateup_input.contiguous()
    slots = src2dst.contiguous()
    n, h = source.shape
    n = min(n, slots.numel() // topk) if topk else 0
    if not n or not h:
        return target
    b = 1 << (h - 1).bit_length()
    p = min(n, 40)
    if n == 1:
        p = min(topk, 256)
        _split_route[(p,)](
            source, target, slots, n, h, topk, b, num_warps=4, num_stages=1
        )
        return target
    _permute_kernel[(p,)](
        source, target, slots, h, n, topk, b, h == b, num_warps=4, num_stages=1
    )
    return target


@triton.jit
def _split_route(
    X,
    OUT,
    INDICES,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    col = tl.arange(0, B)
    for route in range(tl.program_id(0), N * K, tl.num_programs(0)):
        src = route // K
        dst = tl.load(INDICES + route).to(tl.int64)
        if dst >= 0:
            value = tl.load(X + src * H + col, col < H, 0).to(
                OUT.dtype.element_ty
            )
            tl.store(OUT + dst * H + col, value, col < H)


__all__ = ["deepep_permute"]
