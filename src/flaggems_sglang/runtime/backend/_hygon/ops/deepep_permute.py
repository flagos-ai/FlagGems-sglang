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
):
    out_dtype = out_ptr.dtype.element_ty
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    if EVEN:
        src = tl.load(in_ptr + token * HIDDEN + offs).to(out_dtype)
    else:
        src = tl.load(in_ptr + token * HIDDEN + offs, mask=mask, other=0).to(
            out_dtype
        )
    base = slot_ptr + token * TOPK
    for i in tl.static_range(TOPK):
        dst = tl.load(base + i)
        if WIDE:
            dst = dst.to(tl.int64)
        if dst >= 0:
            if EVEN:
                tl.store(out_ptr + dst * HIDDEN + offs, src)
            else:
                tl.store(out_ptr + dst * HIDDEN + offs, src, mask=mask)


def _block(hidden):
    return triton.next_power_of_2(hidden)


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    source = input.contiguous()
    target = gateup_input.contiguous()
    slots = src2dst
    if not slots.is_contiguous():
        slots = slots.contiguous()
    rows = target.shape[0]
    hidden = source.shape[1]
    count = slots.numel()
    tokens = min(source.shape[0], count // topk)
    written = target
    if (
        source.dtype == target.dtype
        and target.element_size() == 2
        and hidden % 2 == 0
    ):
        source = source.view(torch.int32)
        written = target.view(torch.int32)
        hidden //= 2
    block = _block(hidden)
    wide = rows * hidden >= 2**31
    _permute_kernel[(max(tokens, 1), max(triton.cdiv(hidden, block), 1))](
        source,
        written,
        slots,
        hidden,
        topk,
        block,
        hidden % block == 0,
        wide,
        num_warps=2 if 512 <= tokens < 1024 and topk >= 4 else 4,
    )
    return target


__all__ = ["deepep_permute"]
