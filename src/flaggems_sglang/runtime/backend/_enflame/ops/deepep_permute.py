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

# GCU400: 48 SPs. Persistent single-wave grid keeps the scattered-row store
# pipeline at the flat-copy floor; more programs only add dispatch waves.
_NUM_SP = 48

# 11-07 20:22 toolchain regression guard (memref_ext.slice_pad_start 8 vs 7):
# per-warp slice must stay <= 4096 elems (8KB) and per-CTA <= 32KB. bf16
# BLOCK=8192 @ nw2 = 4096 elems/warp and 16KB/CTA; @ nw4 mode2 = 2048/4KB.
_MAX_BLOCK = 8192


@triton.jit
def _permute_kernel(
    in_ptr,
    out_ptr,
    slot_ptr,
    n_tokens,
    H: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    NB: tl.constexpr,
    WIDE: tl.constexpr,
    EVEN: tl.constexpr,
):
    # E-1 round-2 (2026-11-13): hoist all K destination scalars to the top of
    # the token iteration so the K slot loads issue back-to-back (independent)
    # instead of alternating load->store->load; the scattered stores then fire
    # as one burst per row. Specialized arms for the production widths K=8/4/2
    # (bench core -13%/-15%/-20% at t=8/64/512 vs the in-loop scalar chain),
    # generic static_range fallback keeps arbitrary K correct.
    out_dtype = out_ptr.dtype.element_ty
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for token in range(pid, n_tokens, npg):
        base = slot_ptr + token * K
        if K == 8:
            d0 = tl.load(base + 0)
            d1 = tl.load(base + 1)
            d2 = tl.load(base + 2)
            d3 = tl.load(base + 3)
            d4 = tl.load(base + 4)
            d5 = tl.load(base + 5)
            d6 = tl.load(base + 6)
            d7 = tl.load(base + 7)
            if WIDE:
                d0 = d0.to(tl.int64)
                d1 = d1.to(tl.int64)
                d2 = d2.to(tl.int64)
                d3 = d3.to(tl.int64)
                d4 = d4.to(tl.int64)
                d5 = d5.to(tl.int64)
                d6 = d6.to(tl.int64)
                d7 = d7.to(tl.int64)
            for j in tl.static_range(NB):
                offs = j * B + tl.arange(0, B)
                if EVEN:
                    src = tl.load(in_ptr + token * H + offs).to(out_dtype)
                else:
                    src = tl.load(
                        in_ptr + token * H + offs, mask=offs < H, other=0
                    ).to(out_dtype)
                if d0 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d0 * H + offs, src)
                    else:
                        tl.store(out_ptr + d0 * H + offs, src, mask=offs < H)
                if d1 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d1 * H + offs, src)
                    else:
                        tl.store(out_ptr + d1 * H + offs, src, mask=offs < H)
                if d2 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d2 * H + offs, src)
                    else:
                        tl.store(out_ptr + d2 * H + offs, src, mask=offs < H)
                if d3 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d3 * H + offs, src)
                    else:
                        tl.store(out_ptr + d3 * H + offs, src, mask=offs < H)
                if d4 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d4 * H + offs, src)
                    else:
                        tl.store(out_ptr + d4 * H + offs, src, mask=offs < H)
                if d5 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d5 * H + offs, src)
                    else:
                        tl.store(out_ptr + d5 * H + offs, src, mask=offs < H)
                if d6 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d6 * H + offs, src)
                    else:
                        tl.store(out_ptr + d6 * H + offs, src, mask=offs < H)
                if d7 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d7 * H + offs, src)
                    else:
                        tl.store(out_ptr + d7 * H + offs, src, mask=offs < H)
        elif K == 4:
            d0 = tl.load(base + 0)
            d1 = tl.load(base + 1)
            d2 = tl.load(base + 2)
            d3 = tl.load(base + 3)
            if WIDE:
                d0 = d0.to(tl.int64)
                d1 = d1.to(tl.int64)
                d2 = d2.to(tl.int64)
                d3 = d3.to(tl.int64)
            for j in tl.static_range(NB):
                offs = j * B + tl.arange(0, B)
                if EVEN:
                    src = tl.load(in_ptr + token * H + offs).to(out_dtype)
                else:
                    src = tl.load(
                        in_ptr + token * H + offs, mask=offs < H, other=0
                    ).to(out_dtype)
                if d0 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d0 * H + offs, src)
                    else:
                        tl.store(out_ptr + d0 * H + offs, src, mask=offs < H)
                if d1 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d1 * H + offs, src)
                    else:
                        tl.store(out_ptr + d1 * H + offs, src, mask=offs < H)
                if d2 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d2 * H + offs, src)
                    else:
                        tl.store(out_ptr + d2 * H + offs, src, mask=offs < H)
                if d3 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d3 * H + offs, src)
                    else:
                        tl.store(out_ptr + d3 * H + offs, src, mask=offs < H)
        elif K == 2:
            d0 = tl.load(base + 0)
            d1 = tl.load(base + 1)
            if WIDE:
                d0 = d0.to(tl.int64)
                d1 = d1.to(tl.int64)
            for j in tl.static_range(NB):
                offs = j * B + tl.arange(0, B)
                if EVEN:
                    src = tl.load(in_ptr + token * H + offs).to(out_dtype)
                else:
                    src = tl.load(
                        in_ptr + token * H + offs, mask=offs < H, other=0
                    ).to(out_dtype)
                if d0 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d0 * H + offs, src)
                    else:
                        tl.store(out_ptr + d0 * H + offs, src, mask=offs < H)
                if d1 >= 0:
                    if EVEN:
                        tl.store(out_ptr + d1 * H + offs, src)
                    else:
                        tl.store(out_ptr + d1 * H + offs, src, mask=offs < H)
        else:
            for j in tl.static_range(NB):
                offs = j * B + tl.arange(0, B)
                if EVEN:
                    src = tl.load(in_ptr + token * H + offs).to(out_dtype)
                else:
                    src = tl.load(
                        in_ptr + token * H + offs, mask=offs < H, other=0
                    ).to(out_dtype)
                for i in tl.static_range(K):
                    dst = tl.load(base + i)
                    if WIDE:
                        dst = dst.to(tl.int64)
                    if dst >= 0:
                        if EVEN:
                            tl.store(out_ptr + dst * H + offs, src)
                        else:
                            tl.store(
                                out_ptr + dst * H + offs, src, mask=offs < H
                            )


@triton.jit
def _permute_one_kernel(
    in_ptr,
    out_ptr,
    slot_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    NB: tl.constexpr,
    WIDE: tl.constexpr,
    EVEN: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty
    token = tl.program_id(0)
    slot = tl.program_id(1)
    dst = tl.load(slot_ptr + token * K + slot)
    if WIDE:
        dst = dst.to(tl.int64)
    if dst >= 0:
        for j in tl.static_range(NB):
            offs = j * B + tl.arange(0, B)
            if EVEN:
                src = tl.load(in_ptr + token * H + offs).to(out_dtype)
                tl.store(out_ptr + dst * H + offs, src)
            else:
                src = tl.load(
                    in_ptr + token * H + offs, mask=offs < H, other=0
                ).to(out_dtype)
                tl.store(out_ptr + dst * H + offs, src, mask=offs < H)


def _plan(hidden):
    block = min(triton.next_power_of_2(hidden), _MAX_BLOCK)
    nblocks = triton.cdiv(hidden, block)
    return block, nblocks, hidden % block == 0


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    source = input.contiguous()
    target = gateup_input.contiguous()
    slots = src2dst
    if not slots.is_contiguous():
        slots = slots.contiguous()
    rows = target.shape[0]
    hidden = source.shape[1]
    tokens = min(source.shape[0], slots.numel() // topk) if topk else 0
    if not tokens or not hidden:
        return target
    wide = rows * hidden >= 2**31
    block, nblocks, even = _plan(hidden)
    if tokens == 1:
        _permute_one_kernel[(1, topk)](
            source,
            target,
            slots,
            hidden,
            topk,
            block,
            nblocks,
            wide,
            even,
            num_warps=4,
        )
    else:
        _permute_kernel[(min(tokens, _NUM_SP),)](
            source,
            target,
            slots,
            tokens,
            hidden,
            topk,
            block,
            nblocks,
            wide,
            even,
            num_warps=2,
            num_stages=2,
        )
    return target


__all__ = ["deepep_permute"]
