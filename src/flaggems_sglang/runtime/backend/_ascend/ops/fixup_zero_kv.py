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

_MAX_GRID = 24
_CHUNK = 32
_BLOCK_O = 8192


@triton.jit
def _fixup_dense_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    HD: tl.constexpr,
    N: tl.constexpr,
    BLOCK_L: tl.constexpr,
):

    pid = tl.program_id(0)
    o_offs = tl.arange(0, 8192)
    l_offs = tl.arange(0, BLOCK_L)
    for base in range(pid * 32, N, tl.num_programs(0) * 32):
        slot = base + tl.arange(0, 32)
        kv = tl.load(kv_lens_ptr + slot, mask=slot < N, other=1)
        if tl.min(kv, axis=0) == 0:
            for j in range(0, 32):
                i = base + j
                if i < N and tl.load(kv_lens_ptr + i) == 0:
                    beg = tl.load(cum_ptr + i)
                    n_rows = tl.load(cum_ptr + i + 1) - beg
                    o_total = n_rows * HD
                    for s in range(0, tl.cdiv(o_total, 8192)):
                        idx = s * 8192 + o_offs
                        tl.store(out_ptr + beg * HD + idx, 0, idx < o_total)
                    l_total = n_rows * H
                    for s in range(0, tl.cdiv(l_total, BLOCK_L)):
                        idx = s * BLOCK_L + l_offs
                        tl.store(
                            lse_ptr + beg * H + idx,
                            -float("inf"),
                            idx < l_total,
                        )


@triton.jit
def _fixup_rows_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    LSE_S0: tl.constexpr,
    LSE_S1: tl.constexpr,
    KV_S: tl.constexpr,
    CUM_S: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    pid = tl.program_id(0)
    i = pid // NUM_BLOCKS
    if tl.load(kv_lens_ptr + i * KV_S) == 0:
        beg = tl.load(cum_ptr + i * CUM_S)
        end = tl.load(cum_ptr + (i + 1) * CUM_S)
        r = pid % NUM_BLOCKS * BLOCK_R + tl.arange(0, BLOCK_R)
        tok = beg + r // H
        head = r % H
        d_offs = tl.arange(0, BLOCK_D)
        mask = (tok[:, None] < end) & (d_offs[None, :] < D)
        tl.store(
            out_ptr
            + tok[:, None] * OUT_S0
            + head[:, None] * OUT_S1
            + d_offs[None, :] * OUT_S2,
            0,
            mask,
        )
        tl.store(
            lse_ptr + tok * LSE_S0 + head * LSE_S1, -float("inf"), tok < end
        )


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    n_tokens, n_heads, hd = out.shape
    if n_tokens:
        if (
            out.is_contiguous()
            and lse.is_contiguous()
            and kv_lens.stride(0) == 1
            and cum_seq_lens.stride(0) == 1
            and hd <= 8192
            and hd & (hd - 1) == 0
        ):
            block_l = max(1, _BLOCK_O // hd)
            _fixup_dense_kernel[(min(_MAX_GRID, kv_lens.numel()),)](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                n_heads,
                hd,
                n_heads * hd,
                kv_lens.numel(),
                block_l,
                num_warps=1,
                num_stages=1,
            )
            return out, lse
        block_d = triton.next_power_of_2(hd)
        block_r = max(1, _BLOCK_O // block_d)
        num_blocks = triton.cdiv(int(max_seq_len) * n_heads, block_r)
        _fixup_rows_kernel[(kv_lens.numel() * num_blocks,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            n_heads,
            hd,
            *out.stride(),
            *lse.stride(),
            kv_lens.stride(0),
            cum_seq_lens.stride(0),
            num_blocks,
            block_r,
            block_d,
            num_warps=4,
            num_stages=1,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
