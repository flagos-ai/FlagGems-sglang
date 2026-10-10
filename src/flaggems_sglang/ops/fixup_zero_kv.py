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


@triton.jit
def _fixup_zero_kv_flat_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_seq_lens_ptr,
    num_heads,
    batch_size,
    hd: tl.constexpr,
    R: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    if hd == BLOCK_HD:
        z = tl.zeros([BLOCK_HD], dtype=out_ptr.dtype.element_ty)
        zi = tl.full([BLOCK_L], float("-inf"), dtype=lse_ptr.dtype.element_ty)
        dr = tl.arange(0, BLOCK_HD)
        lr = tl.arange(0, BLOCK_L)
        for j in tl.static_range(R):
            i = pid * R + j
            if i < batch_size:
                kv_len = tl.load(kv_lens_ptr + i)
                if kv_len == 0:
                    beg = tl.load(cum_seq_lens_ptr + i)
                    end = tl.load(cum_seq_lens_ptr + i + 1)
                    nt = end - beg
                    nl = nt * num_heads
                    tl.store(lse_ptr + beg * num_heads + lr, zi, mask=lr < nl)
                    nl_done = BLOCK_L
                    while nl_done < nl:
                        tl.store(
                            lse_ptr + beg * num_heads + nl_done + lr,
                            zi,
                            mask=lr < (nl - nl_done),
                        )
                        nl_done += BLOCK_L
                    for t0 in range(0, nt, 1):
                        base = tl.multiple_of((beg + t0) * hd, hd)
                        tl.store(out_ptr + base + dr, z)
    else:
        hd_mask = tl.arange(0, BLOCK_HD) < hd
        for j in tl.static_range(R):
            i = pid * R + j
            if i < batch_size:
                kv_len = tl.load(kv_lens_ptr + i)
                if kv_len == 0:
                    beg = tl.load(cum_seq_lens_ptr + i)
                    end = tl.load(cum_seq_lens_ptr + i + 1)
                    nt = end - beg
                    nl = nt * num_heads
                    l_mask = tl.arange(0, BLOCK_L)
                    tl.store(
                        lse_ptr + beg * num_heads + l_mask,
                        tl.full(
                            [BLOCK_L],
                            float("-inf"),
                            dtype=lse_ptr.dtype.element_ty,
                        ),
                        mask=l_mask < nl,
                    )
                    nl_done = BLOCK_L
                    while nl_done < nl:
                        l_tail = nl_done + tl.arange(0, BLOCK_L)
                        tl.store(
                            lse_ptr + beg * num_heads + l_tail,
                            tl.full(
                                [BLOCK_L],
                                float("-inf"),
                                dtype=lse_ptr.dtype.element_ty,
                            ),
                            mask=tl.arange(0, BLOCK_L) < (nl - nl_done),
                        )
                        nl_done += BLOCK_L
                    for t0 in range(0, nt, 1):
                        out_offs = (beg + t0) * hd + tl.arange(0, BLOCK_HD)
                        tl.store(
                            out_ptr + out_offs,
                            tl.zeros(
                                [BLOCK_HD], dtype=out_ptr.dtype.element_ty
                            ),
                            mask=hd_mask,
                        )


@triton.jit
def _fixup_zero_kv_flat_small_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_seq_lens_ptr,
    hd,
    num_heads,
    max_seq_len,
    BLOCK_HD: tl.constexpr,
    MAXT: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    i = tl.program_id(0)

    kv_len = tl.load(kv_lens_ptr + i)
    if kv_len == 0:
        beg = tl.load(cum_seq_lens_ptr + i)
        end = tl.load(cum_seq_lens_ptr + i + 1)
        nt = end - beg

        t = tl.arange(0, MAXT)
        m = t < nt
        d_mask = tl.arange(0, BLOCK_HD) < hd
        out_offs = (beg + t)[:, None] * hd + tl.arange(0, BLOCK_HD)[None, :]
        tl.store(
            out_ptr + out_offs,
            tl.zeros([MAXT, BLOCK_HD], dtype=out_ptr.dtype.element_ty),
            mask=m[:, None] & d_mask[None, :],
        )
        nl = nt * num_heads
        l_mask = tl.arange(0, BLOCK_L)
        lse_offs = beg * num_heads + l_mask
        tl.store(
            lse_ptr + lse_offs,
            tl.full([BLOCK_L], float("-inf"), dtype=lse_ptr.dtype.element_ty),
            mask=l_mask < nl,
        )
        for t0 in range(MAXT, nt):
            tail_offs = (beg + t0) * hd + tl.arange(0, BLOCK_HD)
            tl.store(
                out_ptr + tail_offs,
                tl.zeros([BLOCK_HD], dtype=out_ptr.dtype.element_ty),
                mask=d_mask,
            )
            tail_l_offs = (beg + t0) * num_heads + tl.arange(0, BLOCK_L)
            tl.store(
                lse_ptr + tail_l_offs,
                tl.full(
                    [BLOCK_L], float("-inf"), dtype=lse_ptr.dtype.element_ty
                ),
                mask=tl.arange(0, BLOCK_L) < num_heads,
            )


@triton.jit
def _fixup_zero_kv_strided_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_seq_lens_ptr,
    num_heads,
    v_head_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lt,
    stride_lh,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    i = tl.program_id(0)

    kv_len = tl.load(kv_lens_ptr + i)
    if kv_len != 0:
        return

    beg = tl.load(cum_seq_lens_ptr + i)
    end = tl.load(cum_seq_lens_ptr + i + 1)
    nt = end - beg

    d_mask = tl.arange(0, BLOCK_D) < v_head_dim
    h_mask = tl.arange(0, BLOCK_H) < num_heads
    for t0 in range(0, nt, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tmask = t < nt
        base_t = beg + t
        for h in range(0, num_heads):
            out_offs = (
                base_t[:, None] * stride_ot
                + h * stride_oh
                + tl.arange(0, BLOCK_D)[None, :] * stride_od
            )
            out_val = tl.zeros(
                [BLOCK_T, BLOCK_D], dtype=out_ptr.dtype.element_ty
            )
            tl.store(
                out_ptr + out_offs,
                out_val,
                mask=tmask[:, None] & d_mask[None, :],
            )
        lse_offs = (
            base_t[:, None] * stride_lt
            + tl.arange(0, BLOCK_H)[None, :] * stride_lh
        )
        lse_val = tl.full(
            [BLOCK_T, BLOCK_H], float("-inf"), dtype=lse_ptr.dtype.element_ty
        )
        tl.store(
            lse_ptr + lse_offs, lse_val, mask=tmask[:, None] & h_mask[None, :]
        )


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    batch_size = kv_lens.shape[0]
    num_heads, v_head_dim = out.shape[1], out.shape[2]

    if batch_size == 0:
        return out, lse

    if out.is_contiguous() and lse.is_contiguous():
        hd = num_heads * v_head_dim
        if batch_size >= 1024:
            _fixup_zero_kv_flat_kernel[(triton.cdiv(batch_size, 2),)](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                num_heads,
                batch_size,
                hd,
                R=2,
                BLOCK_HD=triton.next_power_of_2(hd),
                BLOCK_L=min(
                    256,
                    triton.next_power_of_2(
                        max(1, int(max_seq_len))
                        * triton.next_power_of_2(num_heads)
                    ),
                ),
                num_warps=4,
            )
        else:
            _fixup_zero_kv_flat_small_kernel[(batch_size,)](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                hd,
                num_heads,
                max_seq_len,
                BLOCK_HD=triton.next_power_of_2(hd),
                MAXT=triton.next_power_of_2(max(1, int(max_seq_len))),
                BLOCK_L=triton.next_power_of_2(
                    triton.next_power_of_2(max_seq_len if max_seq_len else 1)
                    * num_heads
                ),
                num_warps=4,
            )
    else:
        _fixup_zero_kv_strided_kernel[(batch_size,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            num_heads,
            v_head_dim,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_T=4,
            BLOCK_D=triton.next_power_of_2(v_head_dim),
            BLOCK_H=triton.next_power_of_2(num_heads),
            num_warps=4,
        )

    return out, lse


__all__ = ["fixup_zero_kv"]
