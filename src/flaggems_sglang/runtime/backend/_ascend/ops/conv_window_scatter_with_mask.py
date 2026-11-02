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

"""conv_window_scatter_with_mask -- Triton kernel (FlagOS S2 Track-1, Batch 8, Task 111).

Version: qoder r2 (2026-10-02).

For every request i with step_indices_raw[i] >= 0:
    dst[:, dst_indices_raw[i], :, :] = src[:, i, step_indices_raw[i], :, :]
`src` is an overlapping as_strided conv window, so the (dim, win) block is addressed by its
own strides and adjacent draft steps share elements.  Fills `dst` in place and returns it.
"""

import triton
import triton.language as tl

OP = "conv_window_scatter_with_mask"


@triton.jit
def _cws_rows(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    d_stride_l: tl.constexpr,
    d_stride_c: tl.constexpr,
    d_stride_dim: tl.constexpr,
    d_stride_win: tl.constexpr,
    s_stride_l: tl.constexpr,
    s_stride_r: tl.constexpr,
    s_stride_step: tl.constexpr,
    s_stride_dim: tl.constexpr,
    s_stride_win: tl.constexpr,
    L: tl.constexpr,
    DIM: tl.constexpr,
    WIN: tl.constexpr,
    DBLK: tl.constexpr,
    WBLK: tl.constexpr,
):
    i = tl.program_id(0)
    step = tl.load(step_idx_ptr + i)
    if step < 0:
        return
    slot = tl.load(dst_idx_ptr + i).to(tl.int32)
    w = tl.arange(0, WBLK)
    mw = w < WIN
    # The (dim, win) tile is walked in DBLK-sized chunks of the dim axis, so a geometry whose
    # dim*win exceeds one tile is covered completely rather than silently truncated.
    CHUNKS: tl.constexpr = (DIM + DBLK - 1) // DBLK
    for dc in tl.static_range(CHUNKS):
        d = dc * DBLK + tl.arange(0, DBLK)
        m = (d[:, None] < DIM) & mw[None, :]
        dofs = d[:, None] * d_stride_dim + w[None, :] * d_stride_win
        sofs = (
            step.to(tl.int32) * s_stride_step
            + d[:, None] * s_stride_dim
            + w[None, :] * s_stride_win
        )
        for idx_l in tl.static_range(L):
            data = tl.load(
                src_ptr + idx_l * s_stride_l + i * s_stride_r + sofs, mask=m
            )
            tl.store(
                dst_ptr + idx_l * d_stride_l + slot * d_stride_c + dofs,
                data.to(dst_ptr.dtype.element_ty),
                mask=m,
            )


def _pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    dslot, sstep = dst_indices_raw, step_indices_raw
    if int(dst.dim()) != 4 or int(src.dim()) != 5:
        raise RuntimeError(
            f"{OP}: expected dst[layers,cache,dim,K-1] and "
            f"src[layers,requests,draft,dim,K-1], got {tuple(dst.shape)}"
            f"/{tuple(src.shape)}"
        )
    L, _, DIM, WIN = (
        int(dst.shape[0]),
        int(dst.shape[1]),
        int(dst.shape[2]),
        int(dst.shape[3]),
    )
    requests = int(src.shape[1])
    n_idx = int(sstep.shape[0])
    rows = min(requests, n_idx, int(dslot.shape[0]))
    if rows == 0 or L == 0 or DIM == 0 or WIN == 0:
        return dst

    wb = min(_pow2(WIN), 8192)
    db = _pow2(DIM)
    while db * wb > 8192 and db > 1:
        db //= 2
    nw = 8 if db * wb >= 2048 else (4 if db * wb >= 256 else 1)

    _cws_rows[(rows,)](
        dst,
        src,
        dslot,
        sstep,
        d_stride_l=int(dst.stride(0)),
        d_stride_c=int(dst.stride(1)),
        d_stride_dim=int(dst.stride(2)),
        d_stride_win=int(dst.stride(3)),
        s_stride_l=int(src.stride(0)),
        s_stride_r=int(src.stride(1)),
        s_stride_step=int(src.stride(2)),
        s_stride_dim=int(src.stride(3)),
        s_stride_win=int(src.stride(4)),
        L=L,
        DIM=DIM,
        WIN=WIN,
        DBLK=db,
        WBLK=wb,
        num_warps=nw,
    )
    return dst


__all__ = ["conv_window_scatter_with_mask"]
