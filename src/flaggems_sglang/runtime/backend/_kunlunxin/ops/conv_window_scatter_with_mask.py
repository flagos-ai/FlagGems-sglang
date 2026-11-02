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

Version: qoder r28 (2026-10-05) -- byte-identical to r26 (control: kunlunxin)

For every request i with step_indices_raw[i] >= 0:
    dst[:, dst_indices_raw[i], :, :] = src[:, i, step_indices_raw[i], :, :]
`src` is an overlapping as_strided conv window, so its (dim, win) block is addressed by its own
strides; `dst[idx_l, slot]` is a run of DIM*WIN contiguous elements, and that run is what this variant
copies -- one masked-free load and one fully contiguous store per program.

grid = (requests, chunks of the concatenated destination rows).  Padding lanes wrap onto valid
elements with a
remainder, so the body needs no mask, no compare and no `tl.minimum` -- all three are outside what
TritonXPULegalize accepts for a tensor paired with a scalar.  If the destination is not packed the
way the contiguous form assumes, the tile kernel from r16 runs instead; both are Triton.
"""

import triton
import triton.language as tl

OP = "conv_window_scatter_with_mask"


@triton.jit
def _cws_block(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    win_rt: tl.constexpr,
    nrows_rt: tl.constexpr,
    total_rt: tl.constexpr,
    d_stride_l: tl.constexpr,
    d_stride_c: tl.constexpr,
    s_stride_l: tl.constexpr,
    s_stride_r: tl.constexpr,
    s_stride_step: tl.constexpr,
    s_stride_dim: tl.constexpr,
    s_stride_win: tl.constexpr,
    BLK: tl.constexpr,
):
    # The lane space is the concatenation of the request's L destination rows, each of DIM*WIN
    # contiguous elements, cut into BLK-sized chunks.  Wrapping the chunk's flat index through
    # `% total_rt` and `% nrows_rt` keeps every lane inside a row that has to be written -- the
    # padding lanes of the last chunk repeat elements that are copied anyway -- which is what
    # lets the body stay free of masks, compares and `tl.minimum`.
    # r24: the divisors below are compile-time constants on purpose.  Triton folds a
    # constant division into multiply-shift, and `total_rt // win_rt` at runtime is two
    # integer divisions per lane -- the only mechanism left that can explain the 3.5x
    # gap to the 昆仑 leader, since num_warps and num_stages are inert on this backend.
    i = tl.program_id(0)
    step = tl.load(step_idx_ptr + i)
    if step < 0:
        return
    slot = tl.load(dst_idx_ptr + i).to(tl.int32)
    u = tl.program_id(1) * BLK + tl.arange(0, BLK) % total_rt
    uw = u % total_rt
    idx_l = uw // nrows_rt
    tr = uw - idx_l * nrows_rt
    dd = tr // win_rt
    ww = tr - dd * win_rt
    sofs = (
        idx_l * s_stride_l
        + i * s_stride_r
        + step.to(tl.int32) * s_stride_step
        + dd * s_stride_dim
        + ww * s_stride_win
    )
    data = tl.load(src_ptr + sofs)
    tl.store(
        dst_ptr + slot * d_stride_c + idx_l * d_stride_l + tr,
        data.to(dst_ptr.dtype.element_ty),
    )


@triton.jit
def _cws_tile(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    jt_rt: tl.constexpr,
    nrows_rt: tl.constexpr,
    win_rt: tl.constexpr,
    d_stride_l: tl.constexpr,
    d_stride_c: tl.constexpr,
    d_stride_dim: tl.constexpr,
    d_stride_win: tl.constexpr,
    s_stride_l: tl.constexpr,
    s_stride_r: tl.constexpr,
    s_stride_step: tl.constexpr,
    s_stride_dim: tl.constexpr,
    s_stride_win: tl.constexpr,
    LJT: tl.constexpr,
    JT: tl.constexpr,
    WBLK: tl.constexpr,
):
    i = tl.program_id(0)
    step = tl.load(step_idx_ptr + i)
    if step < 0:
        return
    slot = tl.load(dst_idx_ptr + i).to(tl.int32)
    x = tl.arange(0, LJT)
    xr = x % nrows_rt
    idx_l = xr // jt_rt
    d = tl.program_id(1) * JT + xr % jt_rt
    w = tl.arange(0, WBLK) % win_rt
    dofs = (
        idx_l[:, None] * d_stride_l
        + d[:, None] * d_stride_dim
        + w[None, :] * d_stride_win
    )
    sofs = (
        idx_l[:, None] * s_stride_l
        + d[:, None] * s_stride_dim
        + w[None, :] * s_stride_win
    )
    data = tl.load(
        src_ptr + i * s_stride_r + step.to(tl.int32) * s_stride_step + sofs
    )
    tl.store(
        dst_ptr + slot * d_stride_c + dofs, data.to(dst_ptr.dtype.element_ty)
    )


def _pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def _pow2_divisor(x):
    return int(x) & -int(x)


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
    rows = min(requests, int(sstep.shape[0]), int(dslot.shape[0]))
    if rows == 0 or L == 0 or DIM == 0 or WIN == 0:
        return dst

    dsl, dsc = int(dst.stride(2)), int(dst.stride(3))
    if dsl == WIN and dsc == 1:
        # one contiguous run of DIM*WIN per (layer, cache slot): the store needs no gathering
        nrows = DIM * WIN
        total = L * nrows
        blk = min(_pow2(total), 8192)
        nch = (total + blk - 1) // blk
        nw = 8 if blk >= 2048 else (4 if blk >= 256 else 1)
        _cws_block[(rows, nch)](
            dst,
            src,
            dslot,
            sstep,
            WIN,
            nrows,
            total,
            int(dst.stride(0)),
            int(dst.stride(1)),
            int(src.stride(0)),
            int(src.stride(1)),
            int(src.stride(2)),
            int(src.stride(3)),
            int(src.stride(4)),
            BLK=blk,
            num_warps=nw,
        )
        return dst

    wb = _pow2(WIN)
    jt = _pow2_divisor(DIM)
    ljt = _pow2(L * jt)
    while jt > 1 and ljt * wb > 8192:
        jt //= 2
        ljt = _pow2(L * jt)
    nt = DIM // jt
    lanes = ljt * wb
    nw = 8 if lanes >= 2048 else (4 if lanes >= 256 else 1)
    _cws_tile[(rows, nt)](
        dst,
        src,
        dslot,
        sstep,
        jt,
        L * jt,
        WIN,
        int(dst.stride(0)),
        int(dst.stride(1)),
        dsl,
        dsc,
        int(src.stride(0)),
        int(src.stride(1)),
        int(src.stride(2)),
        int(src.stride(3)),
        int(src.stride(4)),
        LJT=ljt,
        JT=jt,
        WBLK=wb,
        num_warps=nw,
    )
    return dst


__all__ = ["conv_window_scatter_with_mask"]
