"""Chunk-local vector cumulative sum over [B, T, H, S] along the time axis.

Per (b, h, s) column and per contiguous chunk of ``chunk_size`` time steps, an
inclusive prefix (forward) or suffix (reverse) cumulative sum is produced in
fp32 and written into a freshly allocated contiguous fp32 tensor.

Fast path (flat-inner decomposition): when the (H, S) plane is contiguous the two
axes are treated as one flat inner axis P = H*S.  Each program owns a block of BS
flat columns and walks the chunk row by row, so every thread carries its own fp32
accumulator for its own columns and the scan never leaves registers (no shared
memory, no cross-lane communication).  Reverse is expressed purely as flipped row
addressing, so both directions run the identical forward scan.

Fallback (per-head decomposition): used only when the (H, S) plane is not
contiguous; identical math with explicit per-axis strides.
"""

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BS": 128}, num_warps=1),
    triton.Config({"BS": 256}, num_warps=1),
    triton.Config({"BS": 256}, num_warps=2),
    triton.Config({"BS": 512}, num_warps=2),
    triton.Config({"BS": 512}, num_warps=4),
    triton.Config({"BS": 1024}, num_warps=8),
]


@triton.autotune(configs=_CONFIGS, key=["T", "P"])
@triton.jit
def _chunk_cumsum_flat(
    g_ptr,
    o_ptr,
    T,
    P,
    stride_gb,
    stride_gt,
    stride_ob,
    stride_ot,
    scale,
    chunk_size: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)
    i_f = tl.program_id(2)

    t0 = i_t * chunk_size
    n_valid = tl.minimum(chunk_size, T - t0)

    cols = i_f * BS + tl.arange(0, BS)
    cmask = cols < P

    g_base = g_ptr + i_b * stride_gb + cols
    o_base = o_ptr + i_b * stride_ob + cols

    acc = tl.zeros([BS], tl.float32)

    # Eight independent row loads are issued before any of them is consumed, so a
    # warp keeps eight global loads in flight instead of draining after each one.
    # The fp32 accumulation order is still strictly row by row.
    for i0 in tl.static_range(0, BT, 8):
        if REVERSE:
            r0 = t0 + (n_valid - 1 - i0)
            r1 = t0 + (n_valid - 2 - i0)
            r2 = t0 + (n_valid - 3 - i0)
            r3 = t0 + (n_valid - 4 - i0)
            r4 = t0 + (n_valid - 5 - i0)
            r5 = t0 + (n_valid - 6 - i0)
            r6 = t0 + (n_valid - 7 - i0)
            r7 = t0 + (n_valid - 8 - i0)
        else:
            r0 = t0 + i0
            r1 = t0 + i0 + 1
            r2 = t0 + i0 + 2
            r3 = t0 + i0 + 3
            r4 = t0 + i0 + 4
            r5 = t0 + i0 + 5
            r6 = t0 + i0 + 6
            r7 = t0 + i0 + 7
        m0 = cmask & (i0 < n_valid)
        m1 = cmask & (i0 + 1 < n_valid)
        m2 = cmask & (i0 + 2 < n_valid)
        m3 = cmask & (i0 + 3 < n_valid)
        m4 = cmask & (i0 + 4 < n_valid)
        m5 = cmask & (i0 + 5 < n_valid)
        m6 = cmask & (i0 + 6 < n_valid)
        m7 = cmask & (i0 + 7 < n_valid)
        x0 = tl.load(g_base + r0 * stride_gt, mask=m0, other=0.0)
        x1 = tl.load(g_base + r1 * stride_gt, mask=m1, other=0.0)
        x2 = tl.load(g_base + r2 * stride_gt, mask=m2, other=0.0)
        x3 = tl.load(g_base + r3 * stride_gt, mask=m3, other=0.0)
        x4 = tl.load(g_base + r4 * stride_gt, mask=m4, other=0.0)
        x5 = tl.load(g_base + r5 * stride_gt, mask=m5, other=0.0)
        x6 = tl.load(g_base + r6 * stride_gt, mask=m6, other=0.0)
        x7 = tl.load(g_base + r7 * stride_gt, mask=m7, other=0.0)

        acc += x0.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r0 * stride_ot, acc * scale, mask=m0)
        else:
            tl.store(o_base + r0 * stride_ot, acc, mask=m0)
        acc += x1.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r1 * stride_ot, acc * scale, mask=m1)
        else:
            tl.store(o_base + r1 * stride_ot, acc, mask=m1)
        acc += x2.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r2 * stride_ot, acc * scale, mask=m2)
        else:
            tl.store(o_base + r2 * stride_ot, acc, mask=m2)
        acc += x3.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r3 * stride_ot, acc * scale, mask=m3)
        else:
            tl.store(o_base + r3 * stride_ot, acc, mask=m3)
        acc += x4.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r4 * stride_ot, acc * scale, mask=m4)
        else:
            tl.store(o_base + r4 * stride_ot, acc, mask=m4)
        acc += x5.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r5 * stride_ot, acc * scale, mask=m5)
        else:
            tl.store(o_base + r5 * stride_ot, acc, mask=m5)
        acc += x6.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r6 * stride_ot, acc * scale, mask=m6)
        else:
            tl.store(o_base + r6 * stride_ot, acc, mask=m6)
        acc += x7.to(tl.float32)
        if HAS_SCALE:
            tl.store(o_base + r7 * stride_ot, acc * scale, mask=m7)
        else:
            tl.store(o_base + r7 * stride_ot, acc, mask=m7)


@triton.jit
def _chunk_cumsum_head(
    g_ptr,
    o_ptr,
    T,
    S,
    NH,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_ob,
    stride_ot,
    stride_oh,
    scale,
    chunk_size: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_s = tl.program_id(2)

    i_b = i_bh // NH
    i_h = i_bh % NH

    rows = tl.arange(0, BT)
    cols = i_s * BS + tl.arange(0, BS)

    t0 = i_t * chunk_size
    n_valid = tl.minimum(chunk_size, T - t0)
    if REVERSE:
        row_off = t0 + (n_valid - 1 - rows)
    else:
        row_off = t0 + rows

    rmask = rows < n_valid
    cmask = cols < S
    mask = rmask[:, None] & cmask[None, :]

    g_base = g_ptr + i_b * stride_gb + i_h * stride_gh
    g_ptrs = g_base + row_off[:, None] * stride_gt + cols[None, :]
    b_x = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)

    b_o = tl.cumsum(b_x, axis=0)

    if HAS_SCALE:
        b_o = b_o * scale

    o_base = o_ptr + i_b * stride_ob + i_h * stride_oh
    o_ptrs = o_base + row_off[:, None] * stride_ot + cols[None, :]
    tl.store(o_ptrs, b_o, mask=mask)


def run(g, chunk_size, reverse=False, scale=None):
    B, T, H, S = g.shape

    if scale is None:
        has_scale = False
        scale_f = 1.0
    else:
        if isinstance(scale, torch.Tensor):
            scale_f = float(scale.item())
        else:
            scale_f = float(scale)
        has_scale = True

    if isinstance(chunk_size, torch.Tensor):
        cs = int(chunk_size.item())
    else:
        cs = int(chunk_size)

    rev = bool(reverse)

    out = torch.empty((B, T, H, S), dtype=torch.float32, device=g.device)

    if g.numel() == 0 or cs <= 0 or T == 0:
        return out

    NT = triton.cdiv(T, cs)

    stride_gb = g.stride(0)
    stride_gt = g.stride(1)
    stride_gh = g.stride(2)
    stride_gs = g.stride(3)

    stride_ob = out.stride(0)
    stride_ot = out.stride(1)
    stride_oh = out.stride(2)

    flat_ok = (stride_gs == 1) and (stride_gh == S)

    if flat_ok:
        P = H * S
        # Rounded up to a multiple of the eight-row load batch; rows past the true
        # valid length are masked off, so the padding is an exact no-op.
        BT = ((min(cs, T) + 7) // 8) * 8
        grid = lambda meta: (NT, B, triton.cdiv(P, meta["BS"]))
        _chunk_cumsum_flat[grid](
            g,
            out,
            T,
            P,
            stride_gb,
            stride_gt,
            stride_ob,
            stride_ot,
            scale_f,
            chunk_size=cs,
            BT=BT,
            REVERSE=rev,
            HAS_SCALE=has_scale,
        )
    else:
        BT = triton.next_power_of_2(min(cs, T))
        BS = min(128, triton.next_power_of_2(S), max(16, 4096 // BT))
        grid = (NT, B * H, triton.cdiv(S, BS))
        _chunk_cumsum_head[grid](
            g,
            out,
            T,
            S,
            H,
            stride_gb,
            stride_gt,
            stride_gh,
            stride_ob,
            stride_ot,
            stride_oh,
            scale_f,
            chunk_size=cs,
            BT=BT,
            BS=BS,
            REVERSE=rev,
            HAS_SCALE=has_scale,
            num_warps=4,
        )

    return out


chunk_local_cumsum_vector = run

__all__ = ["chunk_local_cumsum_vector"]
