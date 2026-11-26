import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _scaled_x(
    x_ptr,
    dt_ptr,
    dA_ptr,
    off_x,
    off_dt,
    off_dA,
    tt,
    mask_t,
    mask_p,
    p,
    sx1: tl.constexpr,
    sx3,
    s_T,
):
    """x[b, t, h, p] scaled in fp32 by dt*exp(dA[last]-dA[t]) then cast to x.dtype.

    p-major (BLOCK_M, BLOCK_T) so tl.dot consumes it directly: no tl.trans.
    """
    dt_v = tl.load(dt_ptr + off_dt + tt, mask=mask_t, other=0.0).to(tl.float32)
    dA_v = tl.load(dA_ptr + off_dA + tt, mask=mask_t, other=0.0).to(tl.float32)
    sc = dt_v * tl.exp2((s_T - dA_v) * 1.4426950408889634)
    sc = tl.where(mask_t, sc, 0.0)
    xt = tl.load(
        x_ptr + off_x + p[:, None] * sx3 + tt[None, :] * sx1,
        mask=mask_p[:, None] & mask_t[None, :],
        other=0.0,
    )
    return (xt.to(tl.float32) * sc[None, :]).to(xt.dtype)


@triton.autotune(
    configs=[
        # round-9 winner on the hpg=4 workload - regression guard
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 2},
            num_warps=16,
            num_stages=1,
        ),
        # round-9 winner on the hpg=8 workload - regression guard
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 4},
            num_warps=16,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 2},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_T": 128, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 2},
            num_warps=16,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_T": 128, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 4},
            num_warps=16,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 2},
            num_warps=16,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_H": 4},
            num_warps=16,
            num_stages=1,
        ),
    ],
    key=["nheads", "hpg", "chunk_size", "dstate", "headdim"],
)
@triton.jit
def _chunk_state_kernel(
    B_ptr,
    x_ptr,
    dt_ptr,
    dA_ptr,
    out_ptr,
    sB3,
    sx3,
    so4,
    nchunks: tl.constexpr,
    ngroups: tl.constexpr,
    dstate: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    chunk_size: tl.constexpr,
    hpg: tl.constexpr,
    sB0: tl.constexpr,
    sB1: tl.constexpr,
    sB2: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    sdt0: tl.constexpr,
    sdt1: tl.constexpr,
    sdt2: tl.constexpr,
    sda0: tl.constexpr,
    sda1: tl.constexpr,
    sda2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    so2: tl.constexpr,
    so3: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(dstate, BLOCK_N)
    num_m = tl.cdiv(headdim, BLOCK_M)
    num_mn = num_m * num_n
    num_hs = tl.cdiv(hpg, GROUP_H)

    mn = pid % num_mn
    rest = pid // num_mn
    nhs = ngroups * num_hs
    hh = rest % nhs
    r2 = rest // nhs
    c = r2 % nchunks
    b = r2 // nchunks
    g = hh // num_hs
    hs = hh % num_hs

    pid_m = mn // num_n
    pid_n = mn % num_n

    t = tl.arange(0, BLOCK_T)
    p = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_p = p < headdim
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < dstate

    h_base = g * hpg
    hl0 = hs * GROUP_H
    t0 = c * chunk_size

    x_off = b * sx0 + t0 * sx1 + h_base * sx2
    dt_off = b * sdt0 + h_base * sdt1 + c * sdt2
    dA_off = b * sda0 + h_base * sda1 + c * sda2
    B_base = B_ptr + b * sB0 + t0 * sB1 + g * sB2
    out_base = out_ptr + b * so0 + c * so1 + h_base * so2

    hl1c = hl0
    hl2c = hl0
    hl3c = hl0
    sT1 = 0.0
    sT2 = 0.0
    sT3 = 0.0
    if GROUP_H >= 2:
        hl1c = tl.where(hl0 + 1 < hpg, hl0 + 1, hl0)
        sT1 = tl.load(dA_ptr + dA_off + hl1c * sda1 + (chunk_size - 1)).to(
            tl.float32
        )
    if GROUP_H >= 4:
        hl2c = tl.where(hl0 + 2 < hpg, hl0 + 2, hl0)
        sT2 = tl.load(dA_ptr + dA_off + hl2c * sda1 + (chunk_size - 1)).to(
            tl.float32
        )
        hl3c = tl.where(hl0 + 3 < hpg, hl0 + 3, hl0)
        sT3 = tl.load(dA_ptr + dA_off + hl3c * sda1 + (chunk_size - 1)).to(
            tl.float32
        )

    sT0 = tl.load(dA_ptr + dA_off + hl0 * sda1 + (chunk_size - 1)).to(
        tl.float32
    )
    x_off0 = x_off + hl0 * sx2
    dt_off0 = dt_off + hl0 * sdt1
    dA_off0 = dA_off + hl0 * sda1

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_t = tl.cdiv(chunk_size, BLOCK_T)
    for it in range(0, num_t):
        tt = it * BLOCK_T + t
        mask_t = tt < chunk_size

        b_tile = tl.load(
            B_base + tt[:, None] * sB1 + n[None, :] * sB3,
            mask=mask_t[:, None] & mask_n[None, :],
            other=0.0,
        )

        xs = _scaled_x(
            x_ptr,
            dt_ptr,
            dA_ptr,
            x_off0,
            dt_off0,
            dA_off0,
            tt,
            mask_t,
            mask_p,
            p,
            sx1,
            sx3,
            sT0,
        )
        acc0 = tl.dot(xs, b_tile, acc0, out_dtype=tl.float32)

        if GROUP_H >= 2:
            xs = _scaled_x(
                x_ptr,
                dt_ptr,
                dA_ptr,
                x_off + hl1c * sx2,
                dt_off + hl1c * sdt1,
                dA_off + hl1c * sda1,
                tt,
                mask_t,
                mask_p,
                p,
                sx1,
                sx3,
                sT1,
            )
            acc1 = tl.dot(xs, b_tile, acc1, out_dtype=tl.float32)

        if GROUP_H >= 4:
            xs = _scaled_x(
                x_ptr,
                dt_ptr,
                dA_ptr,
                x_off + hl2c * sx2,
                dt_off + hl2c * sdt1,
                dA_off + hl2c * sda1,
                tt,
                mask_t,
                mask_p,
                p,
                sx1,
                sx3,
                sT2,
            )
            acc2 = tl.dot(xs, b_tile, acc2, out_dtype=tl.float32)
            xs = _scaled_x(
                x_ptr,
                dt_ptr,
                dA_ptr,
                x_off + hl3c * sx2,
                dt_off + hl3c * sdt1,
                dA_off + hl3c * sda1,
                tt,
                mask_t,
                mask_p,
                p,
                sx1,
                sx3,
                sT3,
            )
            acc3 = tl.dot(xs, b_tile, acc3, out_dtype=tl.float32)

    store_mask = mask_p[:, None] & mask_n[None, :]
    tl.store(
        out_base + hl0 * so2 + p[:, None] * so3 + n[None, :] * so4,
        acc0,
        mask=store_mask,
    )
    if GROUP_H >= 2:
        tl.store(
            out_base + hl1c * so2 + p[:, None] * so3 + n[None, :] * so4,
            acc1,
            mask=store_mask & (hl1c == hl0 + 1),
        )
    if GROUP_H >= 4:
        tl.store(
            out_base + hl2c * so2 + p[:, None] * so3 + n[None, :] * so4,
            acc2,
            mask=store_mask & (hl2c == hl0 + 2),
        )
        tl.store(
            out_base + hl3c * so2 + p[:, None] * so3 + n[None, :] * so4,
            acc3,
            mask=store_mask & (hl3c == hl0 + 3),
        )


def run(B, x, dt, dA_cumsum):
    batch, seqlen, ngroups, dstate = B.shape
    nheads = x.shape[2]
    headdim = x.shape[3]
    nchunks = dt.shape[2]
    chunk_size = dt.shape[3]
    hpg = nheads // ngroups

    out = torch.empty(
        (batch, nchunks, nheads, headdim, dstate),
        device=x.device,
        dtype=torch.float32,
    )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_M"])
        * triton.cdiv(dstate, META["BLOCK_N"])
        * batch
        * nchunks
        * ngroups
        * triton.cdiv(hpg, META["GROUP_H"]),
    )

    _chunk_state_kernel[grid](
        B,
        x,
        dt,
        dA_cumsum,
        out,
        B.stride(3),
        x.stride(3),
        out.stride(4),
        nchunks=nchunks,
        ngroups=ngroups,
        dstate=dstate,
        nheads=nheads,
        headdim=headdim,
        chunk_size=chunk_size,
        hpg=hpg,
        sB0=B.stride(0),
        sB1=B.stride(1),
        sB2=B.stride(2),
        sx0=x.stride(0),
        sx1=x.stride(1),
        sx2=x.stride(2),
        sdt0=dt.stride(0),
        sdt1=dt.stride(1),
        sdt2=dt.stride(2),
        sda0=dA_cumsum.stride(0),
        sda1=dA_cumsum.stride(1),
        sda2=dA_cumsum.stride(2),
        so0=out.stride(0),
        so1=out.stride(1),
        so2=out.stride(2),
        so3=out.stride(3),
    )
    return out


chunk_state = run
__all__ = ["chunk_state"]
