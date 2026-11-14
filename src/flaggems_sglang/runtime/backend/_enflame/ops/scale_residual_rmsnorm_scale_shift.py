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
def _kernel(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    M: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    RS: tl.constexpr,
    US: tl.constexpr,
    GS: tl.constexpr,
    WS: tl.constexpr,
    SS: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    w = tl.load(W + c * WS, c < D, other=0).to(tl.float32)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        batch = row // L
        seq = row % L
        r = tl.load(
            R + batch * RS[0] + seq * RS[1] + c * RS[2], c < D, other=0
        ).to(tl.float32)
        u = tl.load(
            U + batch * US[0] + seq * US[1] + c * US[2], c < D, other=0
        ).to(tl.float32)
        gate = tl.load(G + batch * GS[0] + c * GS[2], c < D, other=0).to(
            tl.float32
        )
        update = (gate * u).to(O.dtype.element_ty).to(tl.float32)
        res = (r + update).to(O.dtype.element_ty).to(tl.float32)
        tl.store(O + row * D + c, res, c < D, cache_modifier=".cs")
        inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
        norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
        scale = tl.load(S + batch * SS[0] + c * SS[2], c < D, other=0).to(
            tl.float32
        )
        shift = tl.load(H + batch * HS[0] + c * HS[2], c < D, other=0).to(
            tl.float32
        )
        factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
        product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + c, product + shift, c < D, cache_modifier=".cs")


@triton.jit
def _fast(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    L: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // L
    c = tl.arange(0, BLOCK)
    u = tl.load(U + row * D + c, c < D, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    g = tl.load(G + batch * D + c, c < D, other=0).to(tl.float32)
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    p = (g * u).to(O.dtype.element_ty).to(tl.float32)
    r = tl.load(R + row * D + c, c < D, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    res = (r + p).to(O.dtype.element_ty).to(tl.float32)
    tl.store(O + row * D + c, res, c < D, cache_modifier=".cs")
    inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
    norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
    scale = tl.load(S + batch * D + c, c < D, other=0).to(tl.float32)
    shift = tl.load(H + batch * D + c, c < D, other=0).to(tl.float32)
    factor = (1.0 + scale).to(S.dtype.element_ty).to(tl.float32)
    product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * D + c, product + shift, c < D, cache_modifier=".cs")


def _pick_rt(num_batch, seq, hidden, itemsize):
    # Rows per persistent block, from the dtype-dependent tile sweet spot
    # (32768 elements for 2-byte dtypes, 16384 for fp32: at 2048 columns the
    # bf16 program measures 320KB DSM via the compiled llir, inside the
    # 448KB dual-program line; fp32 halves the tile the same way the sibling
    # fused_add_rms_norm recipe does).  RT is shrunk towards 2 so at least
    # 48 blocks exist (one per persistent program slot), and capped at 32
    # rows to stay inside the empirically-validated safe envelope of the
    # 2D-tile broadcast-chain miscompile family.
    tile_elems = 32768 if itemsize <= 2 else 16384
    rt = min(tile_elems // hidden, 32)
    if rt < 2:
        return 0
    while rt > 2 and num_batch * triton.cdiv(seq, rt) < 48:
        rt //= 2
    return rt


@triton.jit
def _persist(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    L,
    BPB,
    D: tl.constexpr,
    EPS: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN: tl.constexpr,
    NS: tl.constexpr,
):
    # Fixed (48, batch) persistent grid: each program strides over the
    # row-blocks of its own batch, so the per-batch gate/scale/shift
    # vectors and (1+scale) factor are hoisted out of the block loop, and
    # the 2-stage software pipeline overlaps the next block's loads with
    # the current block's compute/stores.  Non-persistent grids pay an
    # inter-program drain bubble between row-blocks that this removes.
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    vec = batch * D + cols[None, :]
    w = tl.load(W + cols[None, :]).to(tl.float32)
    g = tl.load(G + vec).to(tl.float32)
    sc = tl.load(S + vec).to(tl.float32)
    sh = tl.load(H + vec).to(tl.float32)
    fac = (1.0 + sc).to(S.dtype.element_ty).to(tl.float32)
    for blk in tl.range(pid, BPB, NPROG, num_stages=NS):
        rows = blk * RT + tl.arange(0, RT)
        row_off = (batch * L + rows)[:, None] * D + cols[None, :]
        if EVEN:
            r = tl.load(R + row_off, cache_modifier=".cg").to(tl.float32)
            u = tl.load(U + row_off, cache_modifier=".cg").to(tl.float32)
            update = (g * u).to(O.dtype.element_ty).to(tl.float32)
            res = (r + update).to(O.dtype.element_ty).to(tl.float32)
            tl.store(O + row_off, res, cache_modifier=".cs")
            inv = tl.rsqrt(tl.sum(res * res, axis=1) / D + EPS)
            norm = (res * inv[:, None] * w).to(Y.dtype.element_ty).to(
                tl.float32
            )
            product = (norm * fac).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(Y + row_off, product + sh, cache_modifier=".cs")
        else:
            rmask = (rows < L)[:, None]
            r = tl.load(
                R + row_off, mask=rmask, other=0, cache_modifier=".cg"
            ).to(tl.float32)
            u = tl.load(
                U + row_off, mask=rmask, other=0, cache_modifier=".cg"
            ).to(tl.float32)
            update = (g * u).to(O.dtype.element_ty).to(tl.float32)
            res = (r + update).to(O.dtype.element_ty).to(tl.float32)
            tl.store(O + row_off, res, mask=rmask, cache_modifier=".cs")
            inv = tl.rsqrt(tl.sum(res * res, axis=1) / D + EPS)
            norm = (res * inv[:, None] * w).to(Y.dtype.element_ty).to(
                tl.float32
            )
            product = (norm * fac).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(
                Y + row_off, product + sh, mask=rmask, cache_modifier=".cs"
            )


@triton.jit
def _rows1d(
    R,
    U,
    G,
    W,
    S,
    H,
    Y,
    O,
    L,
    M,
    D: tl.constexpr,
    EPS: tl.constexpr,
    RROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    w = tl.load(W + c).to(tl.float32)
    num_blocks = (M + RROWS - 1) // RROWS
    for blk in tl.range(pid, num_blocks, NPROG, num_stages=2):
        row0 = blk * RROWS
        for ri in tl.static_range(RROWS):
            row = row0 + ri
            if row < M:
                batch = row // L
                vec = batch * D + c
                r = tl.load(R + row * D + c).to(tl.float32)
                u = tl.load(U + row * D + c).to(tl.float32)
                g = tl.load(G + vec).to(tl.float32)
                update = (g * u).to(O.dtype.element_ty).to(tl.float32)
                res = (r + update).to(O.dtype.element_ty).to(tl.float32)
                tl.store(O + row * D + c, res)
                inv = tl.rsqrt(tl.sum(res * res, 0) / D + EPS)
                norm = (res * inv * w).to(Y.dtype.element_ty).to(tl.float32)
                sc = tl.load(S + vec).to(tl.float32)
                sh = tl.load(H + vec).to(tl.float32)
                factor = (1.0 + sc).to(S.dtype.element_ty).to(tl.float32)
                product = (norm * factor).to(Y.dtype.element_ty).to(tl.float32)
                tl.store(Y + row * D + c, product + sh)


def scale_residual_rmsnorm_scale_shift(
    residual, update, gate, weight, scale, shift, eps
):
    b, s, d = residual.shape
    y = torch.empty_like(residual, memory_format=torch.contiguous_format)
    out = torch.empty_like(y)
    if b * s * d == 0:
        return (y, out)
    block = 1 << (d - 1).bit_length()
    if (
        residual.is_contiguous()
        and update.is_contiguous()
        and gate.is_contiguous()
        and weight.is_contiguous()
        and scale.is_contiguous()
        and shift.is_contiguous()
    ):
        if block == d and b <= 255:
            m = b * s
            if m >= 300:
                rt = _pick_rt(b, s, d, residual.element_size())
                if rt >= 2:
                    _persist[(48, b)](
                        residual,
                        update,
                        gate,
                        weight,
                        scale,
                        shift,
                        y,
                        out,
                        s,
                        triton.cdiv(s, rt),
                        d,
                        float(eps),
                        rt,
                        d,
                        48,
                        (s % rt == 0),
                        2,
                        num_warps=1,
                        num_stages=1,
                        enable_fp_fusion=False,
                    )
                    return (y, out)
            rrows = 2
            nprog = min(48, triton.cdiv(m, rrows))
            nw = 1 if d <= 2048 else 4
            _rows1d[(nprog,)](
                residual,
                update,
                gate,
                weight,
                scale,
                shift,
                y,
                out,
                s,
                m,
                d,
                float(eps),
                rrows,
                d,
                nprog,
                num_warps=nw,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return (y, out)
        _fast.run(
            residual,
            update,
            gate,
            weight,
            scale,
            shift,
            y,
            out,
            s,
            d,
            float(eps),
            block,
            grid=(b * s,),
            warmup=False,
            num_warps=8 if b * s >= 4096 and block <= 2048 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _kernel[min(b * s, 65535),](
            residual,
            update,
            gate,
            weight,
            scale,
            shift,
            y,
            out,
            b * s,
            s,
            d,
            tuple(residual.stride()),
            tuple(update.stride()),
            tuple(gate.stride()),
            weight.stride(0),
            tuple(scale.stride()),
            tuple(shift.stride()),
            float(eps),
            block,
            num_warps=4 if block <= 2048 else 8,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return (y, out)


__all__ = ["scale_residual_rmsnorm_scale_shift"]
