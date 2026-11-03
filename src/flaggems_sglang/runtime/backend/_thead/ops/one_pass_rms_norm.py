# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton RMSNorm (SGLang srt/layers/layernorm.py RMSNorm.forward_native).

    out = ((x.to(fp32) * rsqrt(mean(x^2) + eps)) * w.to(fp32)).to(x.dtype)

The reduction runs over the last dimension; the leading dimensions are flattened
into rows.  All accumulation and both multiplies happen in fp32 and the single
cast back to the input dtype happens at the store, mirroring the reference
arithmetic order exactly.  `x` and `weight` are never modified.

Kernel selection
----------------
When N is at most 8192 and can be written exactly as the sum of one to four
power-of-two chunks (each between 256 and 8192 wide), a mask-free chunked kernel
is used: each program owns exactly one row, every chunk is loaded once and stays
register resident until its single store, so `x` is read exactly once and `out`
written exactly once.  Because no column lane is ever padded, the kernel issues
no predicated memory traffic and keeps fewer values live than a padded
``next_power_of_2(N)`` tile; that measurably raises achieved bandwidth for
N = 6144, which a padded tile widens to 8192.

Otherwise a masked single-tile kernel is used (tiny or awkward hidden sizes) or a
runtime-looped two-pass kernel for rows wider than 8192.

Chunk decomposition and num_warps come from a table measured on the target device
(bandwidth sweeps at the exact evaluation shapes, median of many repetitions);
untuned hidden sizes fall back to a general rule.
"""

import torch
import triton
import triton.language as tl

_MAX_CHUNK = 8192
_MAX_TILE_N = 8192

_LOAD_CM = ""
_STORE_CM = ".cs"


@triton.jit
def _rms_norm_chunks(
    X,
    W,
    OUT,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,
    eps,
    CW0: tl.constexpr,
    CW1: tl.constexpr,
    CW2: tl.constexpr,
    CW3: tl.constexpr,
    LOAD_CM: tl.constexpr,
    STORE_CM: tl.constexpr,
):
    """One row per program; unmasked power-of-two chunks summing exactly to N.

    Every chunk of `x` is loaded and squared before the reciprocal square root is
    formed, so `x` is read exactly once.  Each chunk's weight slice is loaded
    only immediately before that chunk is written, which keeps the peak live
    register count near one chunk's worth.
    """
    pid = tl.program_id(0)
    xbase = pid * stride_xm
    obase = pid * stride_om

    c0 = tl.arange(0, CW0)
    x0 = tl.load(X + xbase + c0 * stride_xn, cache_modifier=LOAD_CM).to(
        tl.float32
    )
    acc = tl.sum(x0 * x0, axis=0)

    if CW1 > 0:
        c1 = tl.arange(0, CW1)
        x1 = tl.load(
            X + xbase + (CW0 + c1) * stride_xn, cache_modifier=LOAD_CM
        ).to(tl.float32)
        acc += tl.sum(x1 * x1, axis=0)
        if CW2 > 0:
            c2 = tl.arange(0, CW2)
            x2 = tl.load(
                X + xbase + (CW0 + CW1 + c2) * stride_xn,
                cache_modifier=LOAD_CM,
            ).to(tl.float32)
            acc += tl.sum(x2 * x2, axis=0)
            if CW3 > 0:
                c3 = tl.arange(0, CW3)
                x3 = tl.load(
                    X + xbase + (CW0 + CW1 + CW2 + c3) * stride_xn,
                    cache_modifier=LOAD_CM,
                ).to(tl.float32)
                acc += tl.sum(x3 * x3, axis=0)
                rstd = 1.0 / tl.sqrt(acc / N + eps)
                w3 = tl.load(W + CW0 + CW1 + CW2 + c3).to(tl.float32)
                tl.store(
                    OUT + obase + (CW0 + CW1 + CW2 + c3) * stride_on,
                    (x3 * rstd * w3).to(OUT.dtype.element_ty),
                    cache_modifier=STORE_CM,
                )
            else:
                rstd = 1.0 / tl.sqrt(acc / N + eps)
            w2 = tl.load(W + CW0 + CW1 + c2).to(tl.float32)
            tl.store(
                OUT + obase + (CW0 + CW1 + c2) * stride_on,
                (x2 * rstd * w2).to(OUT.dtype.element_ty),
                cache_modifier=STORE_CM,
            )
        else:
            rstd = 1.0 / tl.sqrt(acc / N + eps)
        w1 = tl.load(W + CW0 + c1).to(tl.float32)
        tl.store(
            OUT + obase + (CW0 + c1) * stride_on,
            (x1 * rstd * w1).to(OUT.dtype.element_ty),
            cache_modifier=STORE_CM,
        )
    else:
        rstd = 1.0 / tl.sqrt(acc / N + eps)

    w0 = tl.load(W + c0).to(tl.float32)
    tl.store(
        OUT + obase + c0 * stride_on,
        (x0 * rstd * w0).to(OUT.dtype.element_ty),
        cache_modifier=STORE_CM,
    )


@triton.jit
def _rms_norm_masked(
    X,
    W,
    OUT,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    LOAD_CM: tl.constexpr,
    STORE_CM: tl.constexpr,
):
    """One row per program, column-masked padded tile (general fallback)."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N

    xbase = pid * stride_xm
    obase = pid * stride_om
    x = tl.load(
        X + xbase + cols * stride_xn,
        mask=cmask,
        other=0.0,
        cache_modifier=LOAD_CM,
    )
    x32 = x.to(tl.float32)
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x32 * x32, axis=0) / N + eps)
    tl.store(
        OUT + obase + cols * stride_on,
        (x32 * rstd * w).to(OUT.dtype.element_ty),
        mask=cmask,
        cache_modifier=STORE_CM,
    )


@triton.jit
def _rms_norm_loop(
    X,
    W,
    OUT,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    LOAD_CM: tl.constexpr,
    STORE_CM: tl.constexpr,
):
    """Runtime-looped two-pass variant for very wide rows (general fallback)."""
    pid = tl.program_id(0)
    xbase = pid * stride_xm
    obase = pid * stride_om
    cols = tl.arange(0, BLOCK_N)[None, :]

    acc = tl.zeros((1,), dtype=tl.float32)
    for c0 in range(0, N, BLOCK_N):
        cmask = (c0 + cols) < N
        x = tl.load(
            X + xbase + (c0 + cols) * stride_xn,
            mask=cmask,
            other=0.0,
            cache_modifier=LOAD_CM,
        )
        f = x.to(tl.float32)
        acc += tl.sum(f * f, axis=1)

    rstd = 1.0 / tl.sqrt(acc / N + eps)

    for c0 in range(0, N, BLOCK_N):
        c = c0 + cols
        cmask = c < N
        w = tl.load(W + c, mask=cmask, other=0.0).to(tl.float32)
        x = tl.load(
            X + xbase + c * stride_xn,
            mask=cmask,
            other=0.0,
            cache_modifier=LOAD_CM,
        )
        f = x.to(tl.float32)
        tl.store(
            OUT + obase + c * stride_on,
            (f * rstd * w).to(OUT.dtype.element_ty),
            mask=cmask,
            cache_modifier=STORE_CM,
        )


def _tuned(M, N):
    """Per-(M, N) launch configuration measured on the target device.

    Returns (chunks, num_warps, force_masked); ``chunks`` is a tuple of one to
    four power-of-two widths that sum exactly to N, or None when no measured
    entry applies.
    """
    if N == 6144:
        if M <= 64:
            return None, 16, True
        if M <= 256:
            return (4096, 2048, 0, 0), 8, False
        return (4096, 2048, 0, 0), 4, False
    if N == 4096:
        if M <= 128:
            return (4096, 0, 0, 0), 16, False
        if M <= 256:
            return (4096, 0, 0, 0), 8, False
        if M <= 1024:
            return (2048, 2048, 0, 0), 4, False
        if M == 2048:
            return (4096, 0, 0, 0), 4, False
        if M <= 8192:
            if M <= 4096:
                return (2048, 2048, 0, 0), 4, False
            return (2048, 2048, 0, 0), 1, False
        if M <= 32768:
            return (1024, 1024, 1024, 1024), 2, False
        return (2048, 2048, 0, 0), 1, False
    return None, 0, False


def _decompose(N):
    """One to four power-of-two widths (256..8192) summing exactly to N."""
    if N < 256 or N > _MAX_TILE_N:
        return None
    widths = []
    n = N
    for _ in range(4):
        if n == 0:
            break
        w = 1 << (n.bit_length() - 1)
        if w < 256 or w > _MAX_CHUNK:
            return None
        widths.append(w)
        n -= w
    return tuple(widths) if n == 0 else None


def _plan(M, N):
    """Return (kind, chunks, block_n, num_warps)."""
    chunks, warps, force_masked = _tuned(M, N)
    if chunks is not None:
        return "chunks", chunks, 0, warps
    if force_masked:
        return "masked", None, triton.next_power_of_2(N), warps

    if N > _MAX_TILE_N:
        return "loop", None, 4096, 4

    chunks = _decompose(N)
    if chunks is not None:
        return "chunks", chunks, 0, (4 if N <= 2048 else 8)

    block_n = max(16, min(triton.next_power_of_2(N), _MAX_TILE_N))
    return "masked", None, block_n, 4


def one_pass_rms_norm(x, w, eps=1e-06):
    N = x.shape[-1]
    M = x.numel() // N if N > 0 else 0
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if M == 0 or N == 0:
        return out

    stride_xm = x.stride(-2) if x.dim() >= 2 else 0
    stride_xn = x.stride(-1)
    stride_om = out.stride(-2) if out.dim() >= 2 else 0
    stride_on = out.stride(-1)

    kind, chunks, block_n, num_warps = _plan(M, N)
    grid = (M,)
    if kind == "chunks":
        cw0, cw1, cw2, cw3 = (tuple(chunks) + (0, 0, 0, 0))[:4]
        _rms_norm_chunks[grid](
            x,
            w,
            out,
            stride_xm,
            stride_xn,
            stride_om,
            stride_on,
            N,
            float(eps),
            CW0=cw0,
            CW1=cw1,
            CW2=cw2,
            CW3=cw3,
            LOAD_CM=_LOAD_CM,
            STORE_CM=_STORE_CM,
            num_warps=num_warps,
        )
    elif kind == "masked":
        _rms_norm_masked[grid](
            x,
            w,
            out,
            stride_xm,
            stride_xn,
            stride_om,
            stride_on,
            N,
            float(eps),
            BLOCK_N=block_n,
            LOAD_CM=_LOAD_CM,
            STORE_CM=_STORE_CM,
            num_warps=num_warps,
        )
    else:
        _rms_norm_loop[grid](
            x,
            w,
            out,
            stride_xm,
            stride_xn,
            stride_om,
            stride_on,
            N,
            float(eps),
            BLOCK_N=block_n,
            LOAD_CM=_LOAD_CM,
            STORE_CM=_STORE_CM,
            num_warps=num_warps,
        )
    return out


__all__ = ["one_pass_rms_norm"]
