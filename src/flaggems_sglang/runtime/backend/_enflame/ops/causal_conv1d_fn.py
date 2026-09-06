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

"""Causal 1-D convolution over continuous-batched sequences (Enflame GCU
/ NVIDIA portable).

    out[d, start + t] = silu(
        sum_{k=0}^{WIDTH-1} weight[d, k]
            * x[d, start + t - (WIDTH-1) + k]
        + bias[d]
    )

where ``start``/``end`` come from ``query_start_loc`` (a per-sequence
prefix-sum into a packed ``x`` of shape ``(dim, total_t)``), and inputs
before ``start`` are zero-padded.

The two things that decide performance on the GCU backend
---------------------------------------------------------
Profiling on the Enflame GCU revealed two hard constraints that dominate
this kernel:

1. **Any offset in the *index* of a load/store destroys vectorization.**
   A load whose address is
   ``x_ptr + offs_d[:, None] * TOTAL_T + (offs_t - SHIFT)[None, :]`` (the
   shift baked into the index tensor ``offs_t - SHIFT``) runs ~100x
   slower than the same load with a canonical ``arange`` index.  The GCU
   vectorizer only recognizes the canonical tile form
   ``scalar_base + outer * stride + arange(BLOCK)``.

   Fix: keep the index tensor canonical and move the shift into the
   **scalar base pointer** --
   ``tl.load((x_ptr - SHIFT) + offs_d[:, None] * TOTAL_T +
   offs_t[None, :])``.  ``x_ptr - SHIFT`` is a scalar pointer, so the
   index stays an aligned ``arange`` and the load vectorizes (measured:
   ~0.5ms vs ~47ms for a 21M-element tile).

2. **A load whose *mask* depends on a loaded value is just as slow.**
   Masking the causal boundary directly in the load
   (``tl.load(..., mask=src >= seg_start)``) also collapses to ~180ms.
   Only compile-time masks (affine in ``pid`` / ``arange``) vectorize.

   Fix: keep the load mask compile-time (only the *global* left/right
   bounds), and apply the *segment* boundary as a plain elementwise
   ``tl.where(ok, v, 0.0)`` after the load.  Elementwise selects are
   cheap; only the memory-access mask is expensive.

So each of the ``WIDTH`` taps is one vectorized pointer-shifted load
with a compile-time mask, followed by an elementwise zero for the
segment-boundary lanes, then a fused multiply-accumulate.  ``acc`` lives
in float32 and the (optional) SiLU is fused before the single store.

Grid / tiling
-------------
* ``grid.x = cdiv(total_t, BLOCK_T)`` (packed time, dense regardless of
  how the batch is split) and ``grid.y = cdiv(dim, BLOCK_D)`` (channel
  blocks, kept <= 255).
* ``BLOCK_T``/``BLOCK_D`` are powers of two chosen per-shape; the GCU
  strongly prefers a few large tiles over many small ones (a 1024x128
  tile with 2 warps measured fastest for the dim=5120 / width=4
  benchmark).  ``num_warps`` drops to 2 for small tiles to avoid the
  ~1.5MB local-memory cap.

Segment-start index (host pre-processing)
-----------------------------------------
The per-position ``seg_start`` vector is built on-device with a tiny
Triton kernel (``build_seg_start_kernel``) instead of
``torch.searchsorted``.  ``searchsorted`` is surprisingly expensive on
the GCU (~1.1ms for a single 4096-length sequence, ~350us for 256
segments), so building the same index with a branchless ``max``-scan
over the segment boundaries is 1.5-45x cheaper and stays entirely in
Triton (portable, no torch fallback).

Portability
-----------
The kernels are plain Triton: no device-specific branching, no torch
fallback.  The pointer-shift trick is a host-agnostic way to keep
addresses canonical; the int32 offset math stays within
``dim * total_t < 2**31``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def causal_conv1d_fn_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    seg_start_ptr,
    out_ptr,
    TOTAL_T: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
):
    # grid = (num_t_blocks, num_d_blocks): packed time on grid.x
    # (<= 65535), channel blocks on grid.y (<= 255). Each program owns a
    # BLOCK_D x BLOCK_T output tile.
    pid_t = tl.program_id(0)  # packed-time block
    pid_d = tl.program_id(1)  # channel block

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # (BLOCK_T,) pos
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # (BLOCK_D,)
    mask_t = offs_t < TOTAL_T
    mask_d = offs_d < DIM

    # seg_start[t] = start of the segment that packed position t belongs
    # to.
    seg_start = tl.load(seg_start_ptr + offs_t, mask=mask_t, other=0)

    # Canonical aligned tile address (no shift in the index) -- keeps
    # loads vectorized.
    addr = offs_d[:, None] * TOTAL_T + offs_t[None, :]  # (D, T)

    # Left limit of valid taps: tap k reads x[t - (WIDTH-1) + k], which
    # is causal-valid iff t - (WIDTH-1) + k >= seg_start[t]
    # <=> t >= seg_start[t] + (WIDTH-1) - k.
    lo = seg_start + (WIDTH - 1)  # (BLOCK_T,)

    acc = tl.zeros((BLOCK_D, BLOCK_T), dtype=tl.float32)
    for k in range(WIDTH):
        # Scalar pointer shift (NOT an index shift): keeps the index
        # canonical so the GCU backend emits vectorized loads.  Read
        # position = offs_t - (WIDTH-1) + k.
        base = x_ptr - (WIDTH - 1) + k
        # Compile-time-only load mask: global bounds.
        # (offs_t >= (WIDTH-1)-k) covers the left edge of the whole
        # tensor; mask_t the right edge; reads never exceed TOTAL_T-1
        # because the window looks only leftward.
        valid = (
            mask_t[None, :]
            & mask_d[:, None]
            & (offs_t >= (WIDTH - 1) - k)[None, :]
        )
        v = tl.load(base + addr, mask=valid, other=0.0).to(tl.float32)
        # Segment boundary: zero lanes whose window reaches left of the
        # segment start. Done elementwise (cheap) instead of in the
        # load mask (expensive).
        ok = (offs_t >= (lo - k))[None, :]
        v = tl.where(ok, v, 0.0)

        wv = tl.load(weight_ptr + offs_d * WIDTH + k, mask=mask_d, other=0.0)
        acc += v * wv[:, None]

    if HAS_BIAS:
        bv = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0)
        acc += bv[:, None]

    if HAS_ACTIVATION:
        acc = acc / (1.0 + tl.exp(-acc))  # SiLU / swish

    store_mask = mask_t[None, :] & mask_d[:, None]
    tl.store(out_ptr + addr, acc, mask=store_mask)


@triton.jit
def build_seg_start_kernel(
    qsl_ptr,
    seg_start_ptr,
    NSEGS,
    TOTAL_T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """seg_start[t] = max_{i} qsl[i] subject to qsl[i] <= t.

    i.e. the start of the segment containing packed position t.  ``qsl``
    is the sorted prefix-sum of sequence lengths, so the running max over
    all boundaries <= t is exactly the right boundary.

    NSEGS is a *runtime* scalar (number of boundaries = n_seqs + 1), so
    the loop stays dynamic and does not fully unroll for large n_seqs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_T
    t = offs

    # Start from the first boundary qsl[0] (== 0 in practice) and keep
    # the running max of every boundary <= t.
    seg = tl.load(qsl_ptr + 0) + tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(1, NSEGS):
        b = tl.load(qsl_ptr + i)
        seg = tl.maximum(seg, tl.where(b <= t, b, seg))
    tl.store(seg_start_ptr + offs, seg, mask=mask)


def _next_pow2(n):
    """Smallest power of two >= n (n >= 1)."""
    return 1 << (n - 1).bit_length()


def _launch_config(dim, total_t):
    """Pick BLOCK_T / BLOCK_D / num_warps.

    Honors the GCU grid + local-memory caps.
    """
    # Channel block: 128 measured fastest for the dim=5120 benchmark,
    # but never more blocks than the grid.y cap (255).
    block_d = min(128, _next_pow2(dim))
    while (dim + block_d - 1) // block_d > 255:
        block_d *= 2

    # Time block: large tiles win on the GCU, but the fp32 accumulator
    # (BLOCK_T * BLOCK_D) must stay under the ~1.5MB local-memory limit
    # -- cap the tile at 128Ki elements.
    block_t = min(1024, _next_pow2(total_t))
    while block_t > 1 and block_t * block_d > 131072:
        block_t //= 2

    num_t_blocks = triton.cdiv(total_t, block_t)
    # Guard grid.x <= 65535 for pathological total_t (grow the tile
    # instead).
    while num_t_blocks > 65535:
        block_t *= 2
        num_t_blocks = triton.cdiv(total_t, block_t)

    num_warps = 2 if block_t * block_d <= 131072 else 4
    return block_t, block_d, num_warps


def causal_conv1d_fn(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    """Fused causal depthwise conv1d + optional silu.

    Signature matches the reference.
    """
    dim, total_t = x.shape
    width = weight.shape[1]

    # Per packed-position segment start. Built on-device with a Triton
    # kernel (see ``build_seg_start_kernel``): ``torch.searchsorted`` +
    # gather is surprisingly slow on the GCU (~1.1ms for a single long
    # sequence, ~350us for many segments), while the branchless max-scan
    # below is 1.5-45x cheaper and stays entirely in Triton.
    # (seq_lens_cpu is accepted for API compatibility but not needed:
    # query_start_loc already encodes every segment boundary.)
    qsl = query_start_loc.to(device=x.device, dtype=torch.int32)
    seg_start = torch.empty(total_t, device=x.device, dtype=torch.int32)
    seg_block = min(1024, _next_pow2(max(total_t, 1)))
    build_seg_start_kernel[(triton.cdiv(total_t, seg_block),)](
        qsl,
        seg_start,
        qsl.numel(),
        TOTAL_T=total_t,
        BLOCK=seg_block,
        num_warps=1,
    )

    block_t, block_d, num_warps = _launch_config(dim, total_t)
    num_t_blocks = triton.cdiv(total_t, block_t)
    num_d_blocks = triton.cdiv(dim, block_d)

    out = torch.empty_like(x)
    has_activation = activation in ("silu", "swish")

    causal_conv1d_fn_kernel[(num_t_blocks, num_d_blocks)](
        x,
        weight,
        x if bias is None else bias,
        seg_start,
        out,
        TOTAL_T=total_t,
        DIM=dim,
        WIDTH=width,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        HAS_BIAS=bias is not None,
        HAS_ACTIVATION=has_activation,
        num_warps=num_warps,
    )
    return out


__all__ = ["causal_conv1d_fn"]
