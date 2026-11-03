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

"""Fused affine-free LayerNorm over the head dimension of Q and K."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ln_noaffine(x, eps, DH: tl.constexpr):
    """Affine-free LayerNorm along the last axis of a 2-D tile."""
    mean = tl.sum(x, axis=1) / DH
    var = tl.sum(x * x, axis=1) / DH - mean * mean
    return (x - mean[:, None]) * tl.rsqrt(var + eps)[:, None]


@triton.jit
def _qk_ln_kernel(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    NROWS,
    eps,
    DH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    offs = rows[:, None] * DH + cols[None, :]

    if EVEN_R:
        if EVEN_D:
            offs = tl.max_contiguous(tl.multiple_of(offs, (1, DH)), (1, DH))
            xq = tl.load(q_ptr + offs, eviction_policy="evict_first").to(
                tl.float32
            )
            xk = tl.load(k_ptr + offs, eviction_policy="evict_first").to(
                tl.float32
            )
            yq = _ln_noaffine(xq, eps, DH)
            yk = _ln_noaffine(xk, eps, DH)
            tl.store(
                qo_ptr + offs,
                yq.to(qo_ptr.dtype.element_ty),
                cache_modifier=".cs",
            )
            tl.store(
                ko_ptr + offs,
                yk.to(ko_ptr.dtype.element_ty),
                cache_modifier=".cs",
            )
        else:
            # Row range is exact; only the tail of the head dim is masked.
            cmask = (cols < DH)[None, :]
            xq = tl.load(
                q_ptr + offs,
                mask=cmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            xk = tl.load(
                k_ptr + offs,
                mask=cmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            yq = _ln_noaffine(xq, eps, DH)
            yk = _ln_noaffine(xk, eps, DH)
            tl.store(
                qo_ptr + offs,
                yq.to(qo_ptr.dtype.element_ty),
                mask=cmask,
                cache_modifier=".cs",
            )
            tl.store(
                ko_ptr + offs,
                yk.to(ko_ptr.dtype.element_ty),
                mask=cmask,
                cache_modifier=".cs",
            )
    else:
        row_mask = rows < NROWS
        if EVEN_D:
            mask = row_mask[:, None]
        else:
            mask = row_mask[:, None] & (cols[None, :] < DH)

        # Issue both tensor loads before any math so the Q and K requests are
        # in flight together.
        xq = tl.load(
            q_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        xk = tl.load(
            k_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        yq = _ln_noaffine(xq, eps, DH)
        yk = _ln_noaffine(xk, eps, DH)
        tl.store(
            qo_ptr + offs,
            yq.to(qo_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=".cs",
        )
        tl.store(
            ko_ptr + offs,
            yk.to(ko_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=".cs",
        )


def qk_head_layernorm(q, k, eps):
    """Per-head affine-free LayerNorm over q and k, one Triton launch."""
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    qo = torch.empty_like(q)
    ko = torch.empty_like(k)

    dh = q.shape[-1]
    nrows = q.numel() // dh
    if nrows == 0:
        return qo, ko

    block_d = triton.next_power_of_2(dh)
    # All candidate BLOCK_R values are powers of two <= 64, so divisibility by
    # 64 makes the row range exact for whichever config is selected.
    even_r = (nrows % 64) == 0

    # Static launch config: 4 rows x 1 warp for typical sizes; the largest
    # sizes prefer more, smaller programs for better load overlap.
    block_r = 2 if nrows > 262144 else 4
    num_warps = 1

    grid = (triton.cdiv(nrows, block_r),)
    _qk_ln_kernel[grid](
        q,
        k,
        qo,
        ko,
        nrows,
        eps,
        DH=dh,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        EVEN_D=(block_d == dh),
        EVEN_R=even_r,
        num_warps=num_warps,
        num_stages=2,
    )
    return qo, ko


__all__ = ["qk_head_layernorm"]
