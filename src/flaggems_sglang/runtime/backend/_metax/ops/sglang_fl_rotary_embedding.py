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

"""Fused in-place RoPE (SGLang apply_rope_with_cos_sin_cache_inplace) in Triton.

Semantics (identical to the reference):
  * rotary_dim RD = cos_sin_cache.shape[-1]; cos = row[:RD/2], sin = row[RD/2:].
  * positions index rows of the cache (validated upstream).
  * query [nnz, Hq*HS] and key [nnz, Hkv*HS] are rotated in place, lanes
    [RD, HS) of every head are left bit-identical, and the rotation is done in
    fp32 with exactly one rounding back to the input dtype.
  * run() returns None.

One kernel launch covers both tensors: grid axis 0 tiles the token dimension and
grid axis 1 selects query (0) / key (1).  Per head pair index i the kernel loads
the two operands of the rotation directly (a = lane a_off(i), b = lane b_off(i))
so the neox (i, i+RD/2) and interleaved (2i, 2i+1) layouts share one code path.

Two structural forms are compiled and chosen by a static table keyed on nnz (no
in-run autotuning):

  * _rope_rotate_kernel -- inner loop over head groups; the cos/sin row stays in
    registers across heads, so this wins once the kernel is long enough for
    memory traffic to matter.
  * _rope_rotate_hg_kernel -- the head-group index lives on grid axis 2, so the
    head loop disappears and the work is spread over more, shorter programs.
    This wins for very short kernels (nnz <= 128), where the critical path is
    program serialization rather than bandwidth.  Out-of-range head groups exit
    immediately, so the rectangular grid costs nothing.

All tile/num_warps entries were chosen from offline do_bench sweeps on the target
device and every accepted entry was verified against the verbatim reference
before its timing was kept.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_rotate_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cache_ptr,
    nnz,
    rs_q,
    rs_k,
    hs_q,
    hs_k,
    hq,
    hk,
    pstride,
    RD: tl.constexpr,
    P2: tl.constexpr,
    BLOCK_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_ten = tl.program_id(1)

    if pid_ten == 0:
        x_ptr = q_ptr
        rs = rs_q
        hstride = hs_q
        nheads = hq
    else:
        x_ptr = k_ptr
        rs = rs_k
        hstride = hs_k
        nheads = hk

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < nnz
    pos = tl.load(pos_ptr + offs_t * pstride, mask=tmask, other=0).to(tl.int32)

    offs_p = tl.arange(0, BLOCK_P2)
    pmask = offs_p < P2

    # cos / sin for this token block: one fp32 row of RD elements per token.
    cb = pos[:, None] * RD
    cmask = tmask[:, None] & pmask[None, :]
    cos = tl.load(cache_ptr + cb + offs_p[None, :], mask=cmask, other=0.0)
    sin = tl.load(cache_ptr + cb + P2 + offs_p[None, :], mask=cmask, other=0.0)

    if IS_NEOX:
        a_off = offs_p
        b_off = offs_p + P2
    else:
        a_off = 2 * offs_p
        b_off = 2 * offs_p + 1

    for h0 in range(0, nheads, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hmask = offs_h < nheads
        m = tmask[:, None, None] & hmask[None, :, None] & pmask[None, None, :]
        base = offs_t[:, None, None] * rs + (offs_h * hstride)[None, :, None]
        pa = base + a_off[None, None, :]
        pb = base + b_off[None, None, :]

        xa = tl.load(x_ptr + pa, mask=m, other=0.0)
        xb = tl.load(x_ptr + pb, mask=m, other=0.0)
        af = xa.to(tl.float32)
        bf = xb.to(tl.float32)

        c = cos[:, None, :]
        s = sin[:, None, :]
        oa = af * c - bf * s
        ob = bf * c + af * s

        # Single rounding back to the input dtype, written in place.
        tl.store(x_ptr + pa, oa.to(xa.dtype), mask=m)
        tl.store(x_ptr + pb, ob.to(xb.dtype), mask=m)


@triton.jit
def _rope_rotate_hg_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cache_ptr,
    nnz,
    rs_q,
    rs_k,
    hs_q,
    hs_k,
    hq,
    hk,
    pstride,
    RD: tl.constexpr,
    P2: tl.constexpr,
    BLOCK_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    """Same math as _rope_rotate_kernel, with head groups on grid axis 2."""
    pid_t = tl.program_id(0)
    pid_ten = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_ten == 0:
        x_ptr = q_ptr
        rs = rs_q
        hstride = hs_q
        nheads = hq
    else:
        x_ptr = k_ptr
        rs = rs_k
        hstride = hs_k
        nheads = hk

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    if pid_h * BLOCK_H < nheads:
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = offs_t < nnz
        pos = tl.load(pos_ptr + offs_t * pstride, mask=tmask, other=0).to(
            tl.int32
        )

        offs_p = tl.arange(0, BLOCK_P2)
        pmask = offs_p < P2
        cb = pos[:, None] * RD
        cmask = tmask[:, None] & pmask[None, :]
        cos = tl.load(cache_ptr + cb + offs_p[None, :], mask=cmask, other=0.0)
        sin = tl.load(
            cache_ptr + cb + P2 + offs_p[None, :], mask=cmask, other=0.0
        )

        if IS_NEOX:
            a_off = offs_p
            b_off = offs_p + P2
        else:
            a_off = 2 * offs_p
            b_off = 2 * offs_p + 1

        m = (
            tmask[:, None, None]
            & (offs_h < nheads)[None, :, None]
            & pmask[None, None, :]
        )
        base = offs_t[:, None, None] * rs + (offs_h * hstride)[None, :, None]
        pa = base + a_off[None, None, :]
        pb = base + b_off[None, None, :]

        xa = tl.load(x_ptr + pa, mask=m, other=0.0)
        xb = tl.load(x_ptr + pb, mask=m, other=0.0)
        af = xa.to(tl.float32)
        bf = xb.to(tl.float32)

        c = cos[:, None, :]
        s = sin[:, None, :]
        tl.store(x_ptr + pa, (af * c - bf * s).to(xa.dtype), mask=m)
        tl.store(x_ptr + pb, (bf * c + af * s).to(xb.dtype), mask=m)


# Static per-regime launch plan; entries come from offline target sweeps.
_HG_MAX_NNZ = 128


def _pick_config(nnz, block_p2, max_heads):
    """Return (kernel_selector, block_t, block_h, num_warps)."""
    if block_p2 <= 32:
        if nnz <= _HG_MAX_NNZ:
            # 3-D form: short kernels are program-serialization limited.
            kernel = 1
            block_t, block_h, num_warps = 1, 16, 2
        elif nnz <= 512:
            kernel = 0
            block_t, block_h, num_warps = 1, 16, 2
        else:
            kernel = 0
            block_t, block_h, num_warps = 2, 8, 2
    else:
        # rotary_dim >= 128: keep the per-program register tile small.
        kernel = 0
        if nnz <= 1024:
            block_t, block_h, num_warps = 2, 4, 4
        else:
            block_t, block_h, num_warps = 4, 4, 4
    if max_heads > 0:
        block_h = min(block_h, triton.next_power_of_2(max_heads))
    block_h = max(1, block_h)
    return kernel, block_t, block_h, num_warps


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    head_size = int(head_size)
    rotary_dim = int(cos_sin_cache.shape[-1])

    nnz = int(query.shape[0])
    if nnz == 0 or int(key.shape[0]) == 0:
        return query, key

    hq = int(query.shape[1]) // head_size
    hk = int(key.shape[1]) // head_size
    if hq == 0 or hk == 0:
        return query, key

    if positions.dim() != 1:
        positions = positions.reshape(-1)
    pstride = int(positions.stride(0))

    rs_q = int(query.stride(0))
    rs_k = int(key.stride(0))
    hs_q = head_size * int(query.stride(1))
    hs_k = head_size * int(key.stride(1))

    p2 = rotary_dim // 2
    block_p2 = triton.next_power_of_2(p2)

    kernel, block_t, block_h, num_warps = _pick_config(
        nnz, block_p2, max(hq, hk)
    )

    args = (
        query,
        key,
        positions,
        cos_sin_cache,
        nnz,
        rs_q,
        rs_k,
        hs_q,
        hs_k,
        hq,
        hk,
        pstride,
    )
    kwargs = dict(
        RD=rotary_dim,
        P2=p2,
        BLOCK_P2=block_p2,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        IS_NEOX=bool(is_neox),
        num_warps=num_warps,
    )

    if kernel == 1:
        grid = (
            triton.cdiv(nnz, block_t),
            2,
            triton.cdiv(max(hq, hk), block_h),
        )
        _rope_rotate_hg_kernel[grid](*args, **kwargs)
    else:
        grid = (triton.cdiv(nnz, block_t), 2)
        _rope_rotate_kernel[grid](*args, **kwargs)
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
