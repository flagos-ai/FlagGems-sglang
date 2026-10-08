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
def _w8a8_bk_matmul(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSN: tl.constexpr,
    BSK: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    NK: tl.constexpr,
    K_PER_SCALE: tl.constexpr,
    N_PER_SCALE: tl.constexpr,
    GRID_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_m = tile // GRID_N
    pid_n = tile - pid_m * GRID_N
    nblk = pid_n // N_PER_SCALE

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    in_m = rm < M
    in_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * AM + rk[None, :] * AK
    b_ptrs = b_ptr + rn[None, :] * BN + rk[:, None] * BK
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(NK):
        ik = i // K_PER_SCALE
        if EVEN_K:
            a = tl.load(a_ptrs, mask=in_m[:, None], other=0)
            b = tl.load(b_ptrs, mask=in_n[None, :], other=0)
        else:
            in_k = (i * BLOCK_K + rk) < K
            a = tl.load(a_ptrs, mask=in_m[:, None] & in_k[None, :], other=0)
            b = tl.load(b_ptrs, mask=in_k[:, None] & in_n[None, :], other=0)
        sa = tl.load(as_ptr + rm * ASM + ik * ASK, mask=in_m, other=0.0)
        sb = tl.load(bs_ptr + nblk * BSN + ik * BSK)
        a_scaled = a.to(tl.float32) * (sa * sb)[:, None]
        acc = tl.dot(
            a_scaled,
            b.to(tl.float32),
            acc,
            input_precision="ieee",
            out_dtype=tl.float32,
        )
        a_ptrs += BLOCK_K * AK
        b_ptrs += BLOCK_K * BK

    c_ptrs = c_ptr + rm[:, None] * CM + rn[None, :] * CN
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=in_m[:, None] & in_n[None, :],
        )


def _pow2_at_most(value, cap):
    size = 16
    while size * 2 <= value and size * 2 <= cap:
        size = size * 2
    return size


def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()
    m, k = A.shape
    n = B.shape[0]
    block_n, block_k = int(block_size[0]), int(block_size[1])
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    if m == 0 or n == 0:
        return out

    block_m = _pow2_at_most(m, 64)
    tile_n = _pow2_at_most(block_n, 128)
    tile_k = _pow2_at_most(block_k, 128)
    grid_m = (m + block_m - 1) // block_m
    grid_n = (n + tile_n - 1) // tile_n
    nk = (k + tile_k - 1) // tile_k

    _w8a8_bk_matmul[(grid_m * grid_n,)](
        A,
        B,
        As,
        Bs,
        out,
        M=m,
        N=n,
        K=k,
        AM=A.stride(0),
        AK=A.stride(1),
        BN=B.stride(0),
        BK=B.stride(1),
        ASM=As.stride(0),
        ASK=As.stride(1),
        BSN=Bs.stride(0),
        BSK=Bs.stride(1),
        CM=out.stride(0),
        CN=out.stride(1),
        NK=nk,
        K_PER_SCALE=block_k // tile_k,
        N_PER_SCALE=block_n // tile_n,
        GRID_N=grid_n,
        EVEN_M=(m % block_m == 0),
        EVEN_N=(n % tile_n == 0),
        EVEN_K=(k % tile_k == 0),
        BLOCK_M=block_m,
        BLOCK_N=tile_n,
        BLOCK_K=tile_k,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["w8a8_block_int8_matmul"]
