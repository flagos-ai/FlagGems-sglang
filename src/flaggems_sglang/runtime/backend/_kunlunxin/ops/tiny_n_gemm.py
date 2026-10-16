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

import torch
import triton
import triton.language as tl

# Reduction tile along K.  The full K is looped inside one program; this only
# bounds the per-iteration working set.
_BLOCK_K = 512

_N_CONFIGS = {
    64: (4, 3),
    128: (4, 3),
    256: (8, 2),
    512: (8, 2),
}


@triton.jit
def _tiny_n_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST: tl.constexpr,
):
    """out[m, n] = sum_k x[m, k] * w[n, k], for one (m-block, n-block) tile.

    Grid is ``(num_n_blocks, num_m_blocks)``; every program reduces the *whole*
    K axis on its own, so no cross-program accumulation is required.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + offs_k
        mask_k = ks < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + ks[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + ks[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )

        if FAST:
            # bf16/f16 inputs: products are exact in f32 and tl.dot accumulates
            # in f32 -- same arithmetic as the reference, native-
            # precision path.
            acc = tl.dot(x, tl.trans(w), acc)
        else:
            # float32 output: match the reference's exact f32 arithmetic.
            xf = x.to(tl.float32)
            wf = w.to(tl.float32)
            acc = tl.dot(xf, tl.trans(wf), acc, input_precision="ieee")

    out_ptrs = (
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _split_n(num_sms, n):
    """Pick the smallest BLOCK_N that fills the machine with one clean."""
    best = None
    best_key = None
    for block_n in _N_CONFIGS:
        blocks = triton.cdiv(n, block_n)
        remainder = blocks % num_sms
        key = (remainder, block_n)
        if best_key is None or key < best_key:
            best_key = key
            best = block_n
    return best


def tiny_n_gemm(x, w, out_dtype):
    """out = x @ w.T with x:[m, k], w:[n, k] bf16 -> out:[m, n] out_dtype."""
    m, k = x.shape
    n, wk = w.shape
    assert wk == k, f"incompatible K: x has {k}, w has {wk}"

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)

    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    block_n = _split_n(num_sms, n)
    warps_stages = _N_CONFIGS[block_n]
    block_k = min(_BLOCK_K, triton.next_power_of_2(max(1, k)))
    num_k_iters = triton.cdiv(k, block_k)
    if num_k_iters > 8:
        num_stages = 2
    else:
        num_stages = warps_stages[1]
    num_warps = warps_stages[0]

    grid = (triton.cdiv(n, block_n), triton.cdiv(m, 16))
    _tiny_n_gemm_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        FAST=(out_dtype != torch.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["tiny_n_gemm"]
