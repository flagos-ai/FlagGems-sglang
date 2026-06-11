"""Optimized Gemma RMSNorm Triton kernel - v13: 8 warps for small N rmsnorm
to hide memory latency, revert N=5120 to single-pass (tiled read-twice was
worse than padding), keep multi-row counts from v12 for weight amortization.

Key changes from v12:
- rmsnorm small N: 8 warps instead of 4 (same multi-row: 16/8/4 rows)
  → more warps = better memory latency hiding for memory-bound kernel
- Remove tiled 2-pass kernel (N=5120 regression: 42.6us vs single-pass 34.1us)
- Fused kernel unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    stride_x_row,
    stride_out_row,
    N,
    M,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Single-pass RMSNorm — BLOCK_N may be padded to next power of 2."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = 1.0 + w

    for r in tl.static_range(ROWS_PER_PROGRAM):
        row_idx = row_start + r
        if row_idx < M:
            x = tl.load(
                X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0
            )
            x_fp32 = x.to(tl.float32)

            mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / N
            rrms = tl.rsqrt(mean_sq + eps)

            out = x_fp32 * rrms * scale
            tl.store(
                Out_ptr + row_idx * stride_out_row + cols,
                out.to(x.dtype),
                mask=mask,
            )


@triton.jit
def _gemma_fused_add_rmsnorm_kernel(
    X_ptr,
    Residual_ptr,
    W_ptr,
    Out_ptr,
    ResidualOut_ptr,
    stride_x_row,
    stride_res_row,
    stride_out_row,
    stride_resout_row,
    N,
    M,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Single-pass fused add+rmsnorm."""
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = 1.0 + w

    for r in tl.static_range(ROWS_PER_PROGRAM):
        row_idx = row_start + r
        if row_idx < M:
            x = tl.load(
                X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0
            )
            residual = tl.load(
                Residual_ptr + row_idx * stride_res_row + cols,
                mask=mask,
                other=0.0,
            )

            x_fp32 = x.to(tl.float32)
            res_fp32 = residual.to(tl.float32)

            hidden = x_fp32 + res_fp32

            tl.store(
                ResidualOut_ptr + row_idx * stride_resout_row + cols,
                hidden.to(x.dtype),
                mask=mask,
            )

            mean_sq = tl.sum(hidden * hidden, axis=0) / N
            rrms = tl.rsqrt(mean_sq + eps)

            out = hidden * rrms * scale
            tl.store(
                Out_ptr + row_idx * stride_out_row + cols,
                out.to(x.dtype),
                mask=mask,
            )


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_rmsnorm_config(N):
    """Return (BLOCK_N, ROWS_PER_PROGRAM, num_warps).

    Strategy:
    - Small N: aggressive multi-row amortizes weight/launch overhead.
      8 warps hides memory latency better than 4.
    - Medium/large N: single-row with 8 warps for high occupancy.
    """
    if N <= 512:
        return 512, 16, 8  # 512*16/256=32 elem/thd, grid=256, 16 warps/SM
    elif N <= 1024:
        return 1024, 8, 8  # 1024*8/256=32 elem/thd, grid=512, 32 warps/SM
    elif N <= 2048:
        return 2048, 4, 8  # 2048*4/256=32 elem/thd, grid=1024, 64 warps/SM
    elif N <= 3072:
        return 4096, 1, 8  # 4096/256=16 elem/thd (3072→4096 padded)
    elif N <= 4096:
        return 4096, 1, 8  # 4096/256=16 elem/thd
    elif N <= 5120:
        return 8192, 1, 8  # 8192/256=32 elem/thd (5120→8192 padded)
    else:
        return 8192, 1, 8  # 8192/256=32 elem/thd


def _get_fused_config(N):
    """Return (BLOCK_N, ROWS_PER_PROGRAM, num_warps, num_stages)."""
    BLOCK_N = _next_power_of_2(N)
    if N <= 512:
        return BLOCK_N, 8, 8, 2
    elif N <= 1024:
        return BLOCK_N, 4, 8, 2
    elif BLOCK_N <= 4096:
        return BLOCK_N, 2, 8, 2
    else:
        return BLOCK_N, 1, 16, 2


def gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_contiguous()
    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])

    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N, ROWS_PER_PROGRAM, num_warps = _get_rmsnorm_config(N)

    grid = _cdiv(M, ROWS_PER_PROGRAM)
    _gemma_rmsnorm_kernel[(grid,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        N,
        M,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=num_warps,
    )

    if len(orig_shape) != 2:
        out = out.reshape(orig_shape)
    return out


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.shape == residual.shape

    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])
        residual = residual.reshape(-1, orig_shape[-1])

    M, N = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    BLOCK_N, ROWS_PER_PROGRAM, num_warps, num_stages = _get_fused_config(N)

    grid = _cdiv(M, ROWS_PER_PROGRAM)
    _gemma_fused_add_rmsnorm_kernel[(grid,)](
        x,
        residual,
        weight,
        out,
        residual_out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        residual_out.stride(0),
        N,
        M,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if len(orig_shape) != 2:
        out = out.reshape(orig_shape)
        residual_out = residual_out.reshape(orig_shape)
    return out, residual_out


def gemma_rms_norm(x, weight, eps=1e-6, residual=None):
    if residual is not None:
        return gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return gemma_rmsnorm(x, weight, eps)
