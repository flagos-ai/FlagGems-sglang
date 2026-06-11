"""Optimized Gemma RMSNorm Triton kernel with autotuning support.

Multi-row processing for small N hides memory latency by amortizing
weight load and kernel launch overhead across multiple rows.
8-warp configurations improve occupancy for memory-bound small-N kernels.

Based on v13 kernel: 8 warps for small N rmsnorm to hide memory latency,
single-pass for N=5120 (tiled read-twice was worse than padding),
multi-row counts from v12 for weight amortization.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flaggems_sglang.runtime import torch_device_fn
from flaggems_sglang.utils import libentry

logger = logging.getLogger(__name__)


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


@libentry()
@triton.autotune(
    configs=[
        # Small N: aggressive multi-row with 8 warps to hide
        # memory latency and amortize weight/launch overhead.
        triton.Config({"BLOCK_N": 512, "ROWS_PER_PROGRAM": 16}, num_warps=8),
        triton.Config({"BLOCK_N": 512, "ROWS_PER_PROGRAM": 8}, num_warps=8),
        triton.Config({"BLOCK_N": 1024, "ROWS_PER_PROGRAM": 8}, num_warps=8),
        triton.Config({"BLOCK_N": 1024, "ROWS_PER_PROGRAM": 4}, num_warps=8),
        # Medium N: moderate multi-row or single-row with 8 warps.
        triton.Config({"BLOCK_N": 2048, "ROWS_PER_PROGRAM": 4}, num_warps=8),
        triton.Config({"BLOCK_N": 2048, "ROWS_PER_PROGRAM": 2}, num_warps=8),
        triton.Config({"BLOCK_N": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"BLOCK_N": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=16),
        # Large N: single-row with adequate block size.
        triton.Config({"BLOCK_N": 8192, "ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"BLOCK_N": 8192, "ROWS_PER_PROGRAM": 1}, num_warps=16),
    ],
    key=["N"],
    reset_to_zero=["Out_ptr"],
)
@triton.jit
def _gemma_rms_norm_kernel(
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


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 512, "ROWS_PER_PROGRAM": 8}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 1024, "ROWS_PER_PROGRAM": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 2048, "ROWS_PER_PROGRAM": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 4096, "ROWS_PER_PROGRAM": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 4096, "ROWS_PER_PROGRAM": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 8192, "ROWS_PER_PROGRAM": 1},
            num_warps=16,
            num_stages=2,
        ),
    ],
    key=["N"],
    reset_to_zero=["Out_ptr", "ResidualOut_ptr"],
)
@triton.jit
def _gemma_fused_add_rms_norm_kernel(
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


def gemma_rms_norm(x, weight, eps=1e-6, residual=None):
    """Optimized Gemma RMSNorm with optional fused residual addition.

    Args:
        x: Input tensor of any shape where the last dimension is the
            feature dim.
        weight: Weight tensor of shape (N,) for the normalized dimension.
        eps: Epsilon for numerical stability (default: 1e-6).
        residual: Optional residual tensor of same shape as x for fused add.

    Returns:
        If residual is None: normalized output tensor.
        If residual is not None: (normalized_output, updated_residual) tuple.
    """
    assert x.is_contiguous()
    orig_shape = x.shape
    N = weight.shape[0]
    M = math.prod(orig_shape[:-1]) if x.dim() > 1 else 1

    if x.dim() != 2:
        x = x.reshape(-1, N)

    weight = weight.contiguous()

    if residual is not None:
        logger.debug(
            "FLAGGEMS_SGLANG GEMMA_RMS_NORM (fused add), "
            "[input shape]: %s, [residual shape]: %s, [weight shape]: %s",
            x.size(),
            residual.size(),
            weight.size(),
        )
        assert residual.is_contiguous()
        assert (
            x.shape == residual.shape
            or residual.reshape(-1, N).shape == x.shape
        )
        if residual.dim() != 2:
            residual = residual.reshape(-1, N)

        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            row_stride = N
            grid = lambda meta: (_cdiv(M, meta["ROWS_PER_PROGRAM"]),)
            _gemma_fused_add_rms_norm_kernel[grid](
                x,
                residual,
                weight,
                out,
                residual_out,
                row_stride,
                row_stride,
                row_stride,
                row_stride,
                N,
                M,
                eps,
            )

        if len(orig_shape) != 2:
            out = out.reshape(orig_shape)
            residual_out = residual_out.reshape(orig_shape)
        return out, residual_out
    else:
        logger.debug(
            "FLAGGEMS_SGLANG GEMMA_RMS_NORM, [input shape]: %s, [weight shape]: %s",
            x.size(),
            weight.size(),
        )
        out = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            row_stride = N
            grid = lambda meta: (_cdiv(M, meta["ROWS_PER_PROGRAM"]),)
            _gemma_rms_norm_kernel[grid](
                x,
                weight,
                out,
                row_stride,
                row_stride,
                N,
                M,
                eps,
            )

        if len(orig_shape) != 2:
            out = out.reshape(orig_shape)
        return out
