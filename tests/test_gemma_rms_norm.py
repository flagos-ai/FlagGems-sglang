import pytest
import torch

# Reference: SGLang's compiled Triton kernel as correctness baseline.
from sgl_kernel import (  # noqa: E402
    gemma_fused_add_rmsnorm as _sglang_gemma_fused_add_rmsnorm,
)
from sgl_kernel import gemma_rmsnorm as _sglang_gemma_rmsnorm  # noqa: E402

import flaggems_sglang

from . import conftest as cfg

# sgl_kernel.gemma_rmsnorm only dispatches half-precision types (float16/bfloat16).
FLOAT_DTYPES = [torch.float16, torch.bfloat16]


# NORM_SHAPES: list of (M, N) tuples — M rows, N hidden dim.
NORM_SHAPES = [(1, 512), (4, 1024), (32, 2048), (64, 4096), (128, 8192)]


def _ref_gemma_rms_norm(x, weight, eps):
    """Reference: delegate to SGLang's Triton gemma_rmsnorm kernel."""
    return _sglang_gemma_rmsnorm(x, weight, eps)


def _ref_gemma_fused_add_rms_norm(x, residual, weight, eps):
    """Reference: delegate to SGLang's Triton gemma_fused_add_rmsnorm kernel.
    Note: sgl_kernel modifies x and residual in-place and returns None.
    After the call, x holds the normalized output and residual holds x+residual.
    """
    _sglang_gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual


@pytest.mark.parametrize("norm_shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(norm_shape, dtype):
    device = cfg.device
    M, N = norm_shape
    weight = torch.zeros(N, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)

    ref = _ref_gemma_rms_norm(x, weight, 1e-6)
    res = flaggems_sglang.gemma_rms_norm(x, weight, eps=1e-6)

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("norm_shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm_res
def test_gemma_rms_norm_with_residual(norm_shape, dtype):
    device = cfg.device
    M, N = norm_shape
    weight = torch.zeros(N, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device)

    x_ref = x.clone()
    res_ref = residual.clone()
    ref_out, ref_residual = _ref_gemma_fused_add_rms_norm(
        x_ref, res_ref, weight, 1e-6
    )

    out, res_out = flaggems_sglang.gemma_rms_norm(
        x.clone(), weight, eps=1e-6, residual=residual.clone()
    )

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)
    torch.testing.assert_close(res_out, ref_residual, atol=atol, rtol=1e-2)
