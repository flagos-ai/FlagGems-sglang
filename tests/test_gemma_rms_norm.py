import pytest
import torch
import torch.nn as nn

# Reference: SGLang's compiled Triton kernel as correctness baseline.
from sgl_kernel import (  # noqa: E402
    gemma_fused_add_rmsnorm as _sglang_gemma_fused_add_rmsnorm,
)
from sgl_kernel import gemma_rmsnorm as _sglang_gemma_rmsnorm  # noqa: E402

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


class _MockGemmaRMSNorm(nn.Module):
    """Minimal mock of torch.nn.Module with the attributes expected by
    our gemma_rms_norm API."""

    def __init__(
        self, normalized_shape, eps=1e-6, dtype=torch.float32, device="cpu"
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(normalized_shape, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps


def _ref_gemma_rms_norm(x, weight, eps):
    """Reference: delegate to SGLang's Triton gemma_rmsnorm kernel."""
    return _sglang_gemma_rmsnorm(x, weight, eps)


def _ref_gemma_fused_add_rms_norm(x, residual, weight, eps):
    """Reference: delegate to SGLang's Triton gemma_fused_add_rmsnorm kernel.
    Note: sgl_kernel modifies x and residual in-place and returns (x, residual).
    """
    return _sglang_gemma_fused_add_rmsnorm(x, residual, weight, eps)


@pytest.mark.parametrize("norm_shape", utils.NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(norm_shape, dtype):
    device = cfg.device
    x_shape, norm_shape_w = norm_shape
    module = _MockGemmaRMSNorm(
        norm_shape_w, eps=1e-6, dtype=dtype, device=device
    )
    x = torch.randn(x_shape, dtype=dtype, device=device)

    ref = _ref_gemma_rms_norm(x, module.weight.data, module.variance_epsilon)
    res = flaggems_sglang.gemma_rms_norm(module, x)

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("norm_shape", utils.NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm_with_residual(norm_shape, dtype):
    device = cfg.device
    x_shape, norm_shape_w = norm_shape
    module = _MockGemmaRMSNorm(
        norm_shape_w, eps=1e-6, dtype=dtype, device=device
    )
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.randn(x_shape, dtype=dtype, device=device)

    x_ref = x.clone()
    res_ref = residual.clone()
    ref_out, ref_residual = _ref_gemma_fused_add_rms_norm(
        x_ref, res_ref, module.weight.data, module.variance_epsilon
    )

    out, res_out = flaggems_sglang.gemma_rms_norm(
        module, x.clone(), residual.clone()
    )

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)
    torch.testing.assert_close(res_out, ref_residual, atol=atol, rtol=1e-2)
