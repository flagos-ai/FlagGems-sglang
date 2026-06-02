import pytest
import torch
import torch.nn as nn

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


class _MockGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps


def _torch_gemma_rms_norm(x, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * hidden_states).to(x.dtype)


@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(shape, dtype):
    device = cfg.device
    N = shape[-1]
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)
    x = torch.randn(shape, dtype=dtype, device=device)

    ref = _torch_gemma_rms_norm(x, module.weight.data, module.variance_epsilon)
    res = flaggems_sglang.gemma_rms_norm(module, x)

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm_with_residual(shape, dtype):
    device = cfg.device
    N = shape[-1]
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)
    x = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)

    x_ref = (x + residual).clone()
    ref_out = _torch_gemma_rms_norm(x_ref, module.weight.data, module.variance_epsilon)

    out, res_out = flaggems_sglang.gemma_rms_norm(module, x.clone(), residual.clone())

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)
    torch.testing.assert_close(res_out, x_ref, atol=atol, rtol=1e-2)
