import pytest
import torch
import torch.nn as nn

import flaggems_sglang

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Benchmark, Config


class _MockGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps


NORM_SHAPES = [(1, 512), (4, 1024), (32, 2048), (64, 4096), (128, 8192)]


def torch_op_no_residual(module, x):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + module.variance_epsilon)
    return ((1.0 + module.weight.to(torch.float32)) * hidden_states).to(x.dtype)


def torch_op_with_residual(module, x, residual):
    x = x + residual
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + module.variance_epsilon)
    return ((1.0 + module.weight.to(torch.float32)) * hidden_states).to(x.dtype)


@pytest.mark.parametrize(
    "shape", NORM_SHAPES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)
    benchmark(flaggems_sglang.gemma_rms_norm, module, x)


@pytest.mark.parametrize(
    "shape", NORM_SHAPES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm_with_residual(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)

    def run():
        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        return flaggems_sglang.gemma_rms_norm(module, x, residual)

    benchmark(run)
