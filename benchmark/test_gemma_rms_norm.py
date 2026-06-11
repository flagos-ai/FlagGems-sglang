import pytest
import torch
import torch.nn as nn

import flaggems_sglang

from .attri_util import FLOAT_DTYPES, GEMMA_RMS_NORM_BENCH_SHAPES


class _MockGemmaRMSNorm(nn.Module):
    """Minimal mock of torch.nn.Module with the attributes expected by
    our gemma_rms_norm API."""

    def __init__(
        self, hidden_size, eps=1e-6, dtype=torch.float32, device="cpu"
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps


@pytest.mark.parametrize("shape", GEMMA_RMS_NORM_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)
    benchmark(flaggems_sglang.gemma_rms_norm, module, x)


@pytest.mark.parametrize("shape", GEMMA_RMS_NORM_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm_res
def test_gemma_rms_norm_with_residual(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    module = _MockGemmaRMSNorm(N, eps=1e-6, dtype=dtype, device=device)

    def run():
        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        return flaggems_sglang.gemma_rms_norm(module, x, residual)

    benchmark(run)
