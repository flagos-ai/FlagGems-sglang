import pytest
import torch

import flaggems_sglang

from .attri_util import FLOAT_DTYPES, GEMMA_RMS_NORM_BENCH_SHAPES


@pytest.mark.parametrize("shape", GEMMA_RMS_NORM_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    weight = torch.zeros(N, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)
    benchmark(flaggems_sglang.gemma_rms_norm, x, weight, 1e-6)


@pytest.mark.parametrize("shape", GEMMA_RMS_NORM_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.gemma_rms_norm_res
def test_gemma_rms_norm_with_residual(shape, dtype, benchmark):
    M, N = shape
    device = flaggems_sglang.device
    weight = torch.zeros(N, dtype=dtype, device=device)

    def run():
        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        return flaggems_sglang.gemma_rms_norm(x, weight, eps=1e-6, residual=residual)

    benchmark(run)
