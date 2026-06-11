import pytest
import torch

import flaggems_sglang

from .attri_util import FLOAT_DTYPES, FUSED_RECURRENT_BENCH_SHAPES


def _make_inputs(shape, dtype, device):
    """Create input tensors for a benchmark configuration.

    Args:
        shape: (B, H, HV, K, V, pool_size) tuple.
        dtype: Tensor data type.
        device: Target device.
    """
    B, H, HV, K, V, pool_size = shape
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype)
    b = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=dtype)
    dt_bias = torch.randn(HV, device=device, dtype=dtype)
    initial_state = (
        torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.1
    )
    out = mixed_qkv.new_empty(B, 1, HV, V)
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)
    scale = K**-0.5
    return (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
    )


@pytest.mark.parametrize("shape", FUSED_RECURRENT_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_fused_recurrent_gated_delta_rule_packed_decode(
    shape, dtype, benchmark
):
    device = flaggems_sglang.device

    (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
    ) = _make_inputs(shape, dtype, device)

    def run():
        state = initial_state.clone()
        return flaggems_sglang.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    benchmark(run)
