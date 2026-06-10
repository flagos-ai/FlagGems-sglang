import pytest
import torch

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg

# Reference: SGLang's compiled Triton kernel as correctness baseline.
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as _sglang_fn,
)

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


def _make_inputs(shape, dtype, device):
    """Create input tensors for a single test configuration.

    Args:
        shape: (B, H, HV, K, V, pool_size) tuple.
        dtype: Tensor data type.
        device: Target device.

    Returns:
        tuple of inputs for the operator.
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
        ssm_state_indices,
    )


def _ref_fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    """Reference: delegate to SGLang's compiled Triton kernel."""
    return _sglang_fn(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@pytest.mark.parametrize("shape", utils.FUSED_RECURRENT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_fused_recurrent_gated_delta_rule_packed_decode(shape, dtype):
    device = cfg.device

    (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        ssm_state_indices,
    ) = _make_inputs(shape, dtype, device)

    B = shape[0]
    HV = shape[2]
    V = shape[4]

    # Allocate output buffers
    out_ref = mixed_qkv.new_empty(B, 1, HV, V)
    out_res = mixed_qkv.new_empty(B, 1, HV, V)
    state_ref = initial_state.clone()
    state_res = initial_state.clone()

    # Reference (SGLang)
    _ref_fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_ref,
        out=out_ref,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Optimized (FlagGems-sglang Triton)
    flaggems_sglang.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_res,
        out=out_res,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Compare output tensors
    atol = 2e-2
    rtol = 1e-2
    torch.testing.assert_close(out_res, out_ref, atol=atol, rtol=rtol)

    # Compare state tensors at active indices
    indices = ssm_state_indices.long()
    torch.testing.assert_close(
        state_res[indices], state_ref[indices], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("shape", utils.FUSED_RECURRENT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_fused_recurrent_gated_delta_rule_packed_decode_negative_indices(
    shape, dtype
):
    """Test that padding tokens (state_idx == -1) produce zero output without
    modifying state. Requires B >= 2 so half the batch can be masked."""
    device = cfg.device
    B = shape[0]
    if B < 2:
        pytest.skip("Need B >= 2 for negative index test")

    (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        ssm_state_indices,
    ) = _make_inputs(shape, dtype, device)

    HV, V = shape[2], shape[4]

    # Mix valid and invalid (PAD_SLOT_ID = -1) indices
    ssm_state_indices[B // 2 :] = -1

    out_ref = mixed_qkv.new_empty(B, 1, HV, V)
    out_res = mixed_qkv.new_empty(B, 1, HV, V)
    state_ref = initial_state.clone()
    state_res = initial_state.clone()

    _ref_fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_ref,
        out=out_ref,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    flaggems_sglang.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_res,
        out=out_res,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_res, out_ref, atol=2e-2, rtol=1e-2)
