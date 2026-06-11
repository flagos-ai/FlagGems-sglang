import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
    triton_kernel_fused_experts as _sglang_fused_experts,
)
from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.topk import topk as triton_kernels_topk

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg


def _routing(logits, n_expts_act):
    """Build routing data from logits using triton_kernels topk."""
    from triton_kernels.tensor import make_ragged_tensor_metadata

    sparse_logits = triton_kernels_topk(
        logits, n_expts_act, apply_softmax=True
    )
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_metadata.slice_sizes,
        logits.shape[-1],
        n_expts_act,
        ragged_metadata,
    )
    gather_indx = GatherIndx(combine_indx, dispatch_indx)
    scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_indx, scatter_indx


def _make_inputs(M, K, N, E, topk, device="cuda"):
    """Generate test inputs for fused_moe."""
    torch.manual_seed(M + K + N + E + topk)
    hidden_states = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(E, K, N * 2, device=device, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_data, gather_indx, scatter_indx = _routing(logits, topk)
    return hidden_states, w1, w2, routing_data, gather_indx, scatter_indx


def _ref_fused_experts(
    hidden_states,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    activation="silu",
    apply_router_weight_on_input=False,
):
    """Reference: delegate to SGLang's triton_kernel_fused_experts."""
    return _sglang_fused_experts(
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_indx,
        scatter_indx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )


@pytest.mark.parametrize(
    "shape", utils.FUSED_MOE_SHAPES, ids=utils.FUSED_MOE_SHAPE_IDS
)
@pytest.mark.fused_moe
def test_fused_moe(shape):
    device = cfg.device
    M, K, N, E, topk = shape
    inputs = _make_inputs(M, K, N, E, topk, device=device)

    ref = _ref_fused_experts(*inputs)
    res = flaggems_sglang.triton_kernel_fused_experts(*inputs)

    torch.testing.assert_close(res, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape", utils.FUSED_MOE_SHAPES_SMALL, ids=utils.FUSED_MOE_SMALL_IDS
)
@pytest.mark.fused_moe
def test_fused_moe_input_gate(shape):
    device = cfg.device
    M, K, N, E, topk = shape
    inputs = _make_inputs(M, K, N, E, topk, device=device)

    ref = _ref_fused_experts(*inputs, apply_router_weight_on_input=True)
    res = flaggems_sglang.triton_kernel_fused_experts(
        *inputs, apply_router_weight_on_input=True
    )

    torch.testing.assert_close(res, ref, rtol=0, atol=0)
