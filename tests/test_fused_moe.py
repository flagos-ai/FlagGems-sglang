import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
    triton_kernel_fused_experts as _sglang_fused_experts,
)
from triton_kernels.matmul_ogs import GatherIndx, ScatterIndx
from triton_kernels.routing import RoutingData, compute_expt_data_torch

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg


def _routing(logits, n_expts_act):
    """Build routing data from logits using standard topk conversion.

    Uses the same approach as _standard_topk_to_triton_kernels in fused_moe.py
    to remain compatible with the installed triton_kernels version.
    """
    n_tokens, n_expts_tot = logits.shape

    # Compute topk weights and ids from logits
    topk_weights = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, k=n_expts_act, dim=-1)
    # Normalize
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Convert to triton_kernels routing format
    n_gates_pad = n_tokens * n_expts_act
    expt_indx_2d, sort_indices = torch.sort(topk_ids, dim=1)
    expt_scal_2d = torch.gather(topk_weights, 1, sort_indices)
    expt_scal = expt_scal_2d.reshape(-1)
    expt_indx = expt_indx_2d.reshape(-1).to(torch.int32)
    topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
    gate_indx = torch.argsort(topk_indx, stable=True).to(torch.int32)
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(
        expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1
    ).int()
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
    routing_data = RoutingData(
        gate_scal, hist, n_expts_tot, n_expts_act, expt_data
    )
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
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
