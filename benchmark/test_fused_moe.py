import pytest
import torch
from triton_kernels.matmul_ogs import GatherIndx, ScatterIndx
from triton_kernels.routing import RoutingData, compute_expt_data_torch

import flaggems_sglang

from .attri_util import FUSED_MOE_BENCH_SHAPES


def _routing(logits, n_expts_act):
    """Build routing data compatible with installed triton_kernels."""
    n_tokens, n_expts_tot = logits.shape

    topk_weights = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, k=n_expts_act, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

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


@pytest.mark.parametrize("shape", FUSED_MOE_BENCH_SHAPES)
@pytest.mark.fused_moe
def test_fused_moe(shape, benchmark):
    M, K, N, E, topk = shape
    device = flaggems_sglang.device

    hidden_states = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(E, K, N * 2, device=device, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_data, gather_indx, scatter_indx = _routing(logits, topk)

    benchmark(
        flaggems_sglang.triton_kernel_fused_experts,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_indx,
        scatter_indx,
    )
