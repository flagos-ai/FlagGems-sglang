import pytest
import torch
from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.tensor import make_ragged_tensor_metadata
from triton_kernels.topk import topk as triton_kernels_topk

import flaggems_sglang

from .attri_util import FUSED_MOE_BENCH_SHAPES


def _routing(logits, n_expts_act):
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
