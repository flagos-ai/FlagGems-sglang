"""Fused MoE — Ascend NPU CANN native implementation.

Provides fused_moe_ascend: CANN native fused MoE forward pass using
npu_moe_init_routing_v2, npu_grouped_matmul, npu_swiglu, and
npu_moe_finalize_routing.
"""

from __future__ import annotations

import torch
from sglang.srt.layers.moe.token_dispatcher import standard
from sglang.srt.layers.moe.topk import TopKOutputChecker


def fused_moe_ascend(
    obj,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_output,
):
    """Ascend NPU path using CANN native ops (torch.ops.npu.*).

    Uses npu_moe_init_routing_v2, npu_grouped_matmul, npu_swiglu,
    npu_moe_finalize_routing instead of triton kernels.
    """
    if TopKOutputChecker.format_is_triton_kernels(topk_output):
        raise RuntimeError(
            "fused_moe Ascend CANN path does not support triton_kernels "
            "topk format; use StandardTopKOutput instead."
        )
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    num_tokens = hidden_states.shape[0]
    num_experts = layer.num_experts
    top_k = layer.top_k or topk_ids.shape[1]
    original_dtype = hidden_states.dtype

    topk_weights_fp = topk_weights.to(original_dtype)
    topk_ids_i32 = topk_ids.to(torch.int32)

    # routing init
    (
        x,
        expanded_row_idx,
        expert_tokens,
        _,
    ) = torch.ops.npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids_i32,
        active_num=num_tokens * top_k,
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=-1,
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # bias
    w13_bias = None
    w2_bias = None
    if getattr(layer, "w13_weight_bias", None) is not None:
        w13_bias = [layer.w13_weight_bias]
    if getattr(layer, "w2_weight_bias", None) is not None:
        w2_bias = [layer.w2_weight_bias]

    # gmm1: gate+up projection
    x = torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[layer.w13_weight],
        bias=w13_bias,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation
    activation = obj.runner.config.activation
    if activation == "npu_swiglu_oai":
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

        x = swiglu_oai(layer, x)
    elif activation == "silu":
        x = torch.ops.npu.npu_swiglu(x)
    else:
        from sglang.srt.layers.activation import GeluAndMul

        x = GeluAndMul()(x)

    # gmm2: down projection
    x = torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[layer.w2_weight],
        bias=w2_bias,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # finalize routing (scatter+reduce)
    output = torch.ops.npu.npu_moe_finalize_routing(
        x,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights_fp,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids_i32,
        drop_pad_mode=2,
    )

    # post-processing
    config = obj.runner.config
    if config.no_combine:
        output = output.view(num_tokens, top_k, hidden_states.shape[-1])

    if (
        config.routed_scaling_factor is not None
        and config.routed_scaling_factor != 1.0
        and not config.no_combine
    ):
        output.mul_(config.routed_scaling_factor)

    return standard.StandardCombineInput(hidden_states=output)
