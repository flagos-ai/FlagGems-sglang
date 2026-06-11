"""Optimized Fused MoE (Mixture of Experts) Triton kernel.

Implements fused expert computation with custom Triton silu activation
that adaptively selects block sizes based on row count for optimal
performance across decode and prefill workloads.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl
from triton_kernels.matmul_ogs import matmul_ogs

try:
    from sglang.jit_kernel.activation import gelu_and_mul
except Exception:
    from sgl_kernel import gelu_and_mul

logger = logging.getLogger(__name__)

_UNSUPPORTED_FEATURE_MSG = "{name} is not supported"


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    block_n = tl.program_id(1)
    cols = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < half_size
    input_base = row * half_size * 2 + cols
    output_base = row * half_size + cols
    gate = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + input_base + half_size, mask=mask, other=0.0).to(
        tl.float32
    )
    out = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(output_ptr + output_base, out, mask=mask)


@triton.jit
def _silu_and_mul_sigmoid_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    block_n = tl.program_id(1)
    cols = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < half_size
    input_base = row * half_size * 2 + cols
    output_base = row * half_size + cols
    gate = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + input_base + half_size, mask=mask, other=0.0).to(
        tl.float32
    )
    out = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(output_ptr + output_base, out, mask=mask)


def _silu_and_mul_triton(
    intermediate: torch.Tensor, output: torch.Tensor
) -> None:
    """Adaptive Triton silu activation with row-count-based tiling."""
    rows = intermediate.numel() // intermediate.shape[-1]
    half_size = intermediate.shape[-1] // 2
    # Adaptive block/warp selection: wider blocks for small batch,
    # narrower blocks for larger batches to balance occupancy.
    if rows <= 128:
        block_n = 2048
        num_warps = 8
        kernel = _silu_and_mul_kernel
    elif rows <= 768:
        block_n = 1024
        num_warps = 4
        kernel = _silu_and_mul_sigmoid_kernel
    else:
        block_n = 1024
        num_warps = 4
        kernel = _silu_and_mul_sigmoid_kernel
    grid = (rows, triton.cdiv(half_size, block_n))
    kernel[grid](
        intermediate,
        output,
        rows,
        half_size,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def _standard_topk_to_triton_kernels(topk_weights, topk_ids, n_expts_tot):
    """Convert standard topk output format to triton_kernels routing."""
    from triton_kernels.routing import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
        compute_expt_data_torch,
    )

    n_tokens, n_expts_act = topk_weights.shape
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


def triton_kernel_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    logger.debug("FLAGGEMS_SGLANG FUSED_MOE")
    assert use_fp8_w8a8 is False, _UNSUPPORTED_FEATURE_MSG.format(
        name="use_fp8_w8a8"
    )
    assert per_channel_quant is False, _UNSUPPORTED_FEATURE_MSG.format(
        name="per_channel_quant"
    )
    assert expert_map is None, _UNSUPPORTED_FEATURE_MSG.format(
        name="expert_map"
    )
    assert w1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w1_scale")
    assert w2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w2_scale")
    assert a1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a1_scale")
    assert a2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a2_scale")
    assert block_shape is None, _UNSUPPORTED_FEATURE_MSG.format(
        name="block_shape"
    )
    assert inplace is False, "Inplace is not supported"

    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.dtype == torch.bfloat16
    ), "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"
    assert hidden_states.shape[-1] == w1.shape[-2], "hidden size mismatch"
    assert w2.shape[-1] == w1.shape[1], "intermediate size mismatch"

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size_twice = w1.shape[2]
    top_k = routing_data.n_expts_act
    gate_scal = routing_data.gate_scal
    input_gammas = gate_scal if apply_router_weight_on_input else None
    output_gammas = None if apply_router_weight_on_input else gate_scal

    intermediate = matmul_ogs(
        hidden_states,
        w1,
        None,
        routing_data,
        gather_indx=gather_indx,
        gammas=input_gammas,
    )

    activated = intermediate.new_empty(
        (num_tokens * top_k, inter_size_twice // 2)
    )
    if activation == "silu":
        _silu_and_mul_triton(intermediate, activated)
    elif activation == "gelu":
        gelu_and_mul(intermediate.view(-1, inter_size_twice), activated)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    output = matmul_ogs(
        activated,
        w2,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=output_gammas,
    )
    return output.view(num_tokens, hidden_size)


def triton_kernel_fused_experts_with_bias(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("FLAGGEMS_SGLANG FUSED_MOE_WITH_BIAS")
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.swiglu import swiglu_fn

    assert inplace is False, "Inplace is not supported"
    assert (
        hidden_states.dtype == torch.bfloat16
    ), "hidden_states must be bfloat16"

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    gate_scal = routing_data.gate_scal
    input_gammas = gate_scal if apply_router_weight_on_input else None
    output_gammas = None if apply_router_weight_on_input else gate_scal

    if activation == "silu":
        act = FusedActivation(swiglu_fn, FnSpecs(1, 1, 1))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    flex_ctx = FlexCtx()
    if gemm1_alpha is not None:
        alpha_data = InFlexData(
            torch.tensor(gemm1_alpha, dtype=torch.float32),
            flex_ctx,
        )
    else:
        alpha_data = None

    if gemm1_clamp_limit is not None:
        clamp_data = InFlexData(
            torch.tensor(gemm1_clamp_limit, dtype=torch.float32),
            flex_ctx,
        )
    else:
        clamp_data = None

    pcg1 = (
        w1_pcg if w1_pcg is not None else PrecisionConfig(w1.dtype, w1.dtype)
    )
    pcg2 = (
        w2_pcg if w2_pcg is not None else PrecisionConfig(w2.dtype, w2.dtype)
    )

    intermediate = matmul_ogs(
        hidden_states,
        w1,
        pcg1,
        routing_data,
        gather_indx=gather_indx,
        gammas=input_gammas,
        bias=b1,
        fused_activation=act,
        in_flex_data=alpha_data,
        clamp_data=clamp_data,
    )

    output = matmul_ogs(
        intermediate,
        w2,
        pcg2,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=output_gammas,
        bias=b2,
    )
    return output.view(num_tokens, hidden_size)


def fused_moe_flagos(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """SGLang integration wrapper for the fused MoE operator."""
    from sglang.srt.layers.moe.moe_runner.triton_kernels import (
        TritonKernelsQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    w13 = layer.w13_weight.transpose(-1, -2).contiguous()
    w2 = layer.w2_weight.transpose(-1, -2).contiguous()

    quant_info = TritonKernelsQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    if TopKOutputChecker.format_is_triton_kernels(topk_output):
        routing_data, gather_indx, scatter_indx = topk_output
    else:
        (
            routing_data,
            gather_indx,
            scatter_indx,
        ) = _standard_topk_to_triton_kernels(
            topk_output.topk_weights,
            topk_output.topk_ids,
            n_expts_tot=obj.runner.config.num_experts,
        )

    common_kwargs = dict(
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=(None if obj.runner.config.no_combine else scatter_indx),
        inplace=False,
        activation=obj.runner.config.activation,
        apply_router_weight_on_input=(
            obj.runner.config.apply_router_weight_on_input
        ),
        global_num_experts=quant_info.global_num_experts,
    )

    has_bias = (
        quant_info.w13_bias is not None or quant_info.w2_bias is not None
    )

    if has_bias:
        assert (
            quant_info.w13_bias is not None and quant_info.w2_bias is not None
        ), "Bias execution requires both w13_bias and w2_bias"
        output = triton_kernel_fused_experts_with_bias(
            hidden_states=hidden_states,
            w1=quant_info.w13_weight,
            w1_pcg=quant_info.w13_precision_config,
            b1=quant_info.w13_bias,
            w2=quant_info.w2_weight,
            w2_pcg=quant_info.w2_precision_config,
            b2=quant_info.w2_bias,
            gemm1_alpha=obj.runner.config.gemm1_alpha,
            gemm1_clamp_limit=obj.runner.config.gemm1_clamp_limit,
            **common_kwargs,
        )
    else:
        output = triton_kernel_fused_experts(
            hidden_states=hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            **common_kwargs,
        )

    if obj.runner.config.no_combine:
        tokens = dispatch_output.hidden_states.shape[0]
        hidden = dispatch_output.hidden_states.shape[-1]
        total_rows = output.shape[0]
        top_k = total_rows // tokens
        output = output.view(tokens, top_k, hidden)

    if (
        obj.runner.config.routed_scaling_factor is not None
        and obj.runner.config.routed_scaling_factor != 1.0
        and not obj.runner.config.no_combine
    ):
        output.mul_(obj.runner.config.routed_scaling_factor)

    return StandardCombineInput(hidden_states=output)
