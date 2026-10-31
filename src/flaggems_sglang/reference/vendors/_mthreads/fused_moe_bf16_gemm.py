# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent upstream BF16 GEMM plus separately materialized SwiGLU.

The reference uses SGLang's existing unfused GEMM API, never the migrated
kernel or its tuning helpers. ``config`` describes the reference launch.
FP32 Torch GEMM can round different intermediate BF16 values near a rounding
boundary; those errors are amplified by negative-gate SwiGLU. This baseline
keeps the actual SGLang BF16 accumulation/materialization contract.
"""

import torch
import triton.language as tl


def reference(
    a,
    b,
    out,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    config,
    *,
    top_k,
    mul_routed_weight=False,
    fuse_swiglu=False,
    tuning_profile=None,
):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    full_output = torch.empty(
        (*out.shape[:-1], b.shape[1]), dtype=a.dtype, device=a.device
    )
    # This upstream launcher consumes only topk_ids.numel(), while the actual
    # routing is given independently by sorted_token_ids and expert_ids.
    topk_ids = torch.empty_like(topk_weights, dtype=torch.int32)
    invoke_fused_moe_kernel(
        a,
        b,
        None,
        full_output,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        dict(config),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        filter_expert=False,
    )
    if fuse_swiglu:
        gate, up = full_output.chunk(2, dim=-1)
        return (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(
            a.dtype
        )
    return full_output
