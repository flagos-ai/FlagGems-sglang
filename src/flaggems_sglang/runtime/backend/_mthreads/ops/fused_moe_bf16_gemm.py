# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical SGLang BF16 expert GEMM, optionally fusing the W13 SwiGLU tail.

Routing metadata and launch configuration are explicit inputs. ``out`` is
caller-owned and must not overlap any input. No communication or persistent
allocation occurs here. Only the non-quantized, non-TMA SGLang layout is
supported; unsupported modes must be selected out before the first GEMM.
"""

import torch
import triton
import triton.language as tl

from ._fused_moe_bf16 import fused_moe_kernel
from ._tuning import moe_config


def fused_moe_bf16_gemm(
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
    if a.ndim != 2 or b.ndim != 3 or out.ndim != 3:
        raise ValueError("expected 2D activations, 3D weights and output")
    if top_k <= 0 or topk_weights.ndim != 2 or topk_weights.shape[1] <= 0:
        raise ValueError("top-k must be positive")
    n = b.shape[1] // 2 if fuse_swiglu else b.shape[1]
    tokens, routes = topk_weights.shape
    if a.shape[0] != tokens * routes // top_k or a.shape[1] != b.shape[2]:
        raise ValueError("activation/routing/weight shape mismatch")
    if out.shape != (tokens, routes, n):
        raise ValueError("output must have shape [tokens, routes, columns]")
    tensors = (
        a,
        b,
        out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
    if a.device.type != "musa" or any(t.device != a.device for t in tensors):
        raise ValueError("all inputs must be on one MUSA device")
    if any(t.dtype != torch.bfloat16 for t in (a, b, out)):
        raise ValueError("BF16 GEMM requires BF16 activations/weights/output")
    if topk_weights.dtype != torch.float32:
        raise ValueError("routing weights must be FP32")
    if any(t.dtype != torch.int32 for t in tensors[4:]):
        raise ValueError("routing indices and padded count must be int32")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("BF16 expert GEMM requires contiguous tensors")
    if any(torch._C._overlaps(out, t) for t in tensors if t is not out):
        raise ValueError("GEMM output must not alias its inputs")
    if fuse_swiglu and (b.shape[1] % 2 or mul_routed_weight):
        raise ValueError("fused W13 requires even width and unweighted input")
    config = moe_config(
        config,
        tuning_profile,
        tokens=tokens,
        b_shape=b.shape,
        top_k=top_k,
        fuse_swiglu=fuse_swiglu,
        mul_routed_weight=mul_routed_weight,
    )
    if set(config) - {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "enable_backend_opt",
    }:
        raise ValueError("unsupported expert GEMM launch configuration")
    for key in (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
    ):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid expert GEMM configuration: {key}")
    if (
        "enable_backend_opt" in config
        and type(config["enable_backend_opt"]) is not bool
    ):
        raise ValueError("enable_backend_opt must be a boolean")
    bm, bn = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    # The copied BF16 kernel wraps N on weight loads. Require complete tiles
    # for the fused half-width path so its up-half load cannot pass the end.
    if n % bn or sorted_token_ids.ndim != 1:
        raise ValueError("output must contain complete column tiles")
    # SGLang may overallocate a non-aligned routing capacity. The device-side
    # padded count limits active blocks before any routing index is read.
    if expert_ids.ndim != 1 or expert_ids.numel() < triton.cdiv(
        sorted_token_ids.numel(), bm
    ):
        raise ValueError("expert IDs must cover every padded routing block")
    if num_tokens_post_padded.numel() != 1:
        raise ValueError("padded token count must contain one element")
    if tokens == 0:
        return out
    with torch.musa.device(a.device):
        fused_moe_kernel[
            (triton.cdiv(sorted_token_ids.numel(), bm) * triton.cdiv(n, bn),)
        ](
            a,
            b,
            out,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            b.shape[1],
            b.shape[2],
            sorted_token_ids.numel(),
            topk_weights.numel(),
            *a.stride(),
            b.stride(0),
            b.stride(2),
            b.stride(1),
            out.stride(1),
            out.stride(2),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=tl.bfloat16,
            even_Ks=b.shape[2] % config["BLOCK_SIZE_K"] == 0,
            FUSE_SWIGLU=fuse_swiglu,
            **config,
        )
    return out


__all__ = ["fused_moe_bf16_gemm"]
