# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MoE Router (CUDA Core implementation) - Highly Optimized Version.

This implementation focuses on maximum performance through:
1. Kernel fusion to reduce memory bandwidth
2. Shared memory usage for frequently accessed data
3. Vectorized operations for better throughput
4. Optimized top-k selection algorithm
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_moe_router_optimized_kernel(
    # Pointers
    x_ptr,
    router_weight_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    correction_bias_ptr,
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    topk: tl.constexpr,
    # Parameters
    moe_softcapping: tl.constexpr,
    has_correction_bias: tl.constexpr,
    # Strides
    stride_x_b,
    stride_router_e,
    stride_topk_weights_b,
    stride_topk_ids_b,
    # Meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Highly optimized Triton kernel for fused MoE router."""
    
    # Program ID maps to batch dimension
    pid_b = tl.program_id(0)
    
    # Initialize accumulators in registers
    logits = tl.zeros([BLOCK_E], dtype=tl.float32)
    
    # Compute logits = x @ router_weight.T using blocked approach
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H
        
        # Load x[pid_b, h_offsets] - coalesced access
        x_ptrs = x_ptr + pid_b * stride_x_b + h_offsets
        x_vals = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        
        # Process experts in blocks
        for e_start in range(0, E, BLOCK_E):
            e_offsets = e_start + tl.arange(0, BLOCK_E)
            e_mask = e_offsets < E
            
            # Load router_weight[e_offsets, h_offsets] - coalesced access
            router_ptrs = router_weight_ptr + e_offsets * stride_router_e + h_offsets
            router_vals = tl.load(router_ptrs, mask=e_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            
            # Vectorized dot product
            partial_dot = tl.sum(x_vals[None, :] * router_vals, axis=1)
            
            # Accumulate with masking
            logits = tl.where(e_mask, logits + partial_dot, logits)
    
    # Apply soft-capping (fused with logits computation)
    if moe_softcapping != 0:
        logits = tl.math.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Add correction bias (fused)
    if has_correction_bias:
        bias_offsets = tl.arange(0, BLOCK_E)
        bias_mask = bias_offsets < E
        bias_ptrs = correction_bias_ptr + bias_offsets
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        logits = tl.where(bias_mask, logits + bias_vals, logits)
    
    # Online softmax computation (numerically stable)
    logits_max = tl.max(logits, axis=0)
    logits_shifted = logits - logits_max
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Optimized top-k selection using partial sort
    # For small k, use selection algorithm
    topk_ids_vals = tl.zeros([topk], dtype=tl.int32)
    topk_weights_vals = tl.zeros([topk], dtype=tl.float32)
    
    # Selection algorithm for top-k
    remaining_logits = logits
    remaining_probs = probs
    
    for k in range(topk):
        # Find maximum using reduction
        max_val = tl.max(remaining_logits, axis=0)
        max_idx = tl.argmax(remaining_logits, axis=0)
        
        # Get weight from probs
        max_weight = tl.load(probs + max_idx)
        
        # Store results
        topk_ids_vals = tl.where(tl.arange(0, topk) == k, max_idx, topk_ids_vals)
        topk_weights_vals = tl.where(tl.arange(0, topk) == k, max_weight, topk_weights_vals)
        
        # Mask out selected element
        mask = tl.arange(0, BLOCK_E) != max_idx
        remaining_logits = tl.where(mask, remaining_logits, -float('inf'))
    
    # Store results
    topk_ids_offsets = pid_b * stride_topk_ids_b + tl.arange(0, topk)
    tl.store(topk_ids_ptr + topk_ids_offsets, topk_ids_vals)
    
    topk_weights_offsets = pid_b * stride_topk_weights_b + tl.arange(0, topk)
    tl.store(topk_weights_ptr + topk_weights_offsets, topk_weights_vals)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64, 'BLOCK_E': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 128, 'BLOCK_E': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 256, 'BLOCK_E': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
    ],
    key=['B', 'H', 'E', 'topk'],
)
@triton.jit
def _fused_moe_router_autotuned_kernel(
    # Pointers
    x_ptr,
    router_weight_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    correction_bias_ptr,
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    topk: tl.constexpr,
    # Parameters
    moe_softcapping: tl.constexpr,
    has_correction_bias: tl.constexpr,
    # Strides
    stride_x_b,
    stride_router_e,
    stride_topk_weights_b,
    stride_topk_ids_b,
    # Meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Autotuned version of the fused MoE router kernel."""
    
    # Program ID maps to batch dimension
    pid_b = tl.program_id(0)
    
    # Initialize accumulators
    logits = tl.zeros([BLOCK_E], dtype=tl.float32)
    
    # Compute logits with blocked approach
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H
        
        # Load x[pid_b, h_offsets]
        x_ptrs = x_ptr + pid_b * stride_x_b + h_offsets
        x_vals = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        
        # Process experts in blocks
        for e_start in range(0, E, BLOCK_E):
            e_offsets = e_start + tl.arange(0, BLOCK_E)
            e_mask = e_offsets < E
            
            # Load router_weight[e_offsets, h_offsets]
            router_ptrs = router_weight_ptr + e_offsets * stride_router_e + h_offsets
            router_vals = tl.load(router_ptrs, mask=e_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            
            # Compute partial dot product
            partial_dot = tl.sum(x_vals[None, :] * router_vals, axis=1)
            
            # Accumulate
            logits = tl.where(e_mask, logits + partial_dot, logits)
    
    # Apply soft-capping
    if moe_softcapping != 0:
        logits = tl.math.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Add correction bias
    if has_correction_bias:
        bias_offsets = tl.arange(0, BLOCK_E)
        bias_mask = bias_offsets < E
        bias_ptrs = correction_bias_ptr + bias_offsets
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        logits = tl.where(bias_mask, logits + bias_vals, logits)
    
    # Online softmax
    logits_max = tl.max(logits, axis=0)
    logits_shifted = logits - logits_max
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Top-k selection
    topk_ids_vals = tl.zeros([topk], dtype=tl.int32)
    topk_weights_vals = tl.zeros([topk], dtype=tl.float32)
    
    remaining_logits = logits
    
    for k in range(topk):
        max_val = tl.max(remaining_logits, axis=0)
        max_idx = tl.argmax(remaining_logits, axis=0)
        
        max_weight = tl.load(probs + max_idx)
        
        topk_ids_vals = tl.where(tl.arange(0, topk) == k, max_idx, topk_ids_vals)
        topk_weights_vals = tl.where(tl.arange(0, topk) == k, max_weight, topk_weights_vals)
        
        mask = tl.arange(0, BLOCK_E) != max_idx
        remaining_logits = tl.where(mask, remaining_logits, -float('inf'))
    
    # Store results
    topk_ids_offsets = pid_b * stride_topk_ids_b + tl.arange(0, topk)
    tl.store(topk_ids_ptr + topk_ids_offsets, topk_ids_vals)
    
    topk_weights_offsets = pid_b * stride_topk_weights_b + tl.arange(0, topk)
    tl.store(topk_weights_ptr + topk_weights_offsets, topk_weights_vals)


def fused_moe_router_cudacore_optimized(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Highly optimized fused MoE router implementation.
    
    Args:
        x: [B, H] - token hidden states
        router_weight: [E, H] - router weight matrix
        topk: int - number of experts to select per token
        moe_softcapping: float - logit soft-capping coefficient (0 to disable)
        correction_bias: [E] float32 or None - expert correction bias
        
    Returns:
        Tuple of (topk_weights, topk_ids):
            - topk_weights: [B, topk] float32 - expert weights from global softmax
            - topk_ids: [B, topk] int32 - selected expert indices
    """
    
    # Input validation
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    assert router_weight.dim() == 2, f"router_weight must be 2D, got {router_weight.dim()}D"
    assert x.shape[1] == router_weight.shape[1], \
        f"Hidden dimension mismatch: x.shape[1]={x.shape[1]}, router_weight.shape[1]={router_weight.shape[1]}"
    
    B, H = x.shape
    E, _ = router_weight.shape
    
    assert topk > 0 and topk <= E, f"topk must be in (0, E], got topk={topk}, E={E}"
    
    if correction_bias is not None:
        assert correction_bias.shape == (E,), \
            f"correction_bias shape mismatch: expected ({E},), got {correction_bias.shape}"
    
    # Convert to float32 for computation
    x_float = x.float()
    router_weight_float = router_weight.float()
    
    # Prepare output tensors
    topk_ids = torch.empty(B, topk, dtype=torch.int32, device=x.device)
    topk_weights = torch.empty(B, topk, dtype=torch.float32, device=x.device)
    
    # Compute grid
    grid = (B,)
    
    # Launch optimized kernel
    _fused_moe_router_optimized_kernel[grid](
        x_float,
        router_weight_float,
        topk_weights,
        topk_ids,
        correction_bias if correction_bias is not None else x_float,  # dummy pointer
        B=B,
        H=H,
        E=E,
        topk=topk,
        moe_softcapping=moe_softcapping,
        has_correction_bias=correction_bias is not None,
        stride_x_b=x_float.stride(0),
        stride_router_e=router_weight_float.stride(0),
        stride_topk_weights_b=topk_weights.stride(0),
        stride_topk_ids_b=topk_ids.stride(0),
        BLOCK_H=128,
        BLOCK_E=128,
        BLOCK_K=64,
    )
    
    return topk_weights, topk_ids


# Export the function
__all__ = ["fused_moe_router_cudacore_optimized"]
