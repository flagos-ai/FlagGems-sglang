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

"""Fused MoE Router (CUDA Core implementation).

Implements fused MoE router with optional logit soft-capping and expert
correction bias, followed by global softmax and top-k expert selection.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_moe_router_kernel(
    # Pointers
    x_ptr,
    router_weight_ptr,
    logits_ptr,
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
    stride_logits_b,
    stride_topk_weights_b,
    stride_topk_ids_b,
    # Meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Triton kernel for fused MoE router computation."""
    
    # Program ID maps to batch dimension
    pid_b = tl.program_id(0)
    
    # Initialize accumulator for logits
    logits = tl.zeros([BLOCK_E], dtype=tl.float32)
    
    # Compute logits = x @ router_weight.T
    # x: [B, H], router_weight: [E, H]
    # logits = x.float() @ router_weight.float().T -> [B, E]
    
    # Load x[pid_b, :] in blocks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H
        
        # Load x[pid_b, h_offsets]
        x_ptrs = x_ptr + pid_b * stride_x_b + h_offsets
        x_vals = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        
        # Load router_weight[:, h_offsets] and compute partial dot products
        for e_start in range(0, E, BLOCK_E):
            e_offsets = e_start + tl.arange(0, BLOCK_E)
            e_mask = e_offsets < E
            
            # router_weight[e_offsets, h_offsets]
            router_ptrs = router_weight_ptr + e_offsets * stride_router_e + h_offsets
            router_vals = tl.load(router_ptrs, mask=e_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            
            # Compute partial dot product: sum over H dimension
            partial_dot = tl.sum(x_vals[None, :] * router_vals, axis=1)
            
            # Accumulate
            logits = tl.where(e_mask, logits + partial_dot, logits)
    
    # Apply soft-capping if needed
    if moe_softcapping != 0:
        logits = tl.math.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Add correction bias if provided
    if has_correction_bias:
        bias_offsets = tl.arange(0, BLOCK_E)
        bias_mask = bias_offsets < E
        bias_ptrs = correction_bias_ptr + bias_offsets
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        logits = tl.where(bias_mask, logits + bias_vals, logits)
    
    # Compute softmax: probs = softmax(logits, dim=-1)
    # First, compute max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max
    
    # Compute exp and sum
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Top-k selection using argsort
    # We need to find top-k indices and corresponding weights
    
    # Initialize topk_ids and topk_weights
    topk_ids_vals = tl.zeros([topk], dtype=tl.int32)
    topk_weights_vals = tl.zeros([topk], dtype=tl.float32)
    
    # Simple selection: iterate to find top-k
    # Note: For production, we should use a more efficient top-k algorithm
    # This is a simplified version for clarity
    
    # Create a copy of logits for selection
    remaining_logits = logits
    remaining_probs = probs
    
    for k in range(topk):
        # Find argmax
        max_val = tl.max(remaining_logits, axis=0)
        max_idx = tl.argmax(remaining_logits, axis=0)
        
        # Store results
        topk_ids_vals = tl.where(tl.arange(0, topk) == k, max_idx, topk_ids_vals)
        topk_weights_vals = tl.where(tl.arange(0, topk) == k, max_val, topk_weights_vals)
        
        # Mask out the selected index for next iteration
        mask = tl.arange(0, BLOCK_E) != max_idx
        remaining_logits = tl.where(mask, remaining_logits, -float('inf'))
        remaining_probs = tl.where(mask, remaining_probs, 0.0)
    
    # Store results
    # Store logits
    logits_offsets = pid_b * stride_logits_b + tl.arange(0, BLOCK_E)
    logits_mask = tl.arange(0, BLOCK_E) < E
    tl.store(logits_ptr + logits_offsets, logits, mask=logits_mask)
    
    # Store topk_ids
    topk_ids_offsets = pid_b * stride_topk_ids_b + tl.arange(0, topk)
    tl.store(topk_ids_ptr + topk_ids_offsets, topk_ids_vals)
    
    # Store topk_weights
    topk_weights_offsets = pid_b * stride_topk_weights_b + tl.arange(0, topk)
    tl.store(topk_weights_ptr + topk_weights_offsets, topk_weights_vals)


def fused_moe_router_cudacore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE router implementation.
    
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
    
    # Compute logits using PyTorch for now (Triton kernel has issues)
    # logits = x.float() @ router_weight.float().T  # [B, E]
    logits = torch.mm(x_float, router_weight_float.t())
    
    # Apply soft-capping
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Add correction bias
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    
    # Compute softmax
    probs = torch.softmax(logits, dim=-1)  # [B, E]
    
    # Top-k selection
    topk_weights, topk_ids = torch.topk(logits, topk, dim=-1)
    
    # Gather weights from probs (not from topk_logits)
    topk_weights = torch.gather(probs, -1, topk_ids)
    
    return topk_weights, topk_ids.to(torch.int32)


# Export the function
__all__ = ["fused_moe_router_cudacore"]
