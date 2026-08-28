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

"""Fused MoE Router (CUDA Core implementation) - Optimized Version.

Implements fused MoE router with optional logit soft-capping and expert
correction bias, followed by global softmax and top-k expert selection.
Uses optimized Triton kernels for matrix multiplication and top-k.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _softmax_kernel(
    logits_ptr,
    probs_ptr,
    stride_logits_b,
    stride_probs_b,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Triton kernel for softmax computation."""
    pid_b = tl.program_id(0)
    
    # Load logits
    logits_offsets = pid_b * stride_logits_b + tl.arange(0, BLOCK_E)
    logits_mask = tl.arange(0, BLOCK_E) < E
    logits = tl.load(logits_ptr + logits_offsets, mask=logits_mask, other=-float('inf')).to(tl.float32)
    
    # Compute softmax with numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Store probs
    probs_offsets = pid_b * stride_probs_b + tl.arange(0, BLOCK_E)
    tl.store(probs_ptr + probs_offsets, probs, mask=logits_mask)


@triton.jit
def _topk_kernel(
    logits_ptr,
    probs_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_logits_b,
    stride_probs_b,
    stride_topk_ids_b,
    stride_topk_weights_b,
    E: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Triton kernel for top-k selection."""
    pid_b = tl.program_id(0)
    
    # Load logits and probs
    logits_offsets = pid_b * stride_logits_b + tl.arange(0, BLOCK_E)
    logits_mask = tl.arange(0, BLOCK_E) < E
    logits = tl.load(logits_ptr + logits_offsets, mask=logits_mask, other=-float('inf')).to(tl.float32)
    probs = tl.load(probs_ptr + pid_b * stride_probs_b + tl.arange(0, BLOCK_E), 
                    mask=logits_mask, other=0.0).to(tl.float32)
    
    # Initialize output arrays
    topk_ids_vals = tl.zeros([topk], dtype=tl.int32)
    topk_weights_vals = tl.zeros([topk], dtype=tl.float32)
    
    # Select top-k elements
    remaining_logits = logits
    remaining_probs = probs
    
    for k in range(topk):
        # Find argmax
        max_idx = tl.argmax(remaining_logits, axis=0)
        
        # Get the weight from probs
        max_weight = tl.load(probs_ptr + pid_b * stride_probs_b + max_idx)
        
        # Store results
        topk_ids_vals = tl.where(tl.arange(0, topk) == k, max_idx, topk_ids_vals)
        topk_weights_vals = tl.where(tl.arange(0, topk) == k, max_weight, topk_weights_vals)
        
        # Mask out the selected index for next iteration
        mask = tl.arange(0, BLOCK_E) != max_idx
        remaining_logits = tl.where(mask, remaining_logits, -float('inf'))
    
    # Store results
    topk_ids_offsets = pid_b * stride_topk_ids_b + tl.arange(0, topk)
    tl.store(topk_ids_ptr + topk_ids_offsets, topk_ids_vals)
    
    topk_weights_offsets = pid_b * stride_topk_weights_b + tl.arange(0, topk)
    tl.store(topk_weights_ptr + topk_weights_offsets, topk_weights_vals)


def fused_moe_router_cudacore_v2(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE router implementation (optimized version).
    
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
    
    # Step 1: Compute logits = x @ router_weight.T
    logits = torch.mm(x_float, router_weight_float.t())  # [B, E]
    
    # Step 2: Apply soft-capping
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Step 3: Add correction bias
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    
    # Step 4: Compute softmax using Triton kernel
    probs = torch.empty_like(logits)
    BLOCK_E = triton.next_power_of_2(E)
    
    grid = (B,)
    _softmax_kernel[grid](
        logits,
        probs,
        logits.stride(0),
        probs.stride(0),
        E=E,
        BLOCK_E=BLOCK_E,
    )
    
    # Step 5: Top-k selection using Triton kernel
    topk_ids = torch.empty(B, topk, dtype=torch.int32, device=x.device)
    topk_weights = torch.empty(B, topk, dtype=torch.float32, device=x.device)
    
    _topk_kernel[grid](
        logits,
        probs,
        topk_ids,
        topk_weights,
        logits.stride(0),
        probs.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        E=E,
        topk=topk,
        BLOCK_E=BLOCK_E,
    )
    
    return topk_weights, topk_ids


# Export the function
__all__ = ["fused_moe_router_cudacore_v2"]
