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

"""Fused MoE Router (CUDA Core implementation) - PyTorch Reference.

PyTorch reference implementation for fused MoE router with optional
logit soft-capping and expert correction bias.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def fused_moe_router_cudacore_torch(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE router implementation (PyTorch reference).
    
    This is the reference implementation from the competition problem statement.
    
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
    
    # Step 1: Compute logits
    logits = x.float() @ router_weight.float().t()
    
    # Step 2: Apply soft-capping
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Step 3: Add correction bias
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    
    # Step 4: Compute softmax
    probs = torch.softmax(logits, dim=-1)
    
    # Step 5: Top-k selection
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.gather(probs, -1, topk_ids)
    
    return topk_weights, topk_ids.to(torch.int32)


# Export the function
__all__ = ["fused_moe_router_cudacore_torch"]
