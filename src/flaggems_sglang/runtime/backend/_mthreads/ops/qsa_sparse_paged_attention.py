# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MUSA sparse QSA launch geometry.

A single pipeline stage limits shared memory for the 256-wide Qwen3.8 head.
The framework owns optional split workspaces; this backend does not select
an NVIDIA split policy or allocate persistent scratch buffers.
"""

from __future__ import annotations

import torch

from flaggems_sglang.ops.qsa_sparse_paged_attention import (
    _qsa_sparse_paged_attention,
)


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    split_workspace: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None,
) -> torch.Tensor:
    """Run paged sparse QSA with the S5000 launch geometry."""
    return _qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        softmax_scale,
        out,
        gate,
        split_workspace,
        min_block_m=32,
        launch_config=(64, 8, 1),
    )


__all__ = ["qsa_sparse_paged_attention"]
