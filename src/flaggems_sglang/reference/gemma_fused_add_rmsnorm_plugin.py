# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch oracle for the documented gemma_fused_add_rmsnorm_plugin interface."""

from .operator_oracles import gemma_fused_add_rmsnorm

reference = gemma_fused_add_rmsnorm
