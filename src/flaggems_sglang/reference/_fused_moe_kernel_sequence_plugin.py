# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch oracle for the documented _fused_moe_kernel_sequence interface."""

from .operator_oracles import (  # noqa: F401
    _fused_moe_kernel_sequence as reference,
)
