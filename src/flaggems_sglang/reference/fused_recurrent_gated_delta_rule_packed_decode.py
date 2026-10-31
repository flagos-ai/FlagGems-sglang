# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch oracle for the documented fused_recurrent_gated_delta_rule_packed_decode interface."""

from .qwen36 import (  # noqa: F401
    fused_recurrent_gated_delta_rule_packed_decode as reference,
)
