# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Portable Torch mRoPE reference; NVIDIA dispatch retains native Triton."""

from .mrotary_embedding_cases import (  # noqa: F401
    WORKLOADS,
    baseline_key,
    build_inputs,
    case_parameters,
    describe_inputs,
)
from .mrotary_embedding_torch import reference  # noqa: F401
