# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference dispatch: selects vendor-specific or default reference impl.

Mechanism (mirrors runtime/backend dispatch for ops):
1. Try to load from
   ``flaggems_sglang.reference.vendors._<vendor_name>.<op_name>``
2. Fall back to ``flaggems_sglang.reference.<op_name>``
   (default, typically nvidia/torch)

Usage in tests:
    from flaggems_sglang.reference import get_reference
    reference = get_reference("mrope_fused")
    expected = reference(q, k, ...)
"""

import importlib
import logging
import os

from flaggems_sglang import runtime

logger = logging.getLogger(__name__)

_VENDOR_NAME = runtime.device.vendor_name
_REFERENCE_DIR = os.path.dirname(__file__)
_cache = {}


def get_reference(op_name: str):
    """Get the reference function for an operator, with vendor dispatch.

    Looks up vendor-specific reference first, falls back to default.

    Args:
        op_name: operator name (e.g. "mrope_fused", "silu_and_mul")

    Returns:
        The ``reference(...)`` callable for correctness checking.
    """
    if op_name in _cache:
        return _cache[op_name]

    vendor_module_path = (
        f"flaggems_sglang.reference.vendors._{_VENDOR_NAME}.{op_name}"
    )
    try:
        mod = importlib.import_module(vendor_module_path)
        ref_fn = getattr(mod, "reference")
        _cache[op_name] = ref_fn
        logger.info(
            f"[reference dispatch] {op_name}: using vendor "
            f"'{_VENDOR_NAME}' reference"
        )
        return ref_fn
    except (ModuleNotFoundError, AttributeError):
        pass

    default_module_path = f"flaggems_sglang.reference.{op_name}"
    try:
        mod = importlib.import_module(default_module_path)
        ref_fn = getattr(mod, "reference")
        _cache[op_name] = ref_fn
        logger.info(
            f"[reference dispatch] {op_name}: using default reference "
            f"(no vendor override)"
        )
        return ref_fn
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(
            f"No reference implementation found for '{op_name}' "
            f"(tried vendor '{_VENDOR_NAME}' and default)"
        ) from e


def get_tolerance(op_name: str) -> dict:
    """Get vendor-specific or default tolerance for an operator.

    Looks for a ``TOLERANCE`` dict in the reference module.
    Falls back to standard per-dtype defaults.
    """
    import torch

    _DEFAULT_TOLERANCE = {
        torch.float32: dict(atol=1e-4, rtol=1e-4),
        torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
        torch.float16: dict(atol=1e-2, rtol=1e-2),
    }

    vendor_module_path = (
        f"flaggems_sglang.reference.vendors._{_VENDOR_NAME}.{op_name}"
    )
    try:
        mod = importlib.import_module(vendor_module_path)
        if hasattr(mod, "TOLERANCE"):
            return mod.TOLERANCE
    except (ModuleNotFoundError, AttributeError):
        pass

    default_module_path = f"flaggems_sglang.reference.{op_name}"
    try:
        mod = importlib.import_module(default_module_path)
        if hasattr(mod, "TOLERANCE"):
            return mod.TOLERANCE
    except (ModuleNotFoundError, AttributeError):
        pass

    return _DEFAULT_TOLERANCE
