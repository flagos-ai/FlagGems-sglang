# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import flaggems_sglang
from flaggems_sglang.reference import get_reference

if flaggems_sglang.vendor_name == "nvidia":
    # Preserve the portable Torch replay oracle on NVIDIA. The package's
    # NVIDIA override intentionally selects native Triton for ordinary tests.
    from mrotary import load

    reference = load("mrotary_embedding_torch").reference
else:
    reference = get_reference("mrotary_embedding")
