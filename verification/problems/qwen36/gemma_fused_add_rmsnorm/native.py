# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from qwen36_common import native_module
from qwen36_common import reference as make_reference

reference = make_reference("gemma_fused_add_rmsnorm", native=True)
SOURCE = native_module().source_info("gemma_fused_add_rmsnorm")
