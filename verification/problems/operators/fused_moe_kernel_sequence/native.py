# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from operator_common import native_module
from operator_common import reference as make_reference

reference = make_reference("_fused_moe_kernel_sequence", native=True)
SOURCE = native_module().source_info("_fused_moe_kernel_sequence")
