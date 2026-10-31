# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from qwen36_common import native_module
from qwen36_common import reference as make_reference

reference = make_reference("silu_and_mul_out", native=True)
SOURCE = native_module().source_info("silu_and_mul_out")
