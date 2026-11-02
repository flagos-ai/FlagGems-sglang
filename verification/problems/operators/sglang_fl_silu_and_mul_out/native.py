# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from operator_common import native_module
from operator_common import reference as make_reference

reference = make_reference("sglang_fl_silu_and_mul_out", native=True)
SOURCE = native_module().source_info("silu_and_mul_out")
