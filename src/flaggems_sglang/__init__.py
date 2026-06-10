"""
flag_dnn - DNN operations implemented with Triton
"""

from flaggems_sglang import runtime  # noqa: F401
from flaggems_sglang import testing  # noqa: F401
from flaggems_sglang.ops.fused_recurrent_gated_delta_rule_packed_decode import (  # noqa: F401, E501
    fused_recurrent_gated_delta_rule_packed_decode,
)

device = runtime.device.name
vendor_name = runtime.device.vendor_name

__version__ = "0.1.0"
