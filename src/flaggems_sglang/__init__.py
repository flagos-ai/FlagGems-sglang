"""
flag_dnn - DNN operations implemented with Triton
"""

from flaggems_sglang import testing  # noqa: F401
from flaggems_sglang import runtime
from flaggems_sglang.ops.fused_moe import (  # noqa: F401
    triton_kernel_fused_experts,
    triton_kernel_fused_experts_with_bias,
)

device = runtime.device.name
vendor_name = runtime.device.vendor_name

__version__ = "0.1.0"
