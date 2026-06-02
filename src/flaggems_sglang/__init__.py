"""
flag_dnn - DNN operations implemented with Triton
"""

from flaggems_sglang import runtime
from flaggems_sglang import testing  # noqa: F401
from flaggems_sglang.ops.gemma_rms_norm import gemma_rms_norm

device = runtime.device.name
vendor_name = runtime.device.vendor_name

__version__ = "0.1.0"
