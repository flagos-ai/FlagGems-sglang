"""
flag_dnn - DNN operations implemented with Triton
"""

from flaggems_sglang import testing  # noqa: F401
from flaggems_sglang import runtime

device = runtime.device.name
vendor_name = runtime.device.vendor_name

__version__ = "0.1.0"
