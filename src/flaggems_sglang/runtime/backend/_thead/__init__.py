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

"""
T-Head Zhenwu (真武) PPU Backend Configuration

Product: Zhenwu PPU (真武处理器)
- Model: Zhenwu 810E (supports up to 16 cards with ICN interconnect)
- Architecture: Proprietary T-Head AI accelerator architecture
- SDK: PPU SDK v2.0.0+

Key Features:
- Full CUDA API compatibility (cuda runtime & driver APIs)
- Triton support: 2.3.x, 3.0.x - 3.4.x with AIU extensions
- Device management: ppu-smi tool (similar to nvidia-smi)
"""

from ..backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="thead",
    # PPU uses CUDA-compatible API, accessed via torch.cuda
    device_name="cuda",
    # PPU device management tool (similar to nvidia-smi)
    device_query_cmd="ppu-smi",
    dispatch_key=None,
    triton_extra_name=None,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
