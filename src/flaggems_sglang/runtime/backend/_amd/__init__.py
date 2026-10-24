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

from ..backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="amd",
    device_name="cuda",
    device_query_cmd="rocm-smi",
    triton_extra_name="hip",
)

"""
Mapping from an AMD GPU architecture to the directory holding that
architecture's specialized operators.

Keys are looked up from the most to the least specific: the exact target
torch reports in gcnArchName, then "major.minor", then the major version
alone. An architecture absent from the map falls back to the vendor-wide
and generic operators.

Targets are listed individually because the major version alone is too
coarse: every gfx11 target reports major 11, from discrete RDNA3 GPUs to
the RDNA3.5 APUs, and gfx1250 reports major 12 like RDNA4 although it is a
separate family. The rdna3 operators are tuned and tested on gfx1100 only.
"""

ARCH_MAP = {
    "gfx1100": "rdna3",
}

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
