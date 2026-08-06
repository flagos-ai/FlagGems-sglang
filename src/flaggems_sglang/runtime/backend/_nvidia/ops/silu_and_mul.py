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

"""Vendor-specialized silu_and_mul for NVIDIA (placeholder).

Overrides the generic implementation on any NVIDIA GPU that isn't
matched by a more specific arch tier. Competition-entry stub.
"""

import torch


def silu_and_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("silu_and_mul (_nvidia): competition stub")


__all__ = ["silu_and_mul"]
