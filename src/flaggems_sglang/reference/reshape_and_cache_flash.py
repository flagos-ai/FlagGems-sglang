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

"""Reference implementation for kvcache/reshape_and_cache_flash."""

import torch


def reference(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping,
    k_scale,
    v_scale,
):
    kc = key_cache.clone()
    vc = value_cache.clone()

    slots = slot_mapping.to(torch.int64)
    if swa_slot_mapping is not None:
        # Second-stage remap: the first table indexes into the SWA table.
        slots = swa_slot_mapping.to(torch.int64)[slots]

    # slot < 0 means drop the token; the kernel returns early and touches no
    # cache row, so those slots keep their original value.
    live = slots >= 0
    k = key[live].to(torch.float32)
    v = value[live].to(torch.float32)
    if k_scale is not None:
        # Division, not multiplication: the dequantization convention.
        k = k / k_scale
        v = v / v_scale

    dest = slots[live]
    kc.view(-1, *kc.shape[2:])[dest] = k.to(kc.dtype)
    vc.view(-1, *vc.shape[2:])[dest] = v.to(vc.dtype)
    return kc, vc
