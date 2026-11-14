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

"""Reference for reshape_and_cache_flash.

Written as an explicit loop over tokens so the slot resolution order is
visible: skip a negative slot, remap through the sliding-window table when one
is given, skip again if the remapped slot is negative, then scatter the head
rows into the block that slot falls in. The caches are copied first, because
the contract returns them rather than mutating the inputs.
"""


def reference(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
):
    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()
    block_size = key_cache_out.shape[1]
    num_tokens = key.shape[0]

    for token in range(num_tokens):
        slot = int(slot_mapping[token])
        if slot < 0:
            continue

        if swa_slot_mapping is not None:
            slot = int(swa_slot_mapping[slot])
            if slot < 0:
                continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        k = key[token]
        v = value[token]
        if k_scale is not None:
            # Division, matching the contract's wording and upstream's kernel.
            k = (k.float() / k_scale.float()).to(key_cache_out.dtype)
            v = (v.float() / v_scale.float()).to(value_cache_out.dtype)

        key_cache_out[block_idx, block_offset] = k
        value_cache_out[block_idx, block_offset] = v

    return key_cache_out, value_cache_out
