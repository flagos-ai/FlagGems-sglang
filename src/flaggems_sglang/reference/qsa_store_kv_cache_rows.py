# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors


def reference(k_cache, v_cache, slots, key, value):
    valid = (slots >= 0) & (slots < k_cache.shape[0] * k_cache.shape[1])
    positions = slots[valid].long()
    pages = positions.div(k_cache.shape[1], rounding_mode="floor")
    offsets = positions.remainder(k_cache.shape[1])
    k_cache[pages, offsets] = key[valid]
    v_cache[pages, offsets] = value[valid]
