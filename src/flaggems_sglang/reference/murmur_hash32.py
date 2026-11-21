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

"""Reference implementation for utility/murmur_hash32."""

import torch

_MASK32 = 0xFFFFFFFF


def _rotl32(x, r):
    """Rotate left 32-bit x by r bits (uint32 semantics, int64 carrier)."""
    x = x & _MASK32
    return ((x << r) | (x >> (32 - r))) & _MASK32


def _fmix32(h):
    """MurmurHash3 final mixer (uint32)."""
    h = h & _MASK32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _MASK32
    h ^= h >> 16
    return h


def _murmur3_mix(h, k):
    """Murmur3 32-bit block mixer: h = mix(h, k) (uint32)."""
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    h = h & _MASK32
    k = k & _MASK32
    k = (k * c1) & _MASK32
    k = _rotl32(k, 15)
    k = (k * c2) & _MASK32
    h ^= k
    h = _rotl32(h, 13)
    h = (h * 5 + 0xE6546B64) & _MASK32
    return h


def reference(seed, positions, col_indices):
    assert (
        seed.shape == positions.shape
    ), "Seed and positions must have same shape"
    n = seed.shape[0]
    m = col_indices.shape[0]
    device = seed.device

    seed = seed.to(torch.uint64).to(torch.int64)
    positions = positions.to(torch.int64)
    col_indices = col_indices.to(torch.int64)

    seed_lo = seed & _MASK32
    seed_hi = (seed >> 32) & _MASK32

    # h starts at 0 for every element; broadcast against the columns.
    h = torch.zeros((n, m), dtype=torch.int64, device=device)

    def mix(k, h):
        return _murmur3_mix(h, k[:, None] if k.dim() == 1 else k)

    h = mix(seed_lo, h)
    h = mix(seed_hi, h)
    h = mix(positions, h)
    h = mix(col_indices[None, :].expand(n, m), h)

    h = (h ^ 16) & _MASK32
    h = _fmix32(h)
    return h.to(torch.uint32)
