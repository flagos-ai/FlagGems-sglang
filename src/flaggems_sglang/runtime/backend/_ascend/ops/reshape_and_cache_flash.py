# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _pow2(K, V, KC, VC, S, T: tl.constexpr, W: tl.constexpr):
    cols = tl.arange(0, W)
    for i in range(tl.program_id(0), T, tl.num_programs(0)):
        t = i.to(tl.int64)
        slot = tl.load(S + t).to(tl.int64)
        if slot >= 0:
            tl.store(KC + slot * W + cols, tl.load(K + t * W + cols))
            tl.store(VC + slot * W + cols, tl.load(V + t * W + cols))


@triton.jit
def _flat(
    K,
    V,
    KC,
    VC,
    S,
    SWA,
    KS,
    VS,
    T: tl.constexpr,
    NSWA: tl.constexpr,
    W: tl.constexpr,
    HAS_SWA: tl.constexpr,
    SCALE: tl.constexpr,
    EXACT: tl.constexpr,
    B: tl.constexpr,
):
    cols = tl.arange(0, B)
    mask = cols < W
    for i in range(tl.program_id(0), T, tl.num_programs(0)):
        t = i.to(tl.int64)
        slot = tl.load(S + t).to(tl.int64)
        if HAS_SWA:
            slot = tl.load(SWA + tl.where(slot < 0, slot + NSWA, slot)).to(
                tl.int64
            )
        if slot >= 0:
            src = t * W + cols
            dst = slot * W + cols
            if EXACT:
                k = tl.load(K + src)
                v = tl.load(V + src)
                if SCALE:
                    k = k.to(tl.float32) / tl.load(KS).to(tl.float32)
                    v = v.to(tl.float32) / tl.load(VS).to(tl.float32)
                tl.store(KC + dst, k.to(KC.dtype.element_ty))
                tl.store(VC + dst, v.to(VC.dtype.element_ty))
            else:
                k = tl.load(K + src, mask)
                v = tl.load(V + src, mask)
                if SCALE:
                    k = k.to(tl.float32) / tl.load(KS).to(tl.float32)
                    v = v.to(tl.float32) / tl.load(VS).to(tl.float32)
                tl.store(KC + dst, k.to(KC.dtype.element_ty), mask)
                tl.store(VC + dst, v.to(VC.dtype.element_ty), mask)


@triton.jit
def _strided(
    K,
    V,
    KC,
    VC,
    S,
    SWA,
    KS,
    VS,
    NSWA: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    W: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    E2: tl.constexpr,
    E3: tl.constexpr,
    SS: tl.constexpr,
    WS: tl.constexpr,
    HAS_SWA: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(S + t * SS).to(tl.int64)
    if HAS_SWA:
        slot = tl.load(SWA + tl.where(slot < 0, slot + NSWA, slot) * WS).to(
            tl.int64
        )
    if slot >= 0:
        cols = tl.arange(0, B)
        mask = cols < W
        head = cols // D
        dim = cols % D
        k = tl.load(K + t * K0 + head * K1 + dim * K2, mask)
        v = tl.load(V + t * V0 + head * V1 + dim * V2, mask)
        if SCALE:
            k = k.to(tl.float32) / tl.load(KS).to(tl.float32)
            v = v.to(tl.float32) / tl.load(VS).to(tl.float32)
        page = slot // PS
        pos = slot % PS
        tl.store(
            KC + page * C0 + pos * C1 + head * C2 + dim * C3,
            k.to(KC.dtype.element_ty),
            mask,
        )
        tl.store(
            VC + page * E0 + pos * E1 + head * E2 + dim * E3,
            v.to(VC.dtype.element_ty),
            mask,
        )


def reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping,
    k_scale,
    v_scale,
):
    shape = key.shape
    t = shape[0]
    w = shape[1] * shape[2]
    if t and w:
        b = 1 << (w - 1).bit_length()
        has_swa = swa_slot_mapping is not None
        scale = k_scale is not None
        swa = swa_slot_mapping if has_swa else slot_mapping
        nswa = swa.numel() if has_swa else 0
        ks = k_scale if scale else slot_mapping
        vs = v_scale if scale else slot_mapping
        if (
            key.is_contiguous()
            and value.is_contiguous()
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
            and slot_mapping.stride(0) == 1
            and swa.stride(0) == 1
        ):
            grid = (48 if t > 48 else t,)
            if not has_swa and not scale and b == w:
                _pow2[grid](
                    key, value, key_cache, value_cache, slot_mapping, t, w
                )
            else:
                _flat[grid](
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    swa,
                    ks,
                    vs,
                    t,
                    nswa,
                    w,
                    has_swa,
                    scale,
                    b == w,
                    b,
                    num_warps=4,
                )
        else:
            _strided[(t,)](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                swa,
                ks,
                vs,
                nswa,
                shape[2],
                key_cache.shape[1],
                w,
                *key.stride(),
                *value.stride(),
                *key_cache.stride(),
                *value_cache.stride(),
                slot_mapping.stride(0),
                swa.stride(0),
                has_swa,
                scale,
                b,
                num_warps=4,
            )
    return key_cache, value_cache


__all__ = ["reshape_and_cache_flash"]
