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
"""Fused routed-row SiLU, multiplication, and optional routing weights.

All arithmetic is FP32, with one conversion at the output. Routing/reordering
is performed by the caller. This forward-only implementation does not use PDL.
Native autotune and compiler tuning are enabled by default. At most 91
shape-pruned candidates compare geometry, loads, cache/store policy and pipeline
stages. Timing uses ordinary launches and Events, without CUDA Graph.
Only native Triton configuration/timing metadata persists; every call
recomputes its output. The explicit _autotune diagnostic helper remains separate.
Warm up each signature before CUDA graph capture, or use autotune=False.
"""

import itertools
import math
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid(gate, MODE: tl.constexpr):
    if MODE == 0:
        return tl.sigmoid(gate)
    elif MODE == 1:
        return 1.0 / (1.0 + tl.exp(-gate))
    elif MODE == 2:
        return 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
    else:
        denominator = 1.0 + tl.exp2(-gate * 1.4426950408889634)
        # No .ftz: a subnormal sigmoid can become significant after multiplying
        # by a large up value. Do not reassociate the three FP32 multiplies.
        return tl.inline_asm_elementwise(
            "rcp.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


@triton.jit
def _routed_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EVEN_M: tl.constexpr,
    INDEX64: tl.constexpr,
    CACHE: tl.constexpr,
    LOAD: tl.constexpr = 0,
    SIGMOID: tl.constexpr = 0,
    STORE: tl.constexpr = "",
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    tiles: tl.constexpr = tl.cdiv(N, BN)
    rows = (pid // tiles) * BM + tl.arange(0, BM)
    cols = (pid % tiles) * BN + tl.arange(0, BN)
    if EVEN_M:
        mask = cols[None, :] < N
    else:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
    modifier: tl.constexpr = "" if CACHE == "hint" or CACHE == "stream" else CACHE
    eviction: tl.constexpr = "evict_first" if CACHE == "stream" else ""
    if LOAD == 0:
        q = tl.arange(0, 2 * BN)
        if INTERLEAVED:
            pair_cols = (pid % tiles) * (2 * BN) + q
            valid_cols = pair_cols < 2 * N
        else:
            logical_cols = (pid % tiles) * BN + q % BN
            pair_cols = logical_cols + (q // BN) * N
            valid_cols = logical_cols < N
        if EVEN_M:
            pair_mask = valid_cols[None, :]
        else:
            pair_mask = (rows[:, None] < M) & valid_cols[None, :]
        pair = tl.load(
            X + rows[:, None] * S + pair_cols[None, :],
            pair_mask,
            other=0,
            cache_modifier=modifier,
            eviction_policy=eviction,
        )
        if INTERLEAVED:
            gate, up = tl.split(tl.reshape(pair, (BM, BN, 2)))
        else:
            gate, up = tl.split(tl.permute(tl.reshape(pair, (BM, 2, BN)), (0, 2, 1)))
    else:
        if INTERLEAVED:
            offsets = rows[:, None] * S + 2 * cols[None, :]
            delta: tl.constexpr = 1
        else:
            offsets = rows[:, None] * S + cols[None, :]
            delta: tl.constexpr = N
        gate = tl.load(
            X + offsets,
            mask,
            other=0,
            cache_modifier=modifier,
            eviction_policy="evict_last" if CACHE == "hint" else eviction,
        )
        up = tl.load(
            X + offsets + delta,
            mask,
            other=0,
            cache_modifier=modifier,
            eviction_policy="evict_first" if CACHE == "hint" else eviction,
        )
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    result = (gate * _sigmoid(gate, SIGMOID)) * up
    if WEIGHT:
        if EVEN_M:
            weight = tl.load(W + rows, eviction_policy="evict_last").to(tl.float32)
        else:
            weight = tl.load(
                W + rows, rows < M, other=0, eviction_policy="evict_last"
            ).to(tl.float32)
        result = result * weight[:, None]
    tl.store(Y + rows[:, None] * N + cols[None, :], result, mask, cache_modifier=STORE)


@triton.jit
def _routed_flat(
    X,
    W,
    Y,
    TOTAL,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr,
    CACHE: tl.constexpr,
    LOAD: tl.constexpr = 0,
    SIGMOID: tl.constexpr = 0,
    STORE: tl.constexpr = "",
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
        TOTAL = TOTAL.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // N
    cols = offsets % N
    modifier: tl.constexpr = "" if CACHE == "hint" or CACHE == "stream" else CACHE
    eviction: tl.constexpr = "evict_first" if CACHE == "stream" else ""
    if LOAD == 0:
        q = tl.arange(0, 2 * BLOCK)
        if INTERLEAVED:
            if S == 2 * N:
                pair_offsets = pid * (2 * BLOCK) + q
                pair_mask = pair_offsets < 2 * TOTAL
            else:
                logical = pid * BLOCK + q // 2
                pair_offsets = (logical // N) * S + 2 * (logical % N) + q % 2
                pair_mask = logical < TOTAL
        else:
            logical = pid * BLOCK + q % BLOCK
            pair_offsets = (logical // N) * S + logical % N + (q // BLOCK) * N
            pair_mask = logical < TOTAL
        pair = tl.load(
            X + pair_offsets,
            pair_mask,
            other=0,
            cache_modifier=modifier,
            eviction_policy=eviction,
        )
        if INTERLEAVED:
            gate, up = tl.split(tl.reshape(pair, (BLOCK, 2)))
        else:
            gate, up = tl.split(tl.trans(tl.reshape(pair, (2, BLOCK))))
    else:
        if INTERLEAVED:
            gate_offsets = rows * S + 2 * cols
            up_offsets = gate_offsets + 1
        else:
            gate_offsets = rows * S + cols
            up_offsets = gate_offsets + N
        gate = tl.load(
            X + gate_offsets,
            offsets < TOTAL,
            other=0,
            cache_modifier=modifier,
            eviction_policy="evict_last" if CACHE == "hint" else eviction,
        )
        up = tl.load(
            X + up_offsets,
            offsets < TOTAL,
            other=0,
            cache_modifier=modifier,
            eviction_policy="evict_first" if CACHE == "hint" else eviction,
        )
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    result = (gate * _sigmoid(gate, SIGMOID)) * up
    if WEIGHT:
        weight = tl.load(W + rows, offsets < TOTAL, other=0).to(tl.float32)
        result = result * weight
    tl.store(Y + offsets, result, offsets < TOTAL, cache_modifier=STORE)


def _select_config(m, n, interleaved, weighted, element_size):
    """Return the static search seed, not the final autotuned configuration."""
    # Previous H20 winner is always included as a seed in the wider search.
    total = m * n
    if total <= 32768:
        return ("flat", 1, 256, 4, "")
    if m <= 256:
        if m > 64 and total > 196608:
            if element_size == 4:
                return (
                    "tile",
                    1,
                    1024 if n >= 4096 and not interleaved else 512,
                    4,
                    "hint" if not interleaved and n <= 2048 else "",
                )
            if not interleaved:
                return (
                    "tile",
                    1 if n < 2048 else 2,
                    512,
                    4,
                    "hint" if n == 2048 else "",
                )
            if n >= 2048:
                return ("tile", 1, 1024, 4, "")
        return ("flat", 1, 512, 8 if total <= 196608 else 4, "")
    if n <= 512:
        if interleaved and m >= 4096:
            return ("tile", 1, triton.next_power_of_2(n), 1, "")
        return ("flat", 1, 1024 if element_size == 2 else 512, 4, "")
    if element_size == 4:
        if interleaved:
            if m < 4096 and n < 1024:
                return ("tile", 1, triton.next_power_of_2(n), 4, "")
            return ("tile", 1, 512, 1 if total <= 8388608 else 2, "")
        if m < 4096 and n < 1024:
            return ("tile", 4, 256, 4, "")
        if m < 4096 and n >= 2048 and total * 12 <= 60 * 1024 * 1024:
            return ("tile", 1, 512, 4, "hint")
        if m >= 4096 and n <= 1024:
            return ("tile", 1, 1024, 4, "")
        return ("tile", 2, 512 if total <= 8388608 else 1024, 4, "")
    if interleaved:
        if weighted and m >= 16384 and n >= 4096:
            return ("tile", 1, 1024, 2, "")
        if n < 1024:
            return ("tile", 1, triton.next_power_of_2(n), 1 if m >= 4096 else 2, "")
        bn = min(4096, triton.next_power_of_2(n))
        if m < 4096 and total >= 6291456:
            return ("tile", 1, 1024, 1, "")
        if m < 4096 and n == 2048:
            # Balance warm reuse and input pools larger than L2.
            return ("tile", 1, 1024, 4, "")
        warps = 2 if bn <= 2048 and m < 4096 else max(2, bn // 512)
        if bn == 4096 and n != bn and m >= 4096:
            warps = 4
        if n == 1024 and m < 4096:
            warps = 4
        cache = ".cg" if m >= 4096 and bn >= 2048 and n != bn else ""
        return ("tile", 1, bn, warps, cache)
    if n < 1024:
        if m < 4096:
            return ("flat", 1, 1024, 4, "")
        if m <= 16384:
            return ("tile", 1, triton.next_power_of_2(n), 1, "hint")
        return ("tile", 4, 256, 2, "")
    if n % 1024 == 512:
        return ("tile", 4, 256, 4, "")
    if n >= 4096 and m >= 4096:
        return ("tile", 1, 4096, 4, "")
    if m >= 4096:
        if total * 6 <= 60 * 1024 * 1024:
            return ("tile", 2, 512, 1, "hint")
        return ("tile", 2, 1024, 2, "")
    return ("tile", 1, 1024, 4, "hint" if total >= 6291456 else "")


def _launch(x, weights, out, config, interleaved=True):
    m, width = x.shape
    n = width // 2
    normalized = _normalize_config(config, interleaved)
    family, bm, bn, warps, cache, load, sigmoid, store = normalized[:8]
    stages, ctas, maxnreg, fusion = (
        normalized[8:] if len(normalized) == 12 else (1, 1, None, True)
    )
    # Older CUDA backends reject even maxnreg=None as an unknown keyword.
    # Omit unused optional controls, but preserve explicit fusion: its backend
    # default can be changed by a Triton knob. Requested controls are never
    # silently dropped or retried under a different configuration.
    compiler = dict(num_stages=stages, enable_fp_fusion=fusion)
    if ctas != 1:
        compiler["num_ctas"] = ctas
    if maxnreg is not None:
        compiler["maxnreg"] = maxnreg
    # Address span, rather than logical nbytes, matters for padded row strides.
    index64 = max((m - 1) * x.stride(0) + width, m * n) >= 2**31
    if family == "flat":
        return _routed_flat[(triton.cdiv(m * n, bn),)](
            x,
            weights,
            out,
            m * n,
            n,
            x.stride(0),
            weights is not None,
            interleaved,
            bn,
            index64,
            cache,
            load,
            sigmoid,
            store,
            num_warps=warps,
            **compiler,
        )
    else:
        return _routed_tile[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            x,
            weights,
            out,
            m,
            n,
            x.stride(0),
            weights is not None,
            interleaved,
            bm,
            bn,
            m % bm == 0,
            index64,
            cache,
            load,
            sigmoid,
            store,
            num_warps=warps,
            **compiler,
        )


def _normalize_config(config, interleaved):
    if len(config) == 5:
        return (*config, 0 if interleaved else 1, 0, "")
    if len(config) not in (8, 12):
        raise ValueError("Config needs 5, 8, or 12 entries")
    if len(config) == 12:
        stages, ctas, maxnreg, fusion = config[8:]
        if stages not in (1, 2, 3, 4, 5, 6, 8) or ctas not in (1, 2, 4, 8):
            raise ValueError("Unsupported stages or CTA-cluster size")
        if maxnreg not in (None, 32, 64, 96, 128, 192, 255):
            raise ValueError("Unsupported maxnreg")
        if not isinstance(fusion, bool):
            raise TypeError("fp_fusion must be bool")
    return tuple(config)


def _select_measured_config(x, weights, out, interleaved):
    """Select an offline-measured configuration using only input metadata.

    The H20 table is local and is never updated with runtime data. Unmeasured
    strides, alignments, dtype combinations, and shapes use the static seed.
    """
    m, width = x.shape
    n = width // 2
    fallback = _select_config(m, n, interleaved, weights is not None, x.element_size())
    if (
        "H20" not in torch.cuda.get_device_name(x.device)
        or x.stride(0) != width
        or out.dtype != x.dtype
        or x.data_ptr() % 16
        or out.data_ptr() % 16
        or (
            weights is not None
            and (weights.dtype != torch.float32 or weights.data_ptr() % 16)
        )
    ):
        return fallback
    configs = {
        (4096, 64, torch.float16, False, True): ("tile", 8, 64, 4, ".cg", 1, 3, ".wt"),
        (8, 768, torch.bfloat16, True, True): ("flat", 1, 512, 8, ".cg", 0, 3, ""),
        (8, 768, torch.bfloat16, False, True): ("flat", 1, 256, 4, ".ca", 1, 2, ""),
        (8, 768, torch.float16, True, True): ("flat", 1, 512, 8, ".cg", 0, 2, ""),
        (8, 768, torch.float16, False, True): ("flat", 1, 512, 8, ".ca", 1, 2, ""),
        (8, 768, torch.float32, True, True): ("flat", 1, 512, 8, ".cg", 0, 3, ""),
        (8, 768, torch.float32, False, True): ("flat", 1, 512, 8, ".cg", 1, 2, ".cs"),
        (256, 1024, torch.bfloat16, True, True): ("flat", 1, 512, 4, ".ca", 0, 3, ""),
        (256, 1024, torch.bfloat16, False, True): ("tile", 1, 512, 4, ".cg", 1, 2, ""),
        (256, 1024, torch.float16, True, True): ("flat", 1, 512, 4, "", 0, 3, ""),
        (256, 1024, torch.float16, False, True): ("tile", 1, 512, 4, "", 1, 3, ""),
        (256, 1024, torch.float32, True, True): ("flat", 1, 512, 8, "stream", 0, 3, ""),
        (256, 1024, torch.float32, False, True): (
            "tile",
            1,
            512,
            4,
            "stream",
            0,
            2,
            "",
        ),
        (1536, 2048, torch.bfloat16, True, True): ("tile", 1, 2048, 8, ".cg", 0, 2, ""),
        (1536, 2048, torch.bfloat16, False, True): (
            "tile",
            1,
            2048,
            8,
            ".cg",
            1,
            2,
            "",
        ),
        (1536, 2048, torch.float16, True, True): (
            "tile",
            1,
            1024,
            4,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (1536, 2048, torch.float16, False, True): ("tile", 1, 1024, 4, ".cg", 1, 2, ""),
        (1536, 2048, torch.float32, True, True): (
            "tile",
            2,
            256,
            1,
            ".cv",
            0,
            1,
            ".cs",
        ),
        (1536, 2048, torch.float32, False, True): (
            "tile",
            32,
            32,
            4,
            ".ca",
            1,
            1,
            ".cs",
        ),
        (8192, 768, torch.bfloat16, True, True): (
            "tile",
            1,
            1024,
            1,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (8192, 768, torch.bfloat16, False, True): (
            "tile",
            1,
            1024,
            1,
            ".cg",
            1,
            3,
            ".cs",
        ),
        (8192, 768, torch.float16, True, True): (
            "tile",
            1,
            1024,
            1,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (8192, 768, torch.float16, False, True): ("tile", 1, 1024, 1, "hint", 1, 2, ""),
        (8192, 768, torch.float32, True, True): (
            "tile",
            1,
            1024,
            4,
            "stream",
            0,
            0,
            "",
        ),
        (8192, 768, torch.float32, False, True): (
            "tile",
            1,
            1024,
            4,
            "stream",
            1,
            2,
            ".cs",
        ),
        (8192, 4096, torch.bfloat16, True, True): ("tile", 1, 4096, 8, ".cg", 0, 2, ""),
        (8192, 4096, torch.bfloat16, False, True): (
            "tile",
            1,
            1024,
            2,
            ".cg",
            1,
            2,
            ".cs",
        ),
        (8192, 4096, torch.float16, True, True): ("tile", 1, 512, 2, ".cg", 0, 2, ""),
        (8192, 4096, torch.float16, False, True): ("tile", 1, 2048, 4, "", 1, 2, ""),
        (8192, 4096, torch.float32, True, True): ("tile", 2, 512, 4, ".ca", 0, 2, ""),
        (8192, 4096, torch.float32, False, True): ("tile", 2, 256, 4, ".ca", 1, 2, ""),
        (49152, 3072, torch.bfloat16, True, True): (
            "tile",
            1,
            4096,
            4,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (49152, 3072, torch.bfloat16, False, True): (
            "tile",
            1,
            1024,
            2,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (49152, 3072, torch.float16, True, True): ("tile", 1, 4096, 4, ".ca", 0, 1, ""),
        (49152, 3072, torch.float16, False, True): (
            "tile",
            1,
            1024,
            2,
            ".cg",
            1,
            2,
            ".cs",
        ),
        (49152, 3072, torch.float32, True, True): (
            "tile",
            2,
            512,
            4,
            ".cg",
            0,
            1,
            ".cs",
        ),
        (49152, 3072, torch.float32, False, True): ("flat", 1, 2048, 8, "", 1, 3, ""),
        (1536, 2048, torch.bfloat16, True, False): (
            "tile",
            1,
            2048,
            8,
            ".ca",
            0,
            2,
            "",
        ),
        (1536, 2048, torch.bfloat16, False, False): ("tile", 1, 2048, 8, "", 0, 3, ""),
        (1536, 2048, torch.float16, True, False): (
            "tile",
            1,
            1024,
            4,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (1536, 2048, torch.float16, False, False): (
            "tile",
            1,
            1024,
            4,
            ".cg",
            1,
            2,
            "",
        ),
        (1536, 2048, torch.float32, True, False): (
            "tile",
            2,
            256,
            2,
            ".cg",
            0,
            2,
            ".cs",
        ),
        (1536, 2048, torch.float32, False, False): (
            "tile",
            2,
            1024,
            8,
            ".cg",
            1,
            0,
            ".cs",
        ),
    }
    return configs.get((m, n, x.dtype, interleaved, weights is not None), fallback)


def _spread_configs(configs, limit):
    """Select a deterministic spread of geometry from every family/load pair."""
    configs = list(dict.fromkeys(configs))
    if len(configs) <= limit:
        return configs
    groups = [
        [c for c in configs if c[0] == family and c[5] == load]
        for family in ("flat", "tile")
        for load in (0, 1)
    ]
    count = max(1, triton.cdiv(limit, 4))
    sampled = []
    for group in groups:
        if len(group) <= count:
            sampled.append(group)
        else:
            sampled.append(
                [group[i * (len(group) - 1) // max(1, count - 1)] for i in range(count)]
            )
    result = [
        c for row in itertools.zip_longest(*sampled) for c in row if c is not None
    ]
    if len(result) < limit:
        selected = set(result)
        result.extend(c for c in configs if c not in selected)
    return result[:limit]


def _geometry_configs(m, n, interleaved, weighted, element_size, *, exhaustive=False):
    """At most 384 shape-appropriate geometries, shared by both search modes.

    BM 1..128, BN 16..16384, flat blocks 32..16384, warps 1/2/4/8.
    Tile area is at most 16384; excessive padding, sparse grids and more than
    128 output elements per lane are excluded to bound cold compilation.
    ``exhaustive`` remains accepted, but no longer enables giant geometries.
    """
    seed = _normalize_config(
        _select_config(m, n, interleaved, weighted, element_size), interleaved
    )
    configs = [seed]
    total = m * n
    for bn in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        if bn > max(256, triton.next_power_of_2(total)):
            continue
        for warps in (1, 2, 4, 8):
            if not 1 <= bn / (32 * warps) <= 128:
                continue
            for load in (0, 1):
                configs.append(("flat", 1, bn, warps, "", load, 0, ""))
    for bm in (1, 2, 4, 8, 16, 32, 64, 128):
        if bm > triton.next_power_of_2(m):
            continue
        for bn in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            area = bm * bn
            if bn > max(64, triton.next_power_of_2(n)) or area > 16384 or area < 64:
                continue
            if total >= 32768 and triton.cdiv(m, bm) * triton.cdiv(n, bn) < 32:
                continue
            preferred = min(8, max(1, triton.next_power_of_2(triton.cdiv(area, 256))))
            for warps in sorted({preferred, min(8, 2 * preferred)}):
                if not 1 <= area / (32 * warps) <= 128:
                    continue
                for load in (0, 1):
                    configs.append(("tile", bm, bn, warps, "", load, 0, ""))
    return _spread_configs(configs, 384)


def _staged_geometry_configs(m, n, interleaved, weighted, element_size):
    """Compatibility alias for the finite shape-appropriate geometry set."""
    return _geometry_configs(m, n, interleaved, weighted, element_size)


def _coordinate_configs(
    config,
    *,
    capability=(9, 0),
    math_modes=(0, 1, 2, 3),
    caches=("", ".ca", ".cg", ".cv", "stream", "hint"),
    stores=("", ".wb", ".cg", ".cs", ".wt"),
    stages=(1, 2, 3, 4),
    ctas=(1, 2),
    registers=(None, 64, 128),
    fusion=(True, False),
):
    """One-coordinate trials around one geometry, never a Cartesian product."""
    base = (*config[:8], 1, 1, None, True)
    dimensions = (
        (6, math_modes),
        (4, caches),
        (7, stores),
        (8, stages),
        (9, tuple(c for c in ctas if c == 1 or capability[0] >= 9)),
        (10, registers),
        (11, fusion),
    )
    for index, values in dimensions:
        for value in values:
            if value != base[index]:
                yield (*base[:index], value, *base[index + 1 :])


def _iter_configs(
    m,
    n,
    interleaved,
    weighted,
    element_size,
    *,
    capability=(9, 0),
    geometries=None,
    math_modes=(0, 1, 2, 3),
    caches=("", ".ca", ".cg", ".cv", "stream", "hint"),
    stores=("", ".wb", ".cg", ".cs", ".wt"),
    stages=(1, 2, 3, 4),
    ctas=(1, 2),
    registers=(None, 64, 128),
    fusion=(True, False),
):
    """Enumerate a finite geometry-plus-coordinate catalog, normally < 500.

    ``exhaustive`` tuning visits this catalog, not a Cartesian product. At most
    384 supplied geometries are consumed; four family/load representatives
    receive independent math/cache/store/compiler trials. Dimension overrides
    change those coordinates only. Legacy five/eight-entry configs still work.
    """
    if geometries is None:
        geometries = _geometry_configs(m, n, interleaved, weighted, element_size)
    geometries = [
        _normalize_config(tuple(c), interleaved)
        for c in itertools.islice(geometries, 384)
    ]
    seen = set()
    for config in geometries:
        if config not in seen:
            seen.add(config)
            yield config
    seed_area = max(256, min(4096, n))
    for family in ("flat", "tile"):
        for load in (0, 1):
            group = [c for c in geometries if c[0] == family and c[5] == load]
            if not group:
                continue
            anchor = min(group, key=lambda c: abs(c[1] * c[2] - seed_area))
            for config in _coordinate_configs(
                anchor,
                capability=capability,
                math_modes=math_modes,
                caches=caches,
                stores=stores,
                stages=stages,
                ctas=ctas,
                registers=registers,
                fusion=fusion,
            ):
                if config not in seen:
                    seen.add(config)
                    yield config


def _refine_geometry_configs(anchors, m, n):
    """Small neighbors of measured winners within the same compilation bounds."""
    candidates = []
    for config in anchors:
        family, bm, bn, _, _, load, _, _ = config[:8]
        rows = (1,) if family == "flat" else (max(1, bm // 2), bm, min(128, 2 * bm))
        for row_tile in rows:
            for column_tile in (max(16, bn // 2), bn, min(16384, 2 * bn)):
                area = row_tile * column_tile
                extent = m * n if family == "flat" else n
                if row_tile > triton.next_power_of_2(m) or column_tile > max(
                    256, triton.next_power_of_2(extent)
                ):
                    continue
                if area < 64 or area > 16384:
                    continue
                grid = (
                    triton.cdiv(m * n, column_tile)
                    if family == "flat"
                    else triton.cdiv(m, row_tile) * triton.cdiv(n, column_tile)
                )
                if m * n >= 32768 and grid < 32:
                    continue
                for warps in (1, 2, 4, 8):
                    if 1 <= area / (32 * warps) <= 128:
                        candidates.append(
                            (family, row_tile, column_tile, warps, "", load, 0, "")
                        )
    return list(dict.fromkeys(candidates))


def _autotune(
    x,
    w,
    y,
    interleaved,
    *,
    search="staged",
    configs=None,
    max_configs=64,
    max_seconds=15.0,
    validate=None,
):
    """Explicit local search, defaulting to 64 candidates and a 15-second budget.

    Staged mode reserves candidate slots for geometry, neighbors, math,
    cache/store and compiler coordinates. ``exhaustive`` visits the finite
    catalog from _iter_configs; it does not form a Cartesian product. configs
    can supply an explicit iterable. Duplicate and rejected entries count
    toward max_configs. Either budget can be disabled explicitly with None.

    The time limit is soft: it is checked between candidates, stages and final
    probes, but cannot interrupt a compiler call or an in-flight GPU operation.
    Timing uses ordinary launches and CUDA Events, including dispatch gaps;
    no CUDA Graph is constructed. Finalists share the same deadline. If it expires, existing measurements
    select a validated candidate and the true output is still recomputed.
    The offline seed always receives one ordinary launch and optional validation,
    counted toward max_configs, even if setup exhausted the time budget. An
    unmeasured fallback reports milliseconds=None. info distinguishes warmup,
    warm measurements and completed conditioned probes.
    Repeated final probes are not additional candidate entries.

    All state is local, and normal public calls never invoke this helper or
    remember its result. Call serially with valid nonempty tensors, on the
    input's current CUDA device and outside graph capture. validate, if given,
    receives poisoned ordinary-launch output and must return a boolean.
    """
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    class CandidateRejected(Exception):
        pass

    if not x.numel():
        raise ValueError("Explicit autotuning requires a nonempty input")
    if torch.cuda.current_device() != x.device.index:
        raise ValueError("Enter the input's CUDA device context before autotuning")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Explicit autotuning must run outside CUDA graph capture")
    if search not in ("staged", "exhaustive"):
        raise ValueError("search must be staged or exhaustive")
    if max_configs is not None and (
        not isinstance(max_configs, int)
        or isinstance(max_configs, bool)
        or max_configs < 1
    ):
        raise ValueError("max_configs must be a positive integer or None")
    if max_seconds is not None and (
        not isinstance(max_seconds, (int, float))
        or isinstance(max_seconds, bool)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be a finite positive number or None")
    started = time.perf_counter()
    deadline = None if max_seconds is None else started + max_seconds
    m, width = x.shape
    n = width // 2
    capability = torch.cuda.get_device_capability(x.device)
    measured_seed = _normalize_config(
        _select_measured_config(x, w, y, interleaved), interleaved
    )
    mode = "custom" if configs is not None else search
    repetitions = 32 if m * n < 8 * 1024 * 1024 else 8
    timings, skipped, attempted = {}, [], set()
    validated = {}
    progress = {"consumed": 0}
    stage_counts = {}
    final_times, address_times = {}, {}
    conditioning = None
    completed_addresses = 0
    final_status = "not_started"
    stream = torch.cuda.Stream(device=x.device)
    caller = torch.cuda.current_stream(x.device)
    stream.wait_stream(caller)

    def time_available():
        return deadline is None or time.perf_counter() < deadline

    def budget_available():
        return time_available() and (
            max_configs is None or progress["consumed"] < max_configs
        )

    def launch_candidate(config, target):
        # Classify compiler failures only at the launch site, so a validation
        # callback's KeyError or RuntimeError is never mistaken for rejection.
        try:
            return _launch(x, w, target, config, interleaved)
        except (OutOfResources, CompilationError) as error:
            raise CandidateRejected(str(error)) from error
        except KeyError as error:
            names = ["num_stages", "enable_fp_fusion"]
            if len(config) == 12:
                if config[9] != 1:
                    names.append("num_ctas")
                if config[10] is not None:
                    names.append("maxnreg")
            message = error.args[0] if len(error.args) == 1 else None
            unsupported = (
                f"Keyword argument {name} was specified but unrecognised"
                for name in names
            )
            if not isinstance(message, str) or message not in unsupported:
                raise
            # Never retry a requested option after silently removing it.
            raise CandidateRejected("compiler option: " + message) from error
        except RuntimeError as error:
            if str(error).strip() != "PassManager::run failed":
                raise
            raise CandidateRejected("compiler: " + str(error)) from error

    def prepare_candidate(config, target=y):
        if validate is not None:
            target.fill_(float("nan"))
        launch_candidate(config, target)
        stream.synchronize()
        if validate is not None and not validate(target):
            raise CandidateRejected("Ordinary launch failed candidate validation")
        validated[config] = True
        if not time_available():
            return None
        return config, target

    def sample(candidate, conditioning=None):
        if candidate is None or not time_available():
            return None
        config, target = candidate
        if conditioning is not None:
            conditioning.zero_()
        if not time_available():
            return None
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        completed = 0
        begin.record(stream)
        for _ in range(repetitions):
            if not time_available():
                break
            launch_candidate(config, target)
            completed += 1
        end.record(stream)
        end.synchronize()
        return begin.elapsed_time(end) / completed if completed else None

    def measure(config):
        progress["consumed"] += 1
        config = _normalize_config(tuple(config), interleaved)
        if config in attempted:
            return
        attempted.add(config)
        try:
            candidate = prepare_candidate(config)
            values = []
            for _ in range(3):
                value = sample(candidate)
                if value is None:
                    break
                values.append(value)
            if values:
                timings[config] = statistics.median(values)
        except CandidateRejected as error:
            validated.pop(config, None)
            skipped.append((config, str(error)))

    def consume(iterable, limit=None):
        iterator = iter(iterable)
        count = 0
        while budget_available() and (limit is None or count < limit):
            try:
                config = next(iterator)
            except StopIteration:
                break
            measure(config)
            count += 1
        return count

    def winners():
        result = []
        for family in ("flat", "tile"):
            for load in (0, 1):
                group = [c for c in timings if c[0] == family and c[5] == load]
                if group:
                    result.append(min(group, key=timings.get))
        return result

    def new_configs(iterable):
        # Internal finite stages avoid wasting slots on earlier measurements;
        # explicit custom iterables still count every duplicate entry.
        return (
            c for c in iterable if _normalize_config(c, interleaved) not in attempted
        )

    def coordinate_stage(anchors, phase):
        trials = []
        for anchor in anchors:
            if phase == "math":
                options = dict(
                    caches=(), stores=(), stages=(), ctas=(), registers=(), fusion=()
                )
            elif phase == "policy":
                options = dict(
                    math_modes=(), stages=(), ctas=(), registers=(), fusion=()
                )
            else:
                options = dict(math_modes=(), caches=(), stores=())
            trials.append(
                list(_coordinate_configs(anchor, capability=capability, **options))
            )
        # Stagger the coordinates across family/load anchors so a small phase
        # budget still reaches stores and every compiler axis, not only stages.
        for index, trial in enumerate(trials):
            offset = index * max(0, len(trial) - 1) // max(1, len(trials) - 1)
            trials[index] = trial[offset:] + trial[:offset]
        return (
            c for row in itertools.zip_longest(*trials) for c in row if c is not None
        )

    try:
        with torch.cuda.stream(stream):
            # The known offline seed is the sole forced warmup. It counts as
            # one candidate; deadlines may prevent every subsequent operation.
            measure(measured_seed)
            stage_counts["seed"] = 1
            if configs is not None:
                stage_counts["custom"] = consume(configs)
            elif mode == "exhaustive":
                catalog = _iter_configs(
                    m,
                    n,
                    interleaved,
                    w is not None,
                    x.element_size(),
                    capability=capability,
                )
                stage_counts["catalog"] = consume(catalog)
            else:
                if max_configs is None:
                    quotas = (384, 64, 12, 36, 28)
                else:
                    quotas = tuple(
                        max(1, max_configs * portion // 64)
                        for portion in (24, 8, 12, 12, 8)
                    )
                geometry = _geometry_configs(
                    m, n, interleaved, w is not None, x.element_size()
                )
                spread = _spread_configs(geometry, max(1, quotas[0] - 1))
                initial = new_configs(spread)
                stage_counts["geometry"] = consume(initial, max(0, quotas[0] - 1))
                if budget_available():
                    neighbors = list(
                        new_configs(_refine_geometry_configs(winners(), m, n))
                    )
                    stage_counts["neighbors"] = consume(
                        _spread_configs(neighbors, quotas[1]), quotas[1]
                    )
                for phase, quota in zip(("math", "policy", "compiler"), quotas[2:]):
                    if not budget_available():
                        break
                    stage_counts[phase] = consume(
                        new_configs(coordinate_stage(winners(), phase)), quota
                    )
            if not validated:
                raise RuntimeError("No candidate completed within the autotune budgets")

            # Always retain a warm-measured fallback. Prefer normal cache
            # priority when there is no time to condition stream/hint policies.
            robust = [c for c in timings if c[4] not in ("stream", "hint")]
            robust_validated = [c for c in validated if c[4] not in ("stream", "hint")]
            if robust:
                fallback = min(robust, key=timings.get)
            elif robust_validated:
                fallback = robust_validated[0]
            elif timings:
                fallback = min(timings, key=timings.get)
            else:
                fallback = (
                    measured_seed
                    if measured_seed in validated
                    else next(iter(validated))
                )
            best = fallback
            final_status = (
                "warm_only_time_budget"
                if best in timings
                else "warmup_only_time_budget"
            )
            if timings and time_available():
                finalists = list(
                    dict.fromkeys(
                        [
                            *sorted(timings, key=timings.get)[:4],
                            *winners(),
                            measured_seed,
                        ]
                    )
                )
                finalists = [c for c in finalists if c in timings]
                finalists.sort(key=lambda c: (c[4] in ("stream", "hint"), timings[c]))
                targets = [y]
                if y.numel() * y.element_size() <= 16 * 1024 * 1024:
                    alignment_offset = (y.data_ptr() % 16) // y.element_size()
                    for _ in range(2):
                        if not time_available():
                            break
                        try:
                            storage = torch.empty(
                                y.numel() + 16, device=y.device, dtype=y.dtype
                            )
                        except torch.cuda.OutOfMemoryError:
                            break
                        targets.append(
                            storage[
                                alignment_offset : alignment_offset + y.numel()
                            ].view_as(y)
                        )
                if time_available():
                    l2_bytes = getattr(
                        torch.cuda.get_device_properties(x.device),
                        "L2_cache_size",
                        64 * 1024 * 1024,
                    )
                    if not l2_bytes or l2_bytes < 0:
                        l2_bytes = 64 * 1024 * 1024
                    try:
                        conditioning = torch.empty(
                            max(1, 2 * l2_bytes), device=x.device, dtype=torch.uint8
                        )
                    except torch.cuda.OutOfMemoryError:
                        finalists = [
                            c for c in finalists if c[4] not in ("stream", "hint")
                        ]
                    if not finalists:
                        raise RuntimeError(
                            "No candidate can be evaluated safely "
                            "without cache conditioning"
                        )
                for config in finalists:
                    values = []
                    for target in targets:
                        if not time_available():
                            break
                        candidate = prepare_candidate(config, target)
                        samples = []
                        for _ in range(3):
                            if not time_available():
                                break
                            value = sample(candidate, conditioning)
                            if value is None:
                                break
                            samples.append(value)
                        if not samples:
                            break
                        values.append(statistics.median(samples))
                    address_times[config] = values
                    completed_addresses = max(completed_addresses, len(values))
                    if len(values) == len(targets):
                        final_times[config] = max(values)
                    if not time_available():
                        break
                if final_times:
                    best = min(final_times, key=final_times.get)
                    final_status = (
                        "conditioned"
                        if conditioning is not None
                        else "normal_cache_without_conditioning"
                    )
                elif time_available():
                    final_status = "warm_only"
    finally:
        caller.wait_stream(stream)
    # A deadline never skips the user's actual output computation.
    _launch(x, w, y, best, interleaved)
    return best, {
        "config": best,
        "milliseconds": final_times.get(best, timings.get(best)),
        "seconds": time.perf_counter() - started,
        "search_mode": mode,
        "max_configs": max_configs,
        "max_seconds": max_seconds,
        "time_budget_exhausted": not time_available(),
        "candidate_budget_exhausted": max_configs is not None
        and progress["consumed"] >= max_configs,
        "consumed_configs": progress["consumed"],
        "stage_candidates": stage_counts,
        "geometry_candidates": stage_counts.get(
            "geometry", stage_counts.get("catalog", 0)
        ),
        "measured_candidates": len(timings),
        "validated_candidates": len(validated),
        "attempted_candidates": len(attempted),
        "timings_ms": timings,
        "finalist_ms": final_times,
        "finalist_status": final_status,
        "finalists_completed": len(final_times),
        "output_addresses_tested": completed_addresses,
        "cache_conditioning_bytes": 0 if conditioning is None else conditioning.numel(),
        "address_times_ms": address_times,
        "skipped": skipped,
    }


@triton.jit
def _routed_tunable(
    X,
    W,
    Y,
    TOTAL,
    M,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    INDEX64: tl.constexpr,
    DEVICE: tl.constexpr,
    ARCH: tl.constexpr,
    X_ALIGNMENT: tl.constexpr,
    W_ALIGNMENT: tl.constexpr,
    Y_ALIGNMENT: tl.constexpr,
    COMPILER_TUNING: tl.constexpr,
    FLAT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EVEN_M: tl.constexpr,
    CACHE: tl.constexpr,
    LOAD: tl.constexpr,
    STORE: tl.constexpr,
):
    # TOTAL is formed on the host, before any 32-bit device multiplication.
    # All online candidates use the established sigmoid and FP32 ordering.
    if FLAT:
        _routed_flat(
            X,
            W,
            Y,
            TOTAL,
            N,
            S,
            WEIGHT,
            INTERLEAVED,
            BN,
            INDEX64,
            CACHE,
            LOAD,
            0,
            STORE,
        )
    else:
        _routed_tile(
            X,
            W,
            Y,
            M,
            N,
            S,
            WEIGHT,
            INTERLEAVED,
            BM,
            BN,
            EVEN_M,
            INDEX64,
            CACHE,
            LOAD,
            0,
            STORE,
        )


def _native_config(flat, bm, bn, warps, load, *, cache="", store="", stages=1):
    # Do not pass maxnreg, including None: some deployed backends reject it.
    return triton.Config(
        dict(
            FLAT=flat,
            BM=bm,
            BN=bn,
            LOAD=load,
            CACHE=cache,
            STORE=store,
            enable_fp_fusion=False,
        ),
        num_warps=warps,
        num_stages=stages,
        num_ctas=1,
    )


def _native_configs():
    """90 initial candidates, plus at most one metadata-selected geometry."""
    geometries = (
        (1, 1, 64, 1),
        (1, 1, 128, 1),
        (1, 1, 256, 1),
        (1, 1, 256, 4),
        (1, 1, 512, 2),
        (1, 1, 512, 4),
        (1, 1, 512, 8),
        (1, 1, 1024, 4),
        (1, 1, 1024, 8),
        (1, 1, 2048, 4),
        (1, 1, 4096, 4),
        (1, 1, 4096, 8),
        (1, 1, 8192, 8),
        (1, 1, 16384, 8),
        (0, 1, 64, 1),
        (0, 1, 128, 1),
        (0, 1, 256, 2),
        (0, 1, 512, 1),
        (0, 1, 512, 2),
        (0, 1, 512, 4),
        (0, 1, 1024, 4),
        (0, 1, 2048, 4),
        (0, 1, 4096, 4),
        (0, 1, 8192, 8),
        (0, 1, 16384, 8),
        (0, 2, 256, 2),
        (0, 2, 512, 4),
        (0, 2, 1024, 4),
        (0, 4, 256, 4),
        (0, 4, 1024, 8),
        (0, 8, 128, 4),
        (0, 8, 256, 4),
        (0, 16, 128, 4),
        (0, 32, 64, 4),
        (0, 64, 64, 4),
        (0, 128, 64, 8),
    )
    configs = [
        _native_config(flat, bm, bn, warps, load)
        for flat, bm, bn, warps in geometries
        for load in (0, 1)
    ]
    # Search one coordinate at a time around two anchors; never form the
    # full geometry x cache x store x compiler product on the first call.
    # LOAD=-1 means the layout's natural load: merged interleaved, split otherwise.
    for flat in (0, 1):
        for cache in (".ca", ".cg", "stream", "hint"):
            configs.append(_native_config(flat, 1, 1024, 4, -1, cache=cache))
        for store in (".cs", ".wt"):
            configs.append(_native_config(flat, 1, 1024, 4, -1, store=store))
        for stages in (2, 3, 4):
            configs.append(_native_config(flat, 1, 1024, 4, -1, stages=stages))
    return configs


def _checked_backend():
    from triton.compiler.compiler import make_backend

    target = triton.runtime.driver.active.get_current_target()
    return target, make_backend(target)


def _native_grid(meta):
    return (
        (
            triton.cdiv(meta["M"] * meta["N"], meta["BN"])
            if meta["FLAT"]
            else triton.cdiv(meta["M"], meta["BM"]) * triton.cdiv(meta["N"], meta["BN"])
        ),
    )


def _prune_native(configs, named_args, **kwargs):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up this autotune signature before CUDA graph capture")
    args = {**named_args, **kwargs}
    target, backend = _checked_backend()
    m, n = args["M"], args["N"]
    natural_load = 0 if args["INTERLEAVED"] else 1
    # Keep the previous metadata-selected geometry as a seed, but do not
    # import its approximate sigmoid or compiler arithmetic into online tuning.
    selected = _normalize_config(
        _select_measured_config(args["X"], args["W"], args["Y"], args["INTERLEAVED"]),
        args["INTERLEAVED"],
    )
    seed = _native_config(
        int(selected[0] == "flat"),
        selected[1],
        selected[2],
        selected[3],
        selected[5],
        cache=selected[4],
        store=selected[7],
    )
    result, seen, checked = [], set(), set()
    for candidate in (seed, *configs):
        meta = dict(candidate.kwargs)
        flat, bm, bn = meta["FLAT"], meta["BM"], meta["BN"]
        if bm > triton.next_power_of_2(m):
            continue
        extent = m * n if flat else n
        if bn > max(128, triton.next_power_of_2(extent)):
            continue
        if (
            bm * bn > 16384
            or not 1 <= bm * bn / (target.warp_size * candidate.num_warps) <= 128
        ):
            continue
        if not args["COMPILER_TUNING"] and candidate.num_stages != 1:
            continue
        meta["LOAD"] = natural_load if meta["LOAD"] == -1 else meta["LOAD"]
        meta["EVEN_M"] = m % bm == 0
        profile = dict(
            num_warps=candidate.num_warps,
            num_stages=candidate.num_stages,
            num_ctas=1,
            enable_fp_fusion=False,
        )
        signature = tuple(sorted(profile.items()))
        if signature not in checked:
            parsed = backend.parse_options(profile)
            if any(
                not hasattr(parsed, k) or getattr(parsed, k) != v
                for k, v in profile.items()
            ):
                raise RuntimeError(
                    "Backend ignored or changed a compiler tuning option"
                )
            checked.add(signature)
        identity = (
            tuple(sorted(meta.items())),
            candidate.num_warps,
            candidate.num_stages,
        )
        if identity in seen:
            continue
        seen.add(identity)
        # Fresh Config objects keep shape-derived metadata out of other
        # signatures' cached configurations. No tensor is retained here.
        result.append(
            triton.Config(
                meta,
                num_warps=candidate.num_warps,
                num_stages=candidate.num_stages,
                num_ctas=1,
            )
        )
    if not result:
        raise RuntimeError("No supported autotune configurations")
    return result


def _bench_native(fn, quantiles=(0.5, 0.2, 0.8)):
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    try:
        fn()  # First compilation and ordinary launch are outside device timing.
    except (CompilationError, OutOfResources):
        return [float("inf")] * len(quantiles)
    except RuntimeError as error:
        if str(error).strip() != "PassManager::run failed":
            raise
        return [float("inf")] * len(quantiles)
    # Competition mode disables CUDA Graph. Measure bounded ordinary launches
    # directly; this includes dispatch gaps but never requires capture/replay.
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(5):
        start.record()
        for _ in range(8):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 8)
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


# Native Triton caches configuration/timing metadata only; no output cache.
_native_tuner = triton.autotune(
    configs=_native_configs(),
    key=[
        "M",
        "N",
        "S",
        "WEIGHT",
        "INTERLEAVED",
        "INDEX64",
        "DEVICE",
        "ARCH",
        "X_ALIGNMENT",
        "W_ALIGNMENT",
        "Y_ALIGNMENT",
        "COMPILER_TUNING",
    ],
    prune_configs_by={"early_config_prune": _prune_native},
    do_bench=_bench_native,
    cache_results=False,
)(_routed_tunable)


def _run_autotuned(x, weights, out, interleaved, compiler_tuning):
    target, _ = _checked_backend()
    m, width = x.shape
    n, stride = width // 2, x.stride(0)
    try:
        _native_tuner.run(
            x,
            weights,
            out,
            m * n,
            m,
            n,
            stride,
            weights is not None,
            interleaved,
            max((m - 1) * stride + width, m * n) >= 2**31,
            x.device.index,
            str(target),
            x.data_ptr() % 16,
            weights.data_ptr() % 16 if weights is not None else 0,
            out.data_ptr() % 16,
            compiler_tuning,
            grid=_native_grid,
            warmup=False,
        )
    finally:
        # Native Autotuner clears this on success; release references on errors too.
        _native_tuner.nargs = None


def _run_selected(x, weights, out, interleaved):
    config = _select_measured_config(x, weights, out, interleaved)
    _launch(x, weights, out, config, interleaved)


def silu_and_mul_routed(
    gateup_output,
    topk_weights=None,
    out_dtype=None,
    use_interleaved=True,
    *,
    out=None,
    autotune=True,
    compiler_tuning=True,
):
    """Compute ``(gate * sigmoid(gate)) * up * optional_weight``.

    Input is CUDA FP16/BF16/FP32 [M, 2*N], with contiguous columns; row
    padding is supported. Interleaved rows are [g0,u0,g1,u1,...], otherwise
    [g0,...,gN-1,u0,...,uN-1]. Weights, if supplied, are contiguous [M].
    The optional contiguous output must not share storage with either input.
    All intermediates are FP32. Empty rows/columns return an empty output.
    Native autotuning uses ordinary launches and Events, with no CUDA Graph
    dependency. It is enabled by default, with up to 91 candidates before
    shape pruning. Compiler tuning adds pipeline-stage choices. Both split and
    interleaved loads, flat blocks and single-/multi-row tiles are searched.
    Only configuration metadata is cached; each call recomputes out.
    compiler_tuning=False retains geometry/load/policy tuning at one stage.
    autotune=False uses the previous offline-measured/static configuration.
    Warm up each signature before CUDA graph capture. No PDL is used.
    """
    if not isinstance(autotune, bool) or not isinstance(compiler_tuning, bool):
        raise TypeError("autotune and compiler_tuning must be bool")
    x = gateup_output
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("gateup_output must be a two-dimensional tensor")
    if x.device.type != "cuda" or x.dtype not in supported:
        raise ValueError("gateup_output must be CUDA FP16, BF16, or FP32")
    m, width = x.shape
    if width % 2 or (x.numel() and x.stride(1) != 1):
        raise ValueError("gateup_output needs an even width and contiguous columns")
    if not isinstance(use_interleaved, bool):
        raise TypeError("use_interleaved must be bool")
    n = width // 2
    dtype = x.dtype if out_dtype is None else out_dtype
    if dtype not in supported:
        raise ValueError("out_dtype must be FP16, BF16, or FP32")
    if topk_weights is not None:
        w = topk_weights
        if (
            not isinstance(w, torch.Tensor)
            or w.shape != (m,)
            or not w.is_contiguous()
            or w.device != x.device
            or w.dtype not in supported
        ):
            raise ValueError("topk_weights must be contiguous [M] on the input device")
    allocated_output = out is None
    if allocated_output:
        out = torch.empty((m, n), device=x.device, dtype=dtype)
    elif (
        not isinstance(out, torch.Tensor)
        or out.shape != (m, n)
        or out.dtype != dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must have shape [M,N], requested dtype, and be contiguous"
        )
    if m and n:
        if not allocated_output:
            output_storage = out.untyped_storage().data_ptr()
            if output_storage == x.untyped_storage().data_ptr() or (
                topk_weights is not None
                and output_storage == topk_weights.untyped_storage().data_ptr()
            ):
                raise ValueError("out must not share storage with input or weights")
        if torch.cuda.current_device() == x.device.index:
            if autotune:
                _run_autotuned(x, topk_weights, out, use_interleaved, compiler_tuning)
            else:
                _run_selected(x, topk_weights, out, use_interleaved)
        else:
            with torch.cuda.device(x.device):
                if autotune:
                    _run_autotuned(
                        x, topk_weights, out, use_interleaved, compiler_tuning
                    )
                else:
                    _run_selected(x, topk_weights, out, use_interleaved)
    return out


# Compatibility spelling for callers mirroring SGLang's function name.
silu_and_mul = silu_and_mul_routed
