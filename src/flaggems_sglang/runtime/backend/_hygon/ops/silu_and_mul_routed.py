# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Routed SiLU/multiply for Hygon DCU (64-lane waves).

Native autotune and HCU compiler tuning are enabled by default. Only Triton
configuration metadata is cached; every call computes its current input anew.
No PDL, operating-system imports, or custom persistent caches are used.
"""
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
    elif MODE == 3:
        return tl.div_rn(1.0, 1.0 + tl.exp2(-gate * 1.4426950408889634))
    elif MODE == 6 or MODE == 7:
        # An exact power-of-two rescaling keeps reciprocal inputs/results
        # normal for finite sigmoid denominators. Unlike modes 4/5 this needs
        # no per-element threshold/select; the final multiply restores small
        # sigmoid values with FP32 denormals preserved. Keep exp2 before the
        # scaling so overflow and special-value behavior match those modes.
        scale: tl.constexpr = 5.421010862427522e-20
        scaled_denominator = tl.fma(tl.exp2(-gate * 1.4426950408889634), scale, scale)
        reciprocal = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1",
            constraints="=v,v",
            args=[scaled_denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        if MODE == 7:
            correction = tl.fma(-scaled_denominator, reciprocal, 1.0)
            refined = tl.fma(reciprocal, correction, reciprocal)
            reciprocal = tl.where(
                scaled_denominator == float("inf"), reciprocal, refined
            )
        return reciprocal * scale
    else:
        denominator = 1.0 + tl.exp2(-gate * 1.4426950408889634)
        # v_rcp_f32 flushes subnormal results on gfx936. Scale very large
        # denominators before the reciprocal, then restore the exponent with
        # an ordinary multiply (the kernel preserves FP32 denormals).
        scale = tl.where(
            denominator >= 18446744073709551616.0, 5.421010862427522e-20, 1.0
        )
        scaled_denominator = denominator * scale
        reciprocal = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1",
            constraints="=v,v",
            args=[scaled_denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        if MODE == 5:
            correction = tl.fma(-scaled_denominator, reciprocal, 1.0)
            refined = tl.fma(reciprocal, correction, reciprocal)
            reciprocal = tl.where(denominator == float("inf"), reciprocal, refined)
        return reciprocal * scale


@triton.jit
def _routed_tile(X, W, Y, M, META: tl.constexpr):
    N: tl.constexpr = META[0]
    S: tl.constexpr = META[1]
    WEIGHT: tl.constexpr = META[2]
    INTERLEAVED: tl.constexpr = META[3]
    BM: tl.constexpr = META[4]
    BN: tl.constexpr = META[5]
    EVEN_M: tl.constexpr = META[6]
    INDEX64: tl.constexpr = META[7]
    CACHE: tl.constexpr = META[8]
    LOAD: tl.constexpr = META[9]
    SIGMOID: tl.constexpr = META[10]
    STORE: tl.constexpr = META[11]
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
    if LOAD == 2:
        packed = tl.load(
            X.to(tl.pointer_type(tl.uint32)) + rows[:, None] * (S // 2) + cols[None, :],
            mask,
            other=0,
            cache_modifier=modifier,
            eviction_policy=eviction,
        )
        gate = (packed & 65535).to(tl.uint16).to(X.dtype.element_ty, bitcast=True)
        up = (packed >> 16).to(tl.uint16).to(X.dtype.element_ty, bitcast=True)
    elif LOAD == 0:
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
def _routed_flat(X, W, Y, TOTAL, META: tl.constexpr):
    N: tl.constexpr = META[0]
    S: tl.constexpr = META[1]
    WEIGHT: tl.constexpr = META[2]
    INTERLEAVED: tl.constexpr = META[3]
    BLOCK: tl.constexpr = META[4]
    INDEX64: tl.constexpr = META[5]
    CACHE: tl.constexpr = META[6]
    LOAD: tl.constexpr = META[7]
    SIGMOID: tl.constexpr = META[8]
    STORE: tl.constexpr = META[9]
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
        TOTAL = TOTAL.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // N
    cols = offsets % N
    modifier: tl.constexpr = "" if CACHE == "hint" or CACHE == "stream" else CACHE
    eviction: tl.constexpr = "evict_first" if CACHE == "stream" else ""
    if LOAD == 2:
        packed = tl.load(
            X.to(tl.pointer_type(tl.uint32)) + rows * (S // 2) + cols,
            offsets < TOTAL,
            other=0,
            cache_modifier=modifier,
            eviction_policy=eviction,
        )
        gate = (packed & 65535).to(tl.uint16).to(X.dtype.element_ty, bitcast=True)
        up = (packed >> 16).to(tl.uint16).to(X.dtype.element_ty, bitcast=True)
    elif LOAD == 0:
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


def _launch(x, weights, out, config, interleaved=True, *, metadata=None):
    if len(config) != 9:
        return _launch_extended(x, weights, out, config, interleaved, metadata=metadata)
    # Metadata supplied by the validated public path is local to this call.
    if metadata is None:
        m, width = x.shape
        n, stride = width // 2, x.stride(0)
    else:
        m, width, n, stride = metadata[:4]
    family, bm, bn, warps, cache, load, sigmoid, store, waves = config
    if load == 2 and (
        not interleaved
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.data_ptr() % 4
        or stride % 2
    ):
        raise ValueError("Packed load requires aligned interleaved 16-bit input")
    if warps > 16:
        raise ValueError(
            "Unsupported Hygon configuration: more than 1024 threads/block"
        )
    if bm * bn > (524288 if load == 0 else 1048576):
        raise ValueError("Unsupported Hygon configuration: Triton tensor size limit")
    total = m * n
    index64 = max((m - 1) * stride + width, total) >= 2**31
    if family == "flat":
        return _routed_flat[((total + bn - 1) // bn,)](
            x,
            weights,
            out,
            total,
            (
                n,
                stride,
                weights is not None,
                interleaved,
                bn,
                index64,
                cache,
                load,
                sigmoid,
                store,
            ),
            num_warps=warps,
            num_stages=1,
            waves_per_eu=waves,
            enable_fp_fusion=False,
            allow_flush_denorm=False,
        )
    return _routed_tile[(((m + bm - 1) // bm) * ((n + bn - 1) // bn),)](
        x,
        weights,
        out,
        m,
        (
            n,
            stride,
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
        ),
        num_warps=warps,
        num_stages=1,
        waves_per_eu=waves,
        enable_fp_fusion=False,
        allow_flush_denorm=False,
    )


def _launch_extended(x, weights, out, config, interleaved=True, *, metadata=None):
    # Metadata supplied by the validated public path is local to this call.
    if metadata is None:
        m, width = x.shape
        n, stride = width // 2, x.stride(0)
    else:
        m, width, n, stride = metadata[:4]
    config = _normalize_config(config)
    reason = _config_rejection(
        config, interleaved, x.element_size(), stride, x.data_ptr()
    )
    if reason is not None:
        raise ValueError(reason)
    family, bm, bn, warps, cache, load, sigmoid, store, waves = config[:9]
    stages, ctas, fusion, epilogue, schedule, latency, single_buffer = config[9]
    compiler = dict(
        num_warps=warps,
        num_stages=stages,
        num_ctas=ctas,
        waves_per_eu=waves,
        enable_fp_fusion=fusion,
        optimize_epilogue=epilogue,
        schedule_hint=schedule,
        sched_latency=latency,
        async_copy_use_single_buffer=single_buffer,
        allow_flush_denorm=False,
    )
    if load == 2 and (
        not interleaved
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.data_ptr() % 4
        or stride % 2
    ):
        raise ValueError("Packed load requires aligned interleaved 16-bit input")
    total = m * n
    index64 = max((m - 1) * stride + width, total) >= 2**31
    if family == "flat":
        return _routed_flat[((total + bn - 1) // bn,)](
            x,
            weights,
            out,
            total,
            (
                n,
                stride,
                weights is not None,
                interleaved,
                bn,
                index64,
                cache,
                load,
                sigmoid,
                store,
            ),
            **compiler,
        )
    return _routed_tile[(((m + bm - 1) // bm) * ((n + bn - 1) // bn),)](
        x,
        weights,
        out,
        m,
        (
            n,
            stride,
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
        ),
        **compiler,
    )


def _select_config(x, weights, out, interleaved, *, metadata=None):
    if metadata is None:
        m, width = x.shape
        n = width // 2
        stride, dtype, output_dtype = x.stride(0), x.dtype, out.dtype
    else:
        m, width, n, stride, dtype, output_dtype = metadata
    # Immutable literals avoid rebuilding a local dict on every invocation.
    # Only metadata is reused within this call; every invocation launches.
    if m == 17 and n == 513 and dtype == torch.float16 and interleaved is False:
        selected = ("tile", 1, 256, 2, ".cv", 0, 2, "", 0)
    elif m == 1536 and n == 3072 and dtype == torch.float16 and interleaved is False:
        selected = ("tile", 1, 1024, 4, ".cv", 1, 6, ".cs", 0)
    elif m == 1536 and n == 3072 and dtype == torch.float16 and interleaved is True:
        selected = ("tile", 1, 1024, 4, ".ca", 0, 6, ".wt", 0)
    elif m == 1 and n == 2048 and dtype == torch.bfloat16 and interleaved is True:
        selected = ("flat", 1, 512, 8, "", 0, 6, ".cs", 0)
    elif m == 256 and n == 1024 and dtype == torch.float32 and interleaved is False:
        selected = ("tile", 2, 1024, 8, ".cv", 1, 6, ".wt", 0)
    elif m == 256 and n == 1024 and dtype == torch.float32 and interleaved is True:
        selected = ("tile", 2, 512, 8, ".ca", 0, 6, ".wt", 0)
    elif m == 4096 and n == 64 and dtype == torch.float16 and interleaved is False:
        selected = ("flat", 1, 2048, 4, "", 1, 6, "", 4)
    elif m == 4096 and n == 64 and dtype == torch.float16 and interleaved is True:
        selected = ("tile", 32, 64, 8, ".cg", 0, 6, ".cg", 4)
    elif m == 8192 and n == 4096 and dtype == torch.bfloat16 and interleaved is False:
        selected = ("tile", 2, 1024, 4, ".ca", 0, 4, ".wt", 0)
    elif m == 8192 and n == 4096 and dtype == torch.bfloat16 and interleaved is True:
        selected = ("flat", 1, 8192, 16, ".cv", 0, 6, ".wt", 0)
    elif m == 128 and n == 8192 and dtype == torch.bfloat16 and interleaved is True:
        selected = ("tile", 2, 1024, 4, "", 0, 6, "", 0)
    elif m == 128 and n == 8192 and dtype == torch.bfloat16 and interleaved is False:
        selected = ("tile", 1, 2048, 4, "", 0, 6, "", 0)
    else:
        selected = None
    if (
        selected is not None
        and stride == width
        and x.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and output_dtype == dtype
        and (
            weights is None
            or (weights.dtype == torch.float32 and weights.data_ptr() % 16 == 0)
        )
    ):
        return selected
    return (
        "tile",
        1,
        min(1024, triton.next_power_of_2(n)),
        4,
        "",
        int(not interleaved),
        6,
        "",
        0,
    )


def _staged_geometry_configs(m, n, interleaved):
    configs = []
    total = m * n
    for bn in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        if bn > max(256, triton.next_power_of_2(total)):
            continue
        for warps in (1, 2, 4, 8, 16):
            # Hygon uses 64-lane waves, not NVIDIA's 32-lane warps.
            if not 1 <= bn / (64 * warps) <= 32:
                continue
            for load in (0, 1):
                configs.append(("flat", 1, bn, warps, "", load, 0, "", 0))
    for bm in (1, 2, 4, 8, 16, 32, 64):
        if bm > triton.next_power_of_2(m):
            continue
        for bn in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
            area = bm * bn
            if area < 128 or area > 32768 or bn > max(64, triton.next_power_of_2(n)):
                continue
            if total >= 32768 and triton.cdiv(m, bm) * triton.cdiv(n, bn) < 40:
                continue
            preferred = min(8, max(1, triton.next_power_of_2(triton.cdiv(area, 512))))
            for warps in sorted(
                {max(1, preferred // 2), preferred, min(16, preferred * 2)}
            ):
                if not 1 <= area / (64 * warps) <= 64:
                    continue
                for load in (0, 1):
                    configs.append(("tile", bm, bn, warps, "", load, 0, "", 0))
    return list(dict.fromkeys(configs))


def _geometry_configs(m, n, interleaved, *, exhaustive=False, element_size=2):
    """A finite, moderate geometry catalog, including packed 16-bit loads.

    Both flag values now select this bounded catalog; exhaustive is retained
    only for compatibility with earlier private callers. Flat blocks stop at
    32768, tiles at 32768 elements/BN16384, BM64, and 16 waves. Density limits
    avoid the high-spill giant blocks seen in exploratory tests.
    """
    loads = (0, 1, 2) if interleaved and element_size == 2 else (0, 1)
    configs = []
    for config in _staged_geometry_configs(m, n, interleaved):
        family, bm, bn, warps = config[:4]
        if bm * bn <= 32768 and bm * bn <= 32 * 64 * warps:
            configs.append(config)
            if config[5] == 0 and 2 in loads:
                configs.append((*config[:5], 2, *config[6:]))
    for family, bm, bn, warps in (
        ("flat", 1, 64, 1),
        ("tile", 1, 16384, 8),
        ("tile", 1, 16384, 16),
        ("tile", 2, 16384, 16),
    ):
        extent = m * n if family == "flat" else n
        if bn > max(256, triton.next_power_of_2(extent)) or bm > triton.next_power_of_2(
            m
        ):
            continue
        for load in loads:
            configs.append((family, bm, bn, warps, "", load, 0, "", 0))
    # Alternate family/load groups so a small budget reaches both families.
    groups = [
        iter([c for c in configs if c[0] == family and c[5] == load])
        for family in ("flat", "tile")
        for load in loads
    ]
    import itertools

    ordered = (
        c for row in itertools.zip_longest(*groups) for c in row if c is not None
    )
    return list(dict.fromkeys(ordered))


def _normalize_config(config):
    """Canonical immutable 9-item or extended 10-item configuration."""
    config = tuple(config)
    if len(config) not in (9, 10):
        raise ValueError("Hygon configs require 9 items or a 10th compiler tuple")
    family, bm, bn, warps, cache, load, sigmoid, store, waves = config[:9]
    if family not in ("flat", "tile"):
        raise ValueError("Unknown kernel family")
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 1 or v & (v - 1)
        for v in (bm, bn, warps)
    ):
        raise ValueError("BM, BN, and num_warps must be positive powers of two")
    if family == "flat" and bm != 1:
        raise ValueError("Flat configurations require BM=1")
    if cache not in ("", ".ca", ".cg", ".cv", "hint", "stream"):
        raise ValueError("Unknown load-cache policy")
    if store not in ("", ".wb", ".cg", ".cs", ".wt"):
        raise ValueError("Unknown store-cache policy")
    if load not in (0, 1, 2) or sigmoid not in range(8) or waves not in range(11):
        raise ValueError("Unsupported load/math/waves configuration")
    if len(config) == 10:
        compiler = tuple(config[9])
        if len(compiler) != 7:
            raise ValueError("Compiler tuple requires seven entries")
        stages, ctas, fusion, epilogue, schedule, latency, single_buffer = compiler
        if stages not in range(1, 9) or ctas not in (1, 2, 4, 8):
            raise ValueError("Unsupported stage count or CTA-cluster size")
        if not all(isinstance(v, bool) for v in (fusion, epilogue, single_buffer)):
            raise TypeError("Compiler fusion/epilogue/async options must be bool")
        if schedule not in (
            "none",
            "local-prefetch",
            "attention",
            "memory-bound-attention",
            "attention,memory-bound-attention",
        ):
            raise ValueError("Unsupported schedule hint")
        if latency not in ("none", "mmac5-ds10", "mmac5-ds6"):
            raise ValueError("Unsupported scheduling latency")
        config = (*config[:9], compiler)
    return config


def _config_rejection(config, interleaved, element_size, stride, address):
    """Legal metadata/resource limits; no timing-based pruning."""
    _, bm, bn, warps, _, load, _, _, _ = config[:9]
    if warps * 64 > 1024:
        return "unsupported: gfx936 has 64-lane waves and at most 1024 threads/block"
    if len(config) == 10 and config[9][1] != 1:
        return "unsupported: gfx936 HCU backend does not support num_ctas > 1"
    if bm * bn * (2 if load == 0 else 1) > 1048576:
        return "unsupported: Triton tensor exceeds 1048576 elements"
    if load == 2 and (
        not interleaved or element_size != 2 or stride % 2 or address % 4
    ):
        return "unsupported: packed load requires interleaved 16-bit data, even row stride, and 4-byte alignment"
    return None


def _iter_configs(
    m,
    n,
    interleaved,
    *,
    element_size=2,
    geometries=None,
    math_modes=(0, 1, 2, 3, 4, 5, 6, 7),
    caches=("", ".ca", ".cg", ".cv", "hint", "stream"),
    stores=("", ".wb", ".cg", ".cs", ".wt"),
    waves=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    stages=(1, 2, 3, 4),
    ctas=(1,),
    fusion=(False, True),
    epilogue=(True, False),
    schedules=(
        "none",
        "local-prefetch",
        "memory-bound-attention",
        "attention,memory-bound-attention",
    ),
    latencies=("none", "mmac5-ds10", "mmac5-ds6"),
    single_buffer=(True, False),
):
    """Finite geometry plus orthogonal policy coordinates, never a product.

    Geometry/math/cache/store/waves/compiler streams alternate, so a small
    budget reaches several dimensions. Typical catalogs contain hundreds of
    entries, roughly <=1000, with all deduplication state local to this call.
    Known illegal 32-wave/cluster launches and exploratory giant geometries
    are absent from defaults. Explicit custom configs remain supported.
    """
    import itertools

    geometry = (
        _geometry_configs(m, n, interleaved, element_size=element_size)
        if geometries is None
        else geometries
    )
    loads = (0, 1, 2) if interleaved and element_size == 2 else (0, 1)
    # These modest anchors cover each family/load implementation independently
    # of whether a neighboring geometry happened to win an earlier sample.
    bn = min(1024, triton.next_power_of_2(n))
    tile_warps = min(4, max(1, bn // 64))
    anchors = [
        (
            family,
            1,
            1024 if family == "flat" else bn,
            4 if family == "flat" else tile_warps,
            "",
            load,
            0,
            "",
            0,
        )
        for family in ("flat", "tile")
        for load in loads
    ]

    def coordinate(index, values):
        # Cycle values before repeating an anchor, then rotate the anchors.
        # A short budget therefore reaches all math/cache/waves choices.
        for offset in range(len(anchors)):
            for value_index, value in enumerate(values):
                config = anchors[(value_index + offset) % len(anchors)]
                yield (*config[:index], value, *config[index + 1 :])

    def compiler_coordinates():
        default = (1, 1, False, True, "none", "none", True)
        dimensions = (
            stages,
            ctas,
            fusion,
            epilogue,
            schedules,
            latencies,
            single_buffer,
        )
        groups = [
            iter(
                [
                    (*default[:index], value, *default[index + 1 :])
                    for value in values
                    if value != default[index]
                ]
            )
            for index, values in enumerate(dimensions)
        ]
        # Stage/fusion/epilogue/hint/latency/async changes all appear early,
        # before spending the budget on more anchors of one dimension.
        options = [
            option
            for row in itertools.zip_longest(*groups)
            for option in row
            if option is not None
        ]
        options.append(default)
        for offset in range(len(anchors)):
            for option_index, compiler in enumerate(options):
                config = anchors[(option_index + offset) % len(anchors)]
                yield (*config, compiler)

    streams = (
        iter(geometry),
        coordinate(6, math_modes),
        coordinate(4, caches),
        coordinate(7, stores),
        coordinate(8, waves),
        compiler_coordinates(),
    )
    seen = set()
    for row in itertools.zip_longest(*streams):
        for config in row:
            if config is not None and config not in seen:
                seen.add(config)
                yield config


def _compiler_variants(config):
    """Moderate coordinate set; no clustered launch or giant search product."""
    default = (1, 1, False, True, "none", "none", True)
    dimensions = (
        (1, 2, 3, 4),
        (1,),
        (False, True),
        (True, False),
        (
            "none",
            "local-prefetch",
            "memory-bound-attention",
            "attention,memory-bound-attention",
        ),
        ("none", "mmac5-ds10", "mmac5-ds6"),
        (True, False),
    )
    for index, values in enumerate(dimensions):
        for value in values:
            yield (*config[:9], (*default[:index], value, *default[index + 1 :]))


def _neighbor_configs(config, m, n):
    """Small nearby geometries under the same moderate density limits."""
    family, bm, bn, warps, cache, load, sigmoid, store, waves = config[:9]
    configs = []
    for next_bm in ((1,) if family == "flat" else (bm // 2, bm, bm * 2)):
        for next_bn in (bn // 2, bn, bn * 2):
            for next_warps in (warps // 2, warps, warps * 2):
                if next_warps not in (1, 2, 4, 8, 16) or not 1 <= next_bm <= 64:
                    continue
                if not 64 <= next_bn <= (32768 if family == "flat" else 16384):
                    continue
                area = next_bm * next_bn
                if not 1 <= area / (64 * next_warps) <= 32 or area > 32768:
                    continue
                extent = m * n if family == "flat" else n
                if next_bn > max(256, triton.next_power_of_2(extent)):
                    continue
                configs.append(
                    (
                        family,
                        next_bm,
                        next_bn,
                        next_warps,
                        cache,
                        load,
                        sigmoid,
                        store,
                        waves,
                        *config[9:],
                    )
                )
    return list(dict.fromkeys(configs))


def _autotune(
    x,
    weights,
    out,
    interleaved=True,
    *,
    validate=None,
    search="staged",
    configs=None,
    max_configs=64,
    max_seconds=15.0,
):
    """Explicit bounded launch/Event tuning with no retained cross-call state.

    The seed runs first and counts toward both budgets. Candidate entries,
    including duplicates, count toward max_configs. max_seconds is a soft
    wall-clock limit starting before seed compilation; a single compilation
    or in-flight operation cannot be preempted. Candidates, samples, and
    finalists check the deadline. A validated seed remains usable when its
    compile consumes the budget, with timing recorded as None if unmeasured.
    The chosen configuration always recomputes the real output before return.

    Staged and catalog modes use a moderate finite geometry/orthogonal-policy
    stream. 'exhaustive' is a compatibility alias for that finite catalog,
    not a Cartesian product. configs supplies a custom iterable. Setting a
    budget to None disables that budget explicitly. Run serially on nonempty
    inputs satisfying the public contract, outside graph capture. validate
    should return a bool; runtime errors propagate, compiler/resource failures
    are recorded. Ordinary public calls never invoke this helper.
    """
    import math

    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import HSACOError, OutOfResources

    if not x.numel() or not out.numel():
        raise ValueError("Autotuning requires nonempty tensors")
    if search not in ("staged", "catalog", "exhaustive"):
        raise ValueError("search must be staged, catalog, or exhaustive")
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
        raise ValueError("max_seconds must be finite and positive, or None")
    started = time.perf_counter()
    deadline = None if max_seconds is None else started + max_seconds
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Autotuning must run outside HIP graph capture")
        m, width = x.shape
        n = width // 2
        seed = _normalize_config(_select_config(x, weights, out, interleaved))
        mode = "custom" if configs is not None else search
        candidates = (
            iter(configs)
            if configs is not None
            else _iter_configs(m, n, interleaved, element_size=x.element_size())
        )
        padding = 256 // out.element_size()
        storage = torch.empty(out.numel() + padding, device=out.device, dtype=out.dtype)
        offset = ((out.data_ptr() - storage.data_ptr()) % 256) // out.element_size()
        temporary = storage[offset : offset + out.numel()].view(out.shape)
        times, rejected, attempted, validated = {}, [], set(), {}
        progress = {"consumed": 0}
        repetitions = 16 if out.numel() < 8 * 1024 * 1024 else 4
        stream = torch.cuda.Stream(device=x.device)
        caller = torch.cuda.current_stream(x.device)
        stream.wait_stream(caller)

        def timed_out():
            return deadline is not None and time.perf_counter() >= deadline

        def budget_available():
            return (
                max_configs is None or progress["consumed"] < max_configs
            ) and not timed_out()

        def prepare_candidate(config):
            if validate is not None:
                temporary.fill_(float("nan"))
            _launch(x, weights, temporary, config, interleaved)
            stream.synchronize()
            if validate is not None and not validate(temporary):
                validated.pop(config, None)
                return False
            validated[config] = True
            return True

        def sample(config):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            count = 0
            for _ in range(repetitions):
                if count and timed_out():
                    break
                _launch(x, weights, temporary, config, interleaved)
                count += 1
            end.record(stream)
            end.synchronize()
            return begin.elapsed_time(end) / count

        def measure(config, force_seed=False):
            if (
                not force_seed
                and max_configs is not None
                and progress["consumed"] >= max_configs
            ):
                return
            progress["consumed"] += 1
            if not force_seed and timed_out():
                return
            config = _normalize_config(config)
            if config in attempted:
                return
            attempted.add(config)
            reason = _config_rejection(
                config, interleaved, x.element_size(), x.stride(0), x.data_ptr()
            )
            if reason is not None:
                rejected.append((config, reason))
                return
            try:
                if not prepare_candidate(config):
                    rejected.append((config, "numerical validation"))
                    return
                samples = []
                for _ in range(3):
                    if timed_out():
                        break
                    samples.append(sample(config))
                if samples:
                    times[config] = statistics.median(samples)
            except (OutOfResources, CompilationError, HSACOError) as error:
                rejected.append((config, "compile/resource: " + str(error)))
            except RuntimeError as error:
                if str(error).strip() != "PassManager::run failed":
                    raise
                rejected.append((config, "compiler pass: " + str(error)))

        try:
            with torch.cuda.stream(stream):
                measure(seed, force_seed=True)
                while budget_available():
                    try:
                        config = next(candidates)
                    except StopIteration:
                        break
                    measure(config)
                if not validated:
                    raise RuntimeError(
                        "No Hygon candidate passed validation before the budget expired"
                    )
                # Retest only a few measured leaders, and keep coarse timings
                # when the deadline leaves no time for finalist sampling.
                final = {}
                for config in sorted(times, key=times.get)[:3]:
                    if timed_out():
                        break
                    if not prepare_candidate(config):
                        rejected.append((config, "finalist numerical validation"))
                        times.pop(config, None)
                        continue
                    values = []
                    for _ in range(3):
                        if timed_out():
                            break
                        values.append(sample(config))
                    if values:
                        final[config] = statistics.median(values)
                measured = {
                    c: final.get(c, value)
                    for c, value in times.items()
                    if c in validated
                }
                if measured:
                    best = min(measured, key=measured.get)
                elif validated:
                    best = next(iter(validated))
                else:
                    raise RuntimeError(
                        "No Hygon candidate remained valid after finalist checks"
                    )
        finally:
            caller.wait_stream(stream)
        if validate is not None:
            out.fill_(float("nan"))
        _launch(x, weights, out, best, interleaved)
        if validate is not None:
            caller.synchronize()
            if not validate(out):
                raise RuntimeError(
                    "Selected Hygon configuration failed output validation"
                )
    return best, {
        "seconds": time.perf_counter() - started,
        "search_mode": mode,
        "max_configs": max_configs,
        "max_seconds": max_seconds,
        "time_budget_exhausted": timed_out(),
        "consumed_candidates": progress["consumed"],
        "attempted_candidates": len(attempted),
        "measured_candidates": len(times),
        "validated_candidates": len(validated),
        "geometry_candidates": len(attempted),
        "stage_candidates": {"interleaved_catalog": len(attempted)},
        "timings_ms": times,
        "finalist_ms": final,
        "selected_ms": measured.get(best),
        "rejected": rejected,
    }


@triton.jit
def _routed_tunable(
    X,
    W,
    Y,
    TOTAL,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    INDEX64: tl.constexpr,
    PACKED_OK: tl.constexpr,
    DEVICE: tl.constexpr,
    ARCH: tl.constexpr,
    X_ALIGNMENT: tl.constexpr,
    W_ALIGNMENT: tl.constexpr,
    Y_ALIGNMENT: tl.constexpr,
    COMPILER_TUNING: tl.constexpr,
    FLAT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CACHE: tl.constexpr,
    LOAD: tl.constexpr,
    SIGMOID: tl.constexpr,
    STORE: tl.constexpr,
):
    # Reuse the original arithmetic and load implementations unchanged.
    if FLAT:
        _routed_flat(
            X,
            W,
            Y,
            TOTAL,
            (N, S, WEIGHT, INTERLEAVED, BN, INDEX64, CACHE, LOAD, SIGMOID, STORE),
        )
    else:
        _routed_tile(
            X,
            W,
            Y,
            M,
            (
                N,
                S,
                WEIGHT,
                INTERLEAVED,
                BM,
                BN,
                M % BM == 0,
                INDEX64,
                CACHE,
                LOAD,
                SIGMOID,
                STORE,
            ),
        )


def _native_profiles():
    # These are HCU compiler coordinates, not XPU or NVIDIA-only options.
    base = dict(
        waves_per_eu=0,
        optimize_epilogue=False,
        schedule_hint="none",
        sched_latency="none",
        async_copy_use_single_buffer=True,
        enable_fp_fusion=False,
        allow_flush_denorm=False,
        num_stages=1,
    )
    return (
        base,
        *({**base, "waves_per_eu": waves} for waves in (1, 2, 4)),
        {**base, "optimize_epilogue": True},
        {**base, "schedule_hint": "local-prefetch"},
        {**base, "schedule_hint": "memory-bound-attention"},
        {**base, "num_stages": 2},
    )


def _native_config(meta, warps, profile):
    bm, bn, flat = meta["BM"], meta["BN"], meta["FLAT"]

    def check_grid(args):
        if (args["BM"], args["BN"], args["FLAT"]) != (bm, bn, flat):
            raise RuntimeError("Backend changed an explicitly pruned Hygon tile")
        count = (
            triton.cdiv(args["M"] * args["N"], bn)
            if flat
            else triton.cdiv(args["M"], bm) * triton.cdiv(args["N"], bn)
        )
        if args.get("grid") != (count, 1, 1):
            raise RuntimeError("Hygon grid no longer matches its tuned tile")

    options = {**meta, **profile}
    stages = options.pop("num_stages")
    return triton.Config(
        options, num_warps=warps, num_stages=stages, num_ctas=1, pre_hook=check_grid
    )


def _native_configs():
    # 61 entries before shape/alignment pruning. Do not form the full product
    # of geometry, compiler, load and cache coordinates during online tuning.
    profiles = _native_profiles()
    result = []
    geometries = (
        (0, 1, 128, 1),
        (0, 1, 256, 2),
        (0, 1, 512, 2),
        (0, 1, 1024, 4),
        (0, 1, 2048, 4),
        (0, 1, 4096, 8),
        (0, 2, 256, 2),
        (0, 2, 1024, 4),
        (0, 4, 256, 4),
        (0, 4, 512, 4),
        (0, 16, 128, 4),
        (0, 32, 64, 4),
        (1, 1, 256, 1),
        (1, 1, 512, 2),
        (1, 1, 1024, 4),
        (1, 1, 2048, 4),
        (1, 1, 4096, 8),
        (1, 1, 8192, 8),
    )
    for flat, bm, bn, warps in geometries:
        loads = (
            (0, 1, 2)
            if (
                (flat and bn in (512, 2048, 8192))
                or (not flat and bm == 1 and bn in (1024, 4096))
            )
            else (0, 1)
        )
        for load in loads:
            meta = dict(
                FLAT=flat, BM=bm, BN=bn, CACHE="", LOAD=load, SIGMOID=6, STORE=""
            )
            result.append(_native_config(meta, warps, profiles[0]))
    for flat, bm, bn, load in ((1, 1, 2048, 0), (0, 1, 1024, 1)):
        meta = dict(FLAT=flat, BM=bm, BN=bn, CACHE="", LOAD=load, SIGMOID=6, STORE="")
        for profile in profiles[1:]:
            result.append(_native_config(meta, 4, profile))
        for cache in (".cg", ".ca", "stream"):
            result.append(_native_config({**meta, "CACHE": cache}, 4, profiles[0]))
    return result


def _checked_hcu_backend():
    from triton.compiler.compiler import make_backend

    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    if (
        type(backend).__module__ != "triton.backends.hcu.compiler_hcu"
        or target.warp_size != 64
    ):
        raise RuntimeError("This tuner requires the 64-lane Hygon HCU backend")
    return target, backend


def _prune_native(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up this autotune signature before HIP graph capture")
    _, backend = _checked_hcu_backend()
    m, n = args["M"], args["N"]
    base, result, checked = _native_profiles()[0], [], []
    kernel_keys = ("FLAT", "BM", "BN", "CACHE", "LOAD", "SIGMOID", "STORE")
    for config in configs:
        meta = config.kwargs
        flat, bm, bn = meta["FLAT"], meta["BM"], meta["BN"]
        if bm > triton.next_power_of_2(max(m, 1)):
            continue
        extent = m * n if flat else n
        if bn > max(256, triton.next_power_of_2(max(extent, 1))):
            continue
        if meta["LOAD"] == 2 and not args["PACKED_OK"]:
            continue
        options = config.all_kwargs()
        profile = {
            key: value for key, value in options.items() if key not in kernel_keys
        }
        if not args["COMPILER_TUNING"] and any(
            profile[key] != value for key, value in base.items()
        ):
            continue
        profile_key = tuple(sorted(profile.items()))
        if profile_key not in checked:
            parsed = backend.parse_options(profile)
            if any(
                not hasattr(parsed, key) or getattr(parsed, key) != value
                for key, value in profile.items()
            ):
                raise RuntimeError("HCU backend ignored or changed a compiler option")
            checked.append(profile_key)
        count = (
            triton.cdiv(m * n, bn) if flat else triton.cdiv(m, bm) * triton.cdiv(n, bn)
        )
        # Never mutate a Config shared by another shape's native cache entry.
        fresh = {key: meta[key] for key in kernel_keys}
        fresh["grid"] = (count, 1, 1)
        result.append(
            _native_config(
                fresh,
                config.num_warps,
                {
                    key: value
                    for key, value in profile.items()
                    if key not in ("num_warps", "num_ctas")
                },
            )
        )
    if not result:
        raise RuntimeError("No legal Hygon autotune configurations")
    return result


def _bench_native(fn, quantiles=(0.5, 0.2, 0.8)):
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import HSACOError, OutOfResources

    try:
        fn()  # Compile once before timing ordinary device launches.
    except (CompilationError, HSACOError, OutOfResources):
        return [float("inf")] * len(quantiles)
    except RuntimeError as error:
        # Only this compiler diagnostic is a rejected configuration. Device
        # runtime errors, including failures during synchronization, propagate.
        if str(error).strip() != "PassManager::run failed":
            raise
        return [float("inf")] * len(quantiles)
    torch.cuda.synchronize()
    repeat = 8
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeat)
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


def _make_native_tuner():
    return triton.autotune(
        configs=_native_configs(),
        key=[
            "M",
            "N",
            "S",
            "WEIGHT",
            "INTERLEAVED",
            "INDEX64",
            "PACKED_OK",
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


# Only native Triton configuration/timing metadata is cached, never outputs.
_native_tuner = _make_native_tuner()


def _run_native(x, weights, out, interleaved, compiler_tuning, metadata):
    m, width, n, stride = metadata[:4]
    target, _ = _checked_hcu_backend()
    index64 = max((m - 1) * stride + width, m * n) >= 2**31
    packed_ok = (
        interleaved
        and x.element_size() == 2
        and x.data_ptr() % 4 == 0
        and stride % 2 == 0
    )
    try:
        return _native_tuner.run(
            x,
            weights,
            out,
            m * n,
            m,
            n,
            stride,
            weights is not None,
            interleaved,
            index64,
            packed_ok,
            x.device.index,
            str(target.arch),
            x.data_ptr() % 16,
            weights.data_ptr() % 16 if weights is not None else 0,
            out.data_ptr() % 16,
            compiler_tuning,
            warmup=False,
        )
    finally:
        # Triton normally clears this after success; also release on errors.
        _native_tuner.nargs = None


def _run_selected(x, weights, out, interleaved):
    _launch(x, weights, out, _select_config(x, weights, out, interleaved), interleaved)


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

    Input is Hygon DCU FP16/BF16/FP32 [M, 2*N], with contiguous columns; row
    padding is supported. Interleaved rows are [g0,u0,g1,u1,...], otherwise
    [g0,...,gN-1,u0,...,uN-1]. Weights, if supplied, are contiguous [M].
    The optional contiguous output must not share storage with either input.
    All intermediates are FP32. Empty rows/columns return an empty output.
    The first metadata/dtype/device signature searches at most 61 candidates.
    Later calls reuse its native Triton configuration and recompute output.
    compiler_tuning=False retains geometry tuning with base compiler options;
    autotune=False uses the fixed selector. Both flags default to True.
    Tuning uses ordinary launches and events; no graphs or PDL are used.
    """
    if not isinstance(autotune, bool) or not isinstance(compiler_tuning, bool):
        raise TypeError("autotune and compiler_tuning must be bool")
    x = gateup_output
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("gateup_output must be a two-dimensional tensor")
    device, input_dtype = x.device, x.dtype
    if device.type != "cuda" or not torch.version.hip or input_dtype not in supported:
        raise ValueError("gateup_output must be Hygon DCU FP16, BF16, or FP32")
    m, width = x.shape
    if width % 2 or (m and width and x.stride(1) != 1):
        raise ValueError("gateup_output needs an even width and contiguous columns")
    if not isinstance(use_interleaved, bool):
        raise TypeError("use_interleaved must be bool")
    n = width // 2
    dtype = input_dtype if out_dtype is None else out_dtype
    if dtype not in supported:
        raise ValueError("out_dtype must be FP16, BF16, or FP32")
    if topk_weights is not None:
        w = topk_weights
        if (
            not isinstance(w, torch.Tensor)
            or w.shape != (m,)
            or not w.is_contiguous()
            or w.device != device
            or w.dtype not in supported
        ):
            raise ValueError("topk_weights must be contiguous [M] on the input device")
    allocated_output = out is None
    if allocated_output:
        out = torch.empty((m, n), device=device, dtype=dtype)
    elif (
        not isinstance(out, torch.Tensor)
        or out.shape != (m, n)
        or out.dtype != dtype
        or out.device != device
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
        metadata = (m, width, n, x.stride(0), input_dtype, dtype)
        with torch.cuda.device(device):
            if autotune:
                _run_native(
                    x, topk_weights, out, use_interleaved, compiler_tuning, metadata
                )
            else:
                config = _select_config(
                    x, topk_weights, out, use_interleaved, metadata=metadata
                )
                _launch(
                    x, topk_weights, out, config, use_interleaved, metadata=metadata
                )
    return out


silu_and_mul = silu_and_mul_routed
