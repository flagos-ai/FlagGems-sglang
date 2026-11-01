# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX MACA routed SiLU/multiply, with FP32 intermediate arithmetic.

Autotune and MACA compiler tuning are enabled by default. The native Triton
autotuner caches configuration/timing metadata only; each invocation computes
the current inputs again. The finite search has at most 57 candidates per
signature, with 64-lane warp geometry and MACA compiler/load-policy variants.
Warm up each signature before graph capture, or pass autotune=False.
No PDL, operating-system imports, or custom persistent caches are used.
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
    return tl.sigmoid(gate)


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


def _normalize_config(config):
    config = tuple(config)
    if len(config) != 9:
        raise ValueError("MetaX configs require nine entries")
    family, bm, bn, warps, load, cache, stages, scenario, fusion = config
    if family not in ("flat", "tile") or (family == "flat" and bm != 1):
        raise ValueError("Invalid kernel family or flat row tile")
    if any(type(v) is not int or v < 1 or v & (v - 1) for v in (bm, bn, warps)):
        raise ValueError("Tiles and warps must be positive powers of two")
    if warps not in (1, 2, 4, 8, 16) or bm * bn > 16384:
        raise ValueError("Configuration exceeds the finite MetaX search bounds")
    if load not in (0, 1) or cache not in ("", ".ca", ".cg"):
        raise ValueError("Unsupported load policy")
    if (
        stages not in (1, 2)
        or scenario not in ("", "storeCoalesce", "reduceSmemUsage")
        or type(fusion) is not bool
    ):
        raise ValueError("Unsupported MACA compiler option")
    return config


@triton.jit
def _routed_tunable(
    X,
    W,
    Y,
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
    LOAD: tl.constexpr,
    CACHE: tl.constexpr,
):
    # Reuse the existing FP32 arithmetic and load implementations unchanged.
    if INDEX64:
        # Widen before forming TOTAL; a runtime int32 M*N could overflow
        # before _routed_flat gets a chance to widen its TOTAL parameter.
        # tl.cast also accepts M=1 after Triton's scalar specialization.
        M = tl.cast(M, tl.int64)
    if FLAT:
        _routed_flat(
            X, W, Y, M * N, N, S, WEIGHT, INTERLEAVED, BN, INDEX64, CACHE, LOAD, 0, ""
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
            "",
        )


def _native_configs():
    """Bounded geometry plus single-coordinate compiler/load experiments."""
    configs = []

    def append(
        flat, bm, bn, warps, *, load=-1, cache="", stages=1, scenario="", fusion=True
    ):
        configs.append(
            triton.Config(
                dict(
                    FLAT=flat,
                    BM=bm,
                    BN=bn,
                    LOAD=load,
                    CACHE=cache,
                    pipeline="basic",
                    scenario=scenario,
                    enable_fp_fusion=fusion,
                ),
                num_warps=warps,
                num_stages=stages,
            )
        )

    for bn, warps in (
        (64, (1,)),
        (128, (1,)),
        (256, (1, 2, 4)),
        (512, (1, 2, 4)),
        (1024, (2, 4, 8)),
        (2048, (2, 4, 8)),
        (4096, (4, 8)),
    ):
        for warp in warps:
            append(1, 1, bn, warp)
    for bm, bn, warps in (
        (1, 128, (1,)),
        (4, 64, (1, 2)),
        (8, 128, (2,)),
        (1, 256, (1, 2)),
        (1, 512, (2, 4)),
        (1, 1024, (2, 4, 8)),
        (1, 2048, (4, 8)),
        (1, 4096, (4, 8)),
        (1, 8192, (4, 8)),
        (2, 512, (2, 4)),
        (4, 256, (2, 4)),
        (4, 1024, (4, 8)),
        (8, 512, (4, 8)),
        (16, 256, (4, 8)),
    ):
        for warp in warps:
            append(0, bm, bn, warp)
    for flat in (0, 1):
        append(flat, 1, 1024, 4, load=2)  # Opposite of the layout's default load.
        for cache in (".ca", ".cg"):
            append(flat, 1, 1024, 4, cache=cache)
        append(flat, 1, 1024, 4, stages=2)
        for scenario in ("storeCoalesce", "reduceSmemUsage"):
            append(flat, 1, 1024, 4, scenario=scenario)
        append(flat, 1, 1024, 4, fusion=False)
    return configs


def _checked_backend():
    from triton.compiler.compiler import make_backend

    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    if type(backend).__module__ != "triton.backends.metax.compiler":
        raise RuntimeError("This kernel requires the MetaX MACA Triton backend")
    return target, backend


def _native_config(options, warps, stages):
    bm, bn, flat = options["BM"], options["BN"], options["FLAT"]

    def verify_metadata(meta):
        # AABS may alter tiles: never accept a tile whose cached grid changed.
        if (meta["BM"], meta["BN"], meta["FLAT"]) != (bm, bn, flat):
            raise RuntimeError("MACA backend changed an explicitly pruned tile")
        count = (
            triton.cdiv(meta["M"] * meta["N"], bn)
            if flat
            else triton.cdiv(meta["M"], bm) * triton.cdiv(meta["N"], bn)
        )
        if meta.get("grid") != (count, 1, 1):
            raise RuntimeError("MACA launch grid no longer matches the tuned tile")

    return triton.Config(
        options, num_warps=warps, num_stages=stages, pre_hook=verify_metadata
    )


def _prune_native(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up this autotune signature before graph capture")
    _, backend = _checked_backend()
    m, n = args["M"], args["N"]
    result, checked, seen = [], set(), set()
    for candidate in configs:
        meta = dict(candidate.kwargs)
        flat, bm, bn = meta["FLAT"], meta["BM"], meta["BN"]
        if bm > triton.next_power_of_2(m):
            continue
        if bn > triton.next_power_of_2(m * n if flat else n):
            continue
        if not 1 <= bm * bn / (64 * candidate.num_warps) <= 32:
            continue
        if not args["COMPILER_TUNING"] and (
            candidate.num_stages != 1
            or meta["scenario"] != ""
            or not meta["enable_fp_fusion"]
        ):
            continue
        default_load = 0 if args["INTERLEAVED"] else 1
        meta["LOAD"] = (
            default_load
            if meta["LOAD"] == -1
            else 1 - default_load if meta["LOAD"] == 2 else meta["LOAD"]
        )
        profile = dict(
            num_warps=candidate.num_warps,
            num_stages=candidate.num_stages,
            num_ctas=1,
            pipeline=meta["pipeline"],
            scenario=meta["scenario"],
            enable_fp_fusion=meta["enable_fp_fusion"],
        )
        profile_key = tuple(sorted(profile.items()))
        if profile_key not in checked:
            parsed = backend.parse_options(profile)
            if any(
                not hasattr(parsed, k) or getattr(parsed, k) != v
                for k, v in profile.items()
            ):
                raise RuntimeError("MACA backend ignored/changed a compiler option")
            checked.add(profile_key)
        meta["EVEN_M"] = m % bm == 0
        meta["grid"] = (
            (
                triton.cdiv(m * n, bn)
                if flat
                else triton.cdiv(m, bm) * triton.cdiv(n, bn)
            ),
            1,
            1,
        )
        identity = (
            tuple(sorted(meta.items())),
            candidate.num_warps,
            candidate.num_stages,
        )
        if identity not in seen:
            result.append(
                _native_config(meta, candidate.num_warps, candidate.num_stages)
            )
            seen.add(identity)
    if not result:
        # Tiny shapes still need a legal 64-lane CTA with masked columns.
        base = dict(
            FLAT=1,
            BM=1,
            BN=64,
            LOAD=0 if args["INTERLEAVED"] else 1,
            CACHE="",
            EVEN_M=True,
            pipeline="basic",
            scenario="",
            enable_fp_fusion=True,
            grid=(triton.cdiv(m * n, 64), 1, 1),
        )
        result.append(_native_config(base, 1, 1))
    return result


def _bench_native(fn, quantiles=(0.5, 0.2, 0.8)):
    """Time ordinary MACA launches; compilation is outside the timed part."""
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    try:
        fn()
    except (CompilationError, OutOfResources):
        return [float("inf")] * len(quantiles)
    except RuntimeError as error:
        # MLIR's pass-manager exception comes from MACA compilation. Device
        # launch failures and every other runtime exception must propagate.
        if str(error).strip() != "PassManager::run failed":
            raise
        return [float("inf")] * len(quantiles)
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(8):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 8)
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


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
            m,
            n,
            stride,
            weights is not None,
            interleaved,
            max((m - 1) * stride + width, m * n) >= 2**31,
            x.device.index,
            str(target.arch),
            x.data_ptr() % 16,
            weights.data_ptr() % 16 if weights is not None else 0,
            out.data_ptr() % 16,
            compiler_tuning,
            warmup=False,
        )
    finally:
        # Native Autotuner clears transient arguments only after successful
        # calls; errors must not retain user tensor references either.
        _native_tuner.nargs = None


def _launch(x, weights, out, config, interleaved=True):
    family, bm, bn, warps, load, cache, stages, scenario, fusion = config
    m, width = x.shape
    n, stride = width // 2, x.stride(0)
    index64 = max((m - 1) * stride + width, m * n) >= 2**31
    options = dict(
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=fusion,
        pipeline="basic",
        scenario=scenario,
    )
    if family == "flat":
        return _routed_flat[((m * n + bn - 1) // bn,)](
            x,
            weights,
            out,
            m * n,
            n,
            stride,
            weights is not None,
            interleaved,
            bn,
            index64,
            cache,
            load,
            0,
            "",
            **options
        )
    return _routed_tile[(((m + bm - 1) // bm) * ((n + bn - 1) // bn),)](
        x,
        weights,
        out,
        m,
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
        0,
        "",
        **options
    )


def _select_config(x, weights, out, interleaved):
    m, width = x.shape
    n = width // 2
    load = 0 if interleaved else 1
    # This C550 table is immutable code. Padding, mixed output/weights and
    # other devices use the conservative 64-lane MACA geometry below.
    if m == 8 and n == 768 and x.dtype == torch.bfloat16:
        selected = ("tile", 1, 512, 4, load, "", 1, "", True)
    elif m == 256 and n == 1024 and x.dtype == torch.float16:
        selected = (
            "tile",
            1,
            1024,
            4,
            load,
            "",
            1,
            "" if interleaved else "reduceSmemUsage",
            True,
        )
    elif m == 1536 and n == 2048 and x.dtype == torch.bfloat16:
        selected = (
            "tile",
            1,
            512 if interleaved else 2048,
            2 if interleaved else 8,
            load,
            "",
            1,
            "",
            True,
        )
    elif m == 8192 and n == 4096 and x.dtype == torch.bfloat16:
        selected = (
            "tile",
            1 if interleaved else 8,
            4096 if interleaved else 512,
            4,
            load,
            "",
            1,
            "",
            True,
        )
    else:
        selected = None
    if (
        selected is not None
        and weights is not None
        and weights.dtype == torch.float32
        and out.dtype == x.dtype
        and x.stride(0) == width
        and x.data_ptr() % 16 == 0
        and weights.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and torch.cuda.get_device_name(x.device) == "MetaX C550"
    ):
        return selected
    return (
        "flat",
        1,
        256 if m * n <= 32768 else 1024,
        2 if m * n <= 32768 else 4,
        load,
        "",
        1,
        "",
        True,
    )


def _run_selected(x, weights, out, interleaved):
    _launch(x, weights, out, _select_config(x, weights, out, interleaved), interleaved)


def _iter_configs(m, n, interleaved):
    """Finite 64-lane geometry plus orthogonal MACA compiler/load coordinates."""
    geometry = []
    for family, bm, bn in (
        ("flat", 1, 256),
        ("flat", 1, 512),
        ("flat", 1, 1024),
        ("flat", 1, 2048),
        ("flat", 1, 4096),
        ("tile", 1, 256),
        ("tile", 1, 512),
        ("tile", 1, 1024),
        ("tile", 2, 512),
        ("tile", 4, 256),
        ("tile", 4, 1024),
        ("tile", 8, 512),
        ("tile", 1, min(16384, triton.next_power_of_2(n))),
    ):
        if bm > triton.next_power_of_2(m):
            continue
        for warps in (1, 2, 4, 8):
            if 1 <= bm * bn / (64 * warps) <= 32:
                geometry.append(
                    (family, bm, bn, warps, 0 if interleaved else 1, "", 1, "", True)
                )
    anchors = [
        ("flat", 1, 1024, 4, 0 if interleaved else 1, "", 1, "", True),
        (
            "tile",
            1,
            min(1024, triton.next_power_of_2(n)),
            4,
            0 if interleaved else 1,
            "",
            1,
            "",
            True,
        ),
    ]
    policies = []
    for anchor in anchors:
        for index, values in (
            (4, (0, 1)),
            (5, ("", ".ca", ".cg")),
            (6, (1, 2)),
            (7, ("", "storeCoalesce", "reduceSmemUsage")),
            (8, (True, False)),
        ):
            for value in values:
                policies.append((*anchor[:index], value, *anchor[index + 1 :]))
    seen = set()
    for pair in itertools.zip_longest(geometry, policies):
        for config in pair:
            if config is not None and config not in seen:
                seen.add(config)
                yield config


def _autotune(
    x,
    weights,
    out,
    interleaved=True,
    *,
    configs=None,
    max_configs=64,
    max_seconds=15.0,
    validate=None
):
    """Explicit, call-local diagnostic search; public calls use the native tuner.

    Call serially with legal nonempty inputs on x's current device, outside
    graph capture. Budgets count the mandatory seed and duplicate entries.
    The 15-second limit is soft: compilation and the final real-output launch
    cannot be interrupted. A validated but unmeasured seed reports None.
    Candidate timing uses events around ordinary launches; dispatch gaps can
    contribute to the measured time. No graph construction is required.
    Numerical checking is performed only when the caller supplies validate;
    otherwise validated_candidates counts successful ordinary launches.
    """
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    if not x.numel() or torch.cuda.current_device() != x.device.index:
        raise ValueError("Use valid nonempty input in its current device context")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Explicit tuning requires non-capturing execution")
    if max_configs is not None and (type(max_configs) is not int or max_configs < 1):
        raise ValueError("max_configs must be positive or None")
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be finite and positive or None")
    started = time.perf_counter()
    deadline = None if max_seconds is None else started + max_seconds
    seed = _select_config(x, weights, out, interleaved)

    def available():
        return deadline is None or time.perf_counter() < deadline

    m, width = x.shape
    generated = (
        _iter_configs(m, width // 2, interleaved) if configs is None else iter(configs)
    )
    stream, caller = torch.cuda.Stream(device=x.device), torch.cuda.current_stream(
        x.device
    )
    buffer = torch.empty(out.numel() + 128, dtype=out.dtype, device=out.device)
    delta = (
        (out.data_ptr() % 256 - buffer.data_ptr() % 256) % 256
    ) // out.element_size()
    target = buffer[delta : delta + out.numel()].view_as(out)
    stream.wait_stream(caller)
    measured, validated, rejected, attempted = {}, [], [], set()
    consumed = 0
    try:
        with torch.cuda.stream(stream):
            iterator = itertools.chain((seed,), generated)
            while max_configs is None or consumed < max_configs:
                if consumed and not available():
                    break
                try:
                    config = _normalize_config(next(iterator))
                except StopIteration:
                    break
                consumed += 1
                if config in attempted:
                    continue
                attempted.add(config)
                target.fill_(float("nan"))
                try:
                    _launch(x, weights, target, config, interleaved)
                except (CompilationError, OutOfResources) as error:
                    rejected.append((config, str(error)))
                    continue
                stream.synchronize()
                if validate is not None and not validate(target):
                    rejected.append((config, "numerical validation failed"))
                    continue
                validated.append(config)
                if not available():
                    break
                times = []
                for _ in range(3):
                    if not available():
                        break
                    target.fill_(float("nan"))
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    count = 0
                    begin.record(stream)
                    for _ in range(8):
                        if not available():
                            break
                        _launch(x, weights, target, config, interleaved)
                        count += 1
                    end.record(stream)
                    end.synchronize()
                    if count:
                        times.append(begin.elapsed_time(end) / count)
                        if validate is not None and not validate(target):
                            validated.pop()
                            rejected.append(
                                (config, "repeated launch validation failed")
                            )
                            times = []
                            break
                if times:
                    measured[config] = statistics.median(times)
    finally:
        caller.wait_stream(stream)
    if not validated:
        raise RuntimeError("No MetaX candidate passed validation")
    best = min(measured, key=measured.get) if measured else validated[0]
    _launch(x, weights, out, best, interleaved)
    if validate is not None and not validate(out):
        raise RuntimeError("Selected MetaX output failed final validation")
    return best, dict(
        seconds=time.perf_counter() - started,
        consumed_candidates=consumed,
        attempted_candidates=len(attempted),
        validated_candidates=len(validated),
        measured_candidates=len(measured),
        selected_ms=measured.get(best),
        time_budget_exhausted=not available(),
        numerical_validation=validate is not None,
        rejected=rejected,
        timings_ms=[(config, value) for config, value in measured.items()],
    )


def silu_and_mul_routed(
    gateup_output,
    topk_weights=None,
    out_dtype=None,
    use_interleaved=True,
    *,
    out=None,
    autotune=True,
    compiler_tuning=True
):
    """Compute ``(gate * sigmoid(gate)) * up * optional_weight``.

    Input is CUDA FP16/BF16/FP32 [M, 2*N], with contiguous columns; row
    padding is supported. Interleaved rows are [g0,u0,g1,u1,...], otherwise
    [g0,...,gN-1,u0,...,uN-1]. Weights, if supplied, are contiguous [M].
    The optional contiguous output must not share storage with either input.
    All intermediates are FP32. Empty rows/columns return an empty output.
    The first call per shape/stride/layout/dtype/device signature tunes at most
    57 configurations; later calls reuse only the selected configuration.
    compiler_tuning=False keeps the base compiler profile while tuning geometry
    and load policies. autotune=False uses the fixed metadata selector.
    Both switches default to True. Every call recomputes the current inputs.
    Warm up each signature before graph capture. No PDL is used.
    """
    if type(autotune) is not bool or type(compiler_tuning) is not bool:
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


silu_and_mul = silu_and_mul_routed


__all__ = ["silu_and_mul_routed"]
