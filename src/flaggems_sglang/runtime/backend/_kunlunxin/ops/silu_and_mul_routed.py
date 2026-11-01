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
"""Kunlunxin P800 routed SiLU-and-multiply, implemented with FlagTree Triton.

Requires a working Kunlunxin torch/FlagTree environment. The supported P800
stack presents XPU tensors through torch.cuda. There is no PyTorch fallback.
Autotune and native compiler tuning are enabled by default. Tile widths are
64, 128, 256, 512, 1024, 2048 and 4096. Triton's native autotuner retains only
configuration/timing metadata; every call computes the current input anew.
Warm up each signature before graph capture, or pass autotune=False.
No PDL is used. All arithmetic is FP32.
BF16 input uses two typed loads: packed BF16 bitcasts can trigger a missing
unpack buffer in some XPU compiler releases.
"""

import time

import torch
import triton
import triton.language as tl
from triton.backends.xpu.compiler import XPUOptions
from triton.language.extra.xpu import libdevice
from triton.runtime.errors import OutOfResources


def _compiler_profiles():
    """RMSNorm's seven native XPU profiles, varied one coordinate at a time."""
    base = {
        "buffer_size_limit": 512,
        "budget_tiling": True,
        "unroll_num": 2,
        "isCloseMemoryAsync": False,
        "enable_fp_fusion": False,
    }
    # False means asynchronous memory IS enabled. Keep FP32 operation order;
    # enabling compiler tuning does not mean setting every boolean to True.
    return (
        base,
        *({**base, "buffer_size_limit": size} for size in (256, 1024, 2048)),
        *({**base, "budget_tiling": False, "unroll_num": u} for u in (2, 4)),
        {**base, "isCloseMemoryAsync": True},
    )


def _configs():
    # XPU has 64 column cores, not CUDA warps. Search these aligned widths;
    # num_warps/num_ctas do not add effective coordinates on this backend.
    widths = (64, 128, 256, 512, 1024, 2048, 4096)
    profiles = _compiler_profiles()
    result = []
    # Base profile spans every width and both memory paths. Extra compiler
    # profiles use two anchor widths, avoiding a large Cartesian product.
    for flat, packed in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for bn in widths:
            variants = profiles if bn in (256, 4096) else profiles[:1]
            for profile in variants:
                for vectorization_off in (
                    (False, True) if not flat and not packed and bn > 256 else (False,)
                ):
                    result.append(
                        triton.Config(
                            {
                                "BM": 1,
                                "BN": bn,
                                "FLAT": flat,
                                "PACKED": packed,
                                "MATH": 0,
                                "CACHE": "",
                                "isCloseVectorization": vectorization_off,
                                **profile,
                            },
                            num_warps=None,
                            num_ctas=None,
                            num_stages=None,
                        )
                    )
    # BM is a sequential row loop in this kernel, not a two-dimensional XPU
    # vector tile. Keep a few multi-row choices as well as one CTA per row.
    for bm in (4, 16):
        for bn in (128, 512, 2048):
            result.append(
                triton.Config(
                    {
                        "BM": bm,
                        "BN": bn,
                        "FLAT": 0,
                        "PACKED": 0,
                        "MATH": 0,
                        "CACHE": "",
                        "isCloseVectorization": False,
                        **profiles[0],
                    },
                    num_warps=None,
                    num_ctas=None,
                    num_stages=None,
                )
            )
    return result


def _prune(configs, named_args, **kwargs):
    m, n = named_args["M"], named_args["N"]
    result = []
    for candidate in configs:
        c = candidate.kwargs
        flat = c.get("FLAT", 0)
        if flat:
            # This backend truncates flat BF16 stores even with explicit RTNE.
            if named_args.get("OUT_BF16", False):
                continue
            if c["BN"] > triton.next_power_of_2(max(m * n, 1)):
                continue
            # Stride-two vectorization is unreliable on this backend. Use
            # packed words for interleaved FP16 input, scalar-row otherwise.
            if named_args["INTERLEAVED"] and not c["PACKED"]:
                continue
        elif c["BM"] > triton.next_power_of_2(max(m, 1)) or c["BN"] > max(
            256, triton.next_power_of_2(n)
        ):
            continue
        # Wide multi-row vectorization can truncate BF16 stores on P800.
        # Keep the validated narrow multi-row and all single-row widths.
        if named_args.get("OUT_BF16", False) and c["BM"] > 1 and c["BN"] > 512:
            continue
        if c["PACKED"] and not named_args.get("PACKED_OK", False):
            continue
        if (
            named_args["INTERLEAVED"]
            and c["BM"] == 1
            and c["BN"] > 256
            and not c["isCloseVectorization"]
            and not c["PACKED"]
        ):
            continue
        result.append(candidate)
    return result


def _bench_wall(fn, synchronize, quantiles, *, deadline=None):
    """Bounded synchronized wall time; an in-flight operation is not preempted."""
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    start = time.perf_counter()
    fn()
    synchronize()
    pilot = max(time.perf_counter() - start, 1e-6)
    calls = max(1, min(20, int(0.002 / pilot)))
    samples = []
    for _ in range(7):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        start, completed = time.perf_counter(), 0
        for _ in range(calls):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            fn()
            completed += 1
        synchronize()
        if completed:
            samples.append((time.perf_counter() - start) * 1000 / completed)
    if not samples:
        return None
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


@triton.jit
def _routed_xpu(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    INDEX64: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PACKED: tl.constexpr,
    MATH: tl.constexpr,
    CACHE: tl.constexpr,
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    tiles: tl.constexpr = tl.cdiv(N, BN)
    # Keep a row scalar: the current XPU backend mishandles multi-row weight
    # broadcast and non-aligned row tails in a flattened vector tile.
    cols = ((pid % tiles) * BN + tl.arange(0, BN))[None, :]
    for rr in range(BM):
        row = (pid // tiles) * BM + rr
        # With BM=1, the exact grid proves every row valid; column tails remain.
        if (PACKED and BM == 1) or row < M:
            mask = (cols < N) if PACKED and BM == 1 else (row < M) & (cols < N)
            if PACKED:
                tl.static_assert(
                    X.dtype.element_ty == tl.float16,
                    "Packed input is supported only for FP16",
                )
                words = tl.load(
                    X.to(tl.pointer_type(tl.uint32)) + row * (S // 2) + cols,
                    mask,
                    other=0,
                )
                gate = (
                    (words & 0xFFFF)
                    .to(tl.uint16)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                up = (
                    (words >> 16)
                    .to(tl.uint16)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
            else:
                if INTERLEAVED:
                    offsets = row * S + 2 * cols
                    up_offsets = offsets + 1
                else:
                    offsets = row * S + cols
                    up_offsets = offsets + N
                gate = tl.load(X + offsets, mask, other=0, cache_modifier=CACHE).to(
                    tl.float32
                )
                up = tl.load(X + up_offsets, mask, other=0, cache_modifier=CACHE).to(
                    tl.float32
                )
            if MATH == 3:
                sigmoid = 1.0 / (1.0 + libdevice.fast_expf(-gate))
            elif MATH == 1:
                sigmoid = 1.0 / (1.0 + tl.exp(-gate))
            else:
                sigmoid = tl.sigmoid(gate)
            result = (gate * sigmoid) * up
            if WEIGHT:
                if PACKED and BM == 1:
                    weight = tl.load(W + row).to(tl.float32)
                else:
                    weight = tl.load(W + row, row < M, other=0).to(tl.float32)
                result = result * weight
            tl.store(Y + row * N + cols, result, mask)


@triton.jit
def _routed_flat(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BN: tl.constexpr,
    PACKED: tl.constexpr,
    MATH: tl.constexpr,
    CACHE: tl.constexpr,
    INDEX64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    idx = pid * BN + tl.arange(0, BN)
    valid = idx < M * N
    row = idx // N
    col = idx % N
    if PACKED:
        tl.static_assert(
            X.dtype.element_ty == tl.float16,
            "Packed input is supported only for FP16",
        )
        if S == 2 * N:
            wordidx = idx
        else:
            wordidx = row * (S // 2) + col
        words = tl.load(
            X.to(tl.pointer_type(tl.uint32)) + wordidx,
            valid,
            other=0,
            cache_modifier=CACHE,
        )
        gate = (words & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        up = (words >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        if INTERLEAVED:
            gidx = row * S + 2 * col
            uidx = gidx + 1
        else:
            gidx = row * S + col
            uidx = gidx + N
        gate = tl.load(X + gidx, valid, other=0, cache_modifier=CACHE).to(tl.float32)
        up = tl.load(X + uidx, valid, other=0, cache_modifier=CACHE).to(tl.float32)
    if MATH == 3:
        sigmoid = 1.0 / (1.0 + libdevice.fast_expf(-gate))
    elif MATH == 1:
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    else:
        sigmoid = tl.sigmoid(gate)
    result = gate * sigmoid * up
    if WEIGHT:
        result = result * tl.load(W + row, valid, other=0).to(tl.float32)
    tl.store(Y + idx, result, valid)


@triton.jit
def _routed_tunable(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    INDEX64: tl.constexpr,
    PACKED_OK: tl.constexpr,
    OUT_BF16: tl.constexpr,
    DEVICE: tl.constexpr,
    ARCH: tl.constexpr,
    X_ALIGNMENT: tl.constexpr,
    W_ALIGNMENT: tl.constexpr,
    Y_ALIGNMENT: tl.constexpr,
    COMPILER_TUNING: tl.constexpr,
    FLAT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PACKED: tl.constexpr,
    MATH: tl.constexpr,
    CACHE: tl.constexpr,
):
    # One native tuner compares both paths without changing their arithmetic.
    if FLAT:
        _routed_flat(
            X, W, Y, M, N, S, WEIGHT, INTERLEAVED, BN, PACKED, MATH, CACHE, INDEX64
        )
    else:
        _routed_xpu(
            X, W, Y, M, N, S, WEIGHT, INTERLEAVED, INDEX64, BM, BN, PACKED, MATH, CACHE
        )


def _checked_xpu_backend():
    from triton.compiler.compiler import make_backend

    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    if (
        type(backend).__module__ != "triton.backends.xpu.compiler"
        or backend.binary_ext != "xpubin"
    ):
        raise RuntimeError("This kernel requires the Kunlunxin XPU compiler")
    return target, backend


def _native_config(options):
    bm, bn, flat = options["BM"], options["BN"], options["FLAT"]

    def verify_metadata(meta):
        # AABS must not change a tile after its concrete grid was computed.
        # Capture integers only; never capture input/output tensors in a config.
        if (meta["BM"], meta["BN"], meta["FLAT"]) != (bm, bn, flat):
            raise RuntimeError("XPU backend changed an explicitly pruned tile")
        count = (
            triton.cdiv(meta["M"] * meta["N"], bn)
            if flat
            else triton.cdiv(meta["M"], bm) * triton.cdiv(meta["N"], bn)
        )
        if meta.get("grid") != (count, 1, 1):
            raise RuntimeError("XPU launch grid no longer matches the tuned tile")

    return triton.Config(
        options,
        num_warps=None,
        num_ctas=None,
        num_stages=None,
        pre_hook=verify_metadata,
    )


def _prune_native(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    interface = getattr(torch, args["X"].device.type)
    if getattr(interface, "is_current_stream_capturing", lambda: False)():
        raise RuntimeError("Warm up this autotune signature before graph capture")
    _, backend = _checked_xpu_backend()
    base = _compiler_profiles()[0]
    result, checked_profiles = [], []
    for candidate in _prune(configs, args):
        meta = candidate.kwargs
        # Only packed words provide reliable flat interleaved vector loads.
        # Choose the protected scalar-row vectorization mode for this layout.
        close_vectorization = bool(
            args["INTERLEAVED"]
            and not meta["FLAT"]
            and meta["BM"] == 1
            and meta["BN"] > 256
            and not meta["PACKED"]
        )
        if meta["isCloseVectorization"] != close_vectorization:
            continue
        if meta["buffer_size_limit"] < triton.cdiv(meta["BN"], 64) * 4:
            continue
        if not args["COMPILER_TUNING"] and any(meta[k] != v for k, v in base.items()):
            continue
        options = {k: meta[k] for k in (*base, "isCloseVectorization")}
        profile = tuple(sorted(options.items()))
        if profile not in checked_profiles:
            parsed = backend.parse_options(options)
            if any(
                not hasattr(parsed, k) or getattr(parsed, k) != v
                for k, v in options.items()
            ):
                raise RuntimeError(
                    "XPU backend ignored/changed a compiler tuning option"
                )
            checked_profiles.append(profile)
        count = (
            triton.cdiv(args["M"] * args["N"], meta["BN"])
            if meta["FLAT"]
            else triton.cdiv(args["M"], meta["BM"]) * triton.cdiv(args["N"], meta["BN"])
        )
        # The XPU JIT consumes grid before evaluating callable launch grids.
        # Fresh Configs keep cached grids independent across shape changes.
        result.append(_native_config({**meta, "grid": (count, 1, 1)}))
    if not result:
        raise RuntimeError("No supported XPU configurations after shape pruning")
    return result


def _bench_native(fn, quantiles=(0.5, 0.2, 0.8)):
    # Match RMSNorm's host-clock benchmark, without CUDA events/graphs. Never
    # swallow launch/runtime errors; native Triton handles resource failures.
    device = triton.runtime.driver.active.get_active_torch_device()
    interface = getattr(torch, device.type)
    fn()  # First compilation is outside timing.
    interface.synchronize()
    return _bench_wall(fn, interface.synchronize, quantiles)


def _make_native_tuner():
    return triton.autotune(
        configs=_configs(),
        key=[
            "M",
            "N",
            "S",
            "WEIGHT",
            "INTERLEAVED",
            "INDEX64",
            "PACKED_OK",
            "OUT_BF16",
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


# Native Triton configuration cache only; no manual global containers or
# output/result caching. Tensor dtypes are also included by native Autotuner.
_native_tuner = _make_native_tuner()


def _launch_fixed(x, weights, out, config, interleaved=True, **backend_options):
    """Fixed launch; native defaults use Triton's own positional launch path."""
    if config.get("PACKED", 0) and x.dtype != torch.float16:
        raise ValueError("Packed input is supported only for FP16")
    # Keep nondefault profiles on the established keyword path. The native
    # XPU launcher only caches positional calls; no custom state is retained.
    if (
        not backend_options
        and not config.get("FLAT", 0)
        and config.get("isCloseVectorization", False) == XPUOptions.isCloseVectorization
        and config.keys()
        <= {
            "BM",
            "BN",
            "PACKED",
            "MATH",
            "CACHE",
            "isCloseVectorization",
            "num_warps",
            "num_ctas",
            "num_stages",
        }
        and config.get("num_warps", 4) == 4
        and config.get("num_ctas", 1) == 1
        and config.get("num_stages", 1) == 1
        and XPUOptions.num_stages == 1
        and XPUOptions.num_warps in (-1, 4)
        and XPUOptions.num_ctas in (-1, 1)
    ):
        m, width = x.shape
        n, stride = width // 2, x.stride(0)
        bm, bn = config["BM"], config["BN"]
        index64 = max((m - 1) * stride + width, m * n) >= 2**31
        # num_warps/num_ctas are CUDA compatibility metadata on XPU. Device
        # tests confirmed their omission leaves the xpubin bytes unchanged.
        return _routed_xpu[(triton.cdiv(m, bm) * triton.cdiv(n, bn), 1, 1)](
            x,
            weights,
            out,
            m,
            n,
            stride,
            weights is not None,
            interleaved,
            index64,
            bm,
            bn,
            config["PACKED"],
            config["MATH"],
            config["CACHE"],
        )
    m, width = x.shape
    n = width // 2
    args = dict(config)
    bm, bn = args["BM"], args["BN"]
    flat = args.pop("FLAT", 0)
    index64 = max((m - 1) * x.stride(0) + width, m * n) >= 2**31
    if flat:
        if out.dtype == torch.bfloat16:
            raise ValueError("Flat BF16 output is disabled: vendor RTNE is incorrect")
        args.pop("BM")
        return _routed_flat[(triton.cdiv(m * n, bn),)](
            x,
            weights,
            out,
            m,
            n,
            x.stride(0),
            weights is not None,
            interleaved,
            INDEX64=index64,
            **args,
            **backend_options,
        )
    return _routed_xpu[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        x,
        weights,
        out,
        m,
        n,
        x.stride(0),
        weights is not None,
        interleaved,
        index64,
        **args,
        **backend_options,
    )


def _select_config(
    m,
    n,
    stride,
    input_dtype,
    weight_dtype,
    output_dtype,
    interleaved,
    x_alignment,
    weight_alignment,
    output_alignment,
):
    """Return a fresh config using metadata only; retain no cross-call state."""
    fp16, bf16, fp32 = torch.float16, torch.bfloat16, torch.float32
    # Exact metadata from the 32-case P800 validation. All measured weights
    # were FP32 and aligned; all outputs were aligned. Keys include input
    # row padding and byte alignment, including the tested offset-one views.
    # Each row is split/no-weight, split/weight, interleaved/no-weight, /weight;
    # entries hold BM, BN, PACKED, and the vendor vectorization-off option.
    measured = {
        (17, 513, bf16, bf16, 1043, 2): (
            (1, 1024, 0, True),
            (4, 1024, 0, False),
            (1, 512, 0, True),
            (1, 256, 0, False),
        ),
        (17, 513, fp16, fp16, 1043, 2): (
            (4, 1024, 0, False),
            (1, 1024, 0, False),
            (4, 1024, 0, False),
            (1, 256, 0, False),
        ),
        (17, 513, fp32, fp32, 1043, 4): (
            (1, 1024, 0, False),
            (4, 1024, 0, False),
            (1, 1024, 0, True),
            (4, 1024, 0, False),
        ),
        (129, 384, fp32, bf16, 768, 0): (
            (1, 512, 0, True),
            (16, 512, 0, False),
            (16, 512, 0, False),
            (1, 512, 0, True),
        ),
        (129, 769, bf16, fp32, 1555, 2): (
            (1, 1024, 0, False),
            (1, 1024, 0, True),
            (1, 1024, 0, True),
            (1, 1024, 0, True),
        ),
        (256, 1024, bf16, bf16, 2048, 0): (
            (1, 1024, 0, True),
            (1, 1024, 0, False),
            (1, 1024, 0, True),
            (1, 1024, 0, True),
        ),
        (1536, 3072, fp16, fp16, 6144, 0): (
            (1, 4096, 0, False),
            (1, 4096, 0, False),
            (1, 4096, 1, False),
            (1, 4096, 1, False),
        ),
        (8192, 4096, bf16, bf16, 8192, 0): (
            (1, 4096, 0, False),
            (1, 4096, 0, False),
            (1, 4096, 0, True),
            (1, 4096, 0, True),
        ),
    }
    safe = (1, 128, 0, False)
    options = safe
    choices = None
    if output_alignment == 0 and weight_alignment == 0 and weight_dtype in (None, fp32):
        choices = measured.get((m, n, input_dtype, output_dtype, stride, x_alignment))
        if choices is not None:
            options = choices[2 * int(interleaved) + int(weight_dtype is not None)]
    bm, bn, packed, vectorization_off = options
    packed_ok = (
        interleaved and input_dtype == fp16 and x_alignment % 4 == 0 and stride % 2 == 0
    )
    if (packed and not packed_ok) or (
        interleaved and bm == 1 and bn > 256 and not vectorization_off and not packed
    ):
        bm, bn, packed, vectorization_off = safe
    config = {
        "BM": bm,
        "BN": bn,
        "PACKED": packed,
        "MATH": 0,
        "CACHE": "",
        "isCloseVectorization": vectorization_off,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    }
    if (
        choices is not None
        and (not interleaved or packed_ok)
        and (m, n, input_dtype, output_dtype) == (1536, 3072, fp16, fp16)
    ):
        # Keep the historical flat fallback within the same 4096 tile cap.
        config.update(
            BM=1,
            BN=4096,
            FLAT=1,
            PACKED=int(interleaved),
            isCloseVectorization=False,
            budget_tiling=False,
            unroll_num=4,
        )
    return config


def _autotune(x, weights, out, interleaved=True, *, max_configs=64, max_seconds=15.0):
    """Offline, call-local tuning; normal public calls never invoke this helper.

    At most 64 candidate entries and 15 seconds by default. Either limit may
    be explicitly disabled with None. The time limit includes setup, but is
    soft: compilation and already submitted device work cannot be preempted.
    A protected reference and the first measured-dispatch seed are always
    launched, even with an exhausted time budget. The reference is outside
    the candidate count. The chosen configuration recomputes out before
    return, even after timeout.
    Compiler options must be recognized with the requested values. Same-math
    candidates require bitwise finite results; alternatives permit one ULP.
    Run serially outside graph capture. No state survives this invocation.
    """
    import math

    from triton.compiler.compiler import make_backend
    from triton.compiler.errors import CompilationError

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
        raise ValueError("max_seconds must be finite and positive or None")
    started = time.perf_counter()
    deadline = None if max_seconds is None else started + max_seconds
    m, width = x.shape
    n = width // 2
    if not m or not n:
        raise ValueError("Autotuning requires nonempty tensors")
    interface = getattr(torch, x.device.type)
    with interface.device(x.device):
        if getattr(interface, "is_current_stream_capturing", lambda: False)():
            raise RuntimeError("Autotuning is unavailable during graph capture")
        backend = make_backend(triton.runtime.driver.active.get_current_target())
        if (
            type(backend).__module__ != "triton.backends.xpu.compiler"
            or backend.binary_ext != "xpubin"
        ):
            raise RuntimeError("This tuner requires the Kunlunxin XPU compiler")
        candidates = _prune(
            _configs(),
            {
                "M": m,
                "N": n,
                "INTERLEAVED": interleaved,
                "OUT_BF16": out.dtype == torch.bfloat16,
                "PACKED_OK": interleaved
                and x.dtype == torch.float16
                and x.data_ptr() % 4 == 0
                and x.stride(0) % 2 == 0,
            },
        )
        reference = torch.empty_like(out)
        safe = {
            "BM": 1,
            "BN": 128,
            "PACKED": 0,
            "MATH": 0,
            "CACHE": "",
            "isCloseVectorization": False,
            "num_warps": 4,
            "num_stages": 1,
        }
        _launch_fixed(x, weights, reference, safe, interleaved)
        interface.synchronize()
        reference_cpu = reference.cpu()
        seed = _select_config(
            m,
            n,
            x.stride(0),
            x.dtype,
            weights.dtype if weights is not None else None,
            out.dtype,
            interleaved,
            x.data_ptr() % 16,
            weights.data_ptr() % 16 if weights is not None else 0,
            out.data_ptr() % 16,
        )
        options_stream = [seed] + [candidate.all_kwargs() for candidate in candidates]
        best, best_time, records, seen = None, float("inf"), [], set()
        for index, options in enumerate(options_stream):
            if max_configs is not None and index >= max_configs:
                break
            if index and deadline is not None and time.perf_counter() >= deadline:
                break
            key = tuple(sorted(options.items()))
            if key in seen:
                records.append({"config": dict(options), "status": "duplicate"})
                continue
            seen.add(key)
            compiler = {
                k: v
                for k, v in options.items()
                if k
                not in (
                    "BM",
                    "BN",
                    "FLAT",
                    "PACKED",
                    "MATH",
                    "CACHE",
                    "num_warps",
                    "num_ctas",
                    "num_stages",
                )
            }
            parsed = backend.parse_options(dict(compiler))
            if any(
                not hasattr(parsed, k) or getattr(parsed, k) != v
                for k, v in compiler.items()
            ):
                records.append(
                    {
                        "config": dict(options),
                        "status": "rejected",
                        "error": "backend ignored or changed "
                        "requested compiler profile",
                    }
                )
                continue

            def run():
                return _launch_fixed(x, weights, out, options, interleaved)

            try:
                out.fill_(float("nan"))
                compiled = run()
                interface.synchronize()
                actual = out.cpu()
                for classify in (torch.isnan, torch.isposinf, torch.isneginf):
                    assert torch.equal(classify(actual), classify(reference_cpu))
                finite = torch.isfinite(reference_cpu)
                bits_type = torch.int32 if out.dtype == torch.float32 else torch.int16
                actual_bits = actual[finite].view(bits_type).to(torch.int64)
                reference_bits = reference_cpu[finite].view(bits_type).to(torch.int64)
                max_ulp = (
                    int((actual_bits - reference_bits).abs().max())
                    if actual_bits.numel()
                    else 0
                )
                assert max_ulp <= (
                    0 if options["MATH"] == 0 else 1
                ), f"max_ulp={max_ulp}"
            except (OutOfResources, CompilationError, AssertionError) as error:
                records.append(
                    {"config": dict(options), "status": "rejected", "error": str(error)}
                )
                continue
            # A validated seed remains available even if compilation used the budget.
            if best is None:
                best = dict(options)
            times = _bench_wall(
                run, interface.synchronize, (0.5, 0.2, 0.8), deadline=deadline
            )
            record = {
                "config": dict(options),
                "status": "ok" if times else "validated_unmeasured",
                "times_ms": times,
                "max_ulp": max_ulp,
                "actual_options": {
                    k: getattr(compiled.metadata, k, None) for k in compiler
                },
            }
            records.append(record)
            if times is not None and times[0] < best_time:
                best_time, best = times[0], dict(options)
        if best is None:
            raise RuntimeError("No supported Kunlunxin configuration passed validation")
        _launch_fixed(x, weights, out, best, interleaved)
        interface.synchronize()
    return best, records


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
    """Compute (gate * sigmoid(gate)) * up * optional per-row weight.

    FP16/BF16/FP32 input is [M, 2*N] with contiguous columns. Rows may have
    padding. Interleaved rows contain [g0,u0,g1,u1,...]; otherwise they contain
    [g0,...,gN-1,u0,...,uN-1]. Output is contiguous [M,N].

    The first call per metadata/dtype/device signature tunes at most 63
    shape-pruned candidates; later calls reuse only the selected configuration.
    compiler_tuning=False searches geometry with the base compiler profile.
    autotune=False uses the fixed metadata selector. Both switches default to
    True. Warm up each signature before capturing a graph.
    """
    if not isinstance(autotune, bool) or not isinstance(compiler_tuning, bool):
        raise TypeError("autotune and compiler_tuning must be bool")
    x = gateup_output
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("gateup_output must be a two-dimensional tensor")
    if x.device.type not in ("cuda", "xpu") or x.dtype not in supported:
        raise ValueError("gateup_output must be Kunlunxin FP16, BF16, or FP32")
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
            storage = out.untyped_storage().data_ptr()
            if storage == x.untyped_storage().data_ptr() or (
                topk_weights is not None
                and storage == topk_weights.untyped_storage().data_ptr()
            ):
                raise ValueError("out must not share storage with input or weights")
        interface = getattr(torch, x.device.type)
        with interface.device(x.device):
            if autotune:
                target, _ = _checked_xpu_backend()
                packed_ok = (
                    use_interleaved
                    and x.dtype == torch.float16
                    and x.data_ptr() % 4 == 0
                    and x.stride(0) % 2 == 0
                )
                index64 = max((m - 1) * x.stride(0) + width, m * n) >= 2**31
                _native_tuner.run(
                    x,
                    topk_weights,
                    out,
                    m,
                    n,
                    x.stride(0),
                    topk_weights is not None,
                    use_interleaved,
                    index64,
                    packed_ok,
                    out.dtype == torch.bfloat16,
                    x.device.index,
                    str(target.arch),
                    x.data_ptr() % 16,
                    topk_weights.data_ptr() % 16 if topk_weights is not None else 0,
                    out.data_ptr() % 16,
                    compiler_tuning,
                    warmup=False,
                )
            else:
                config = _select_config(
                    m,
                    n,
                    x.stride(0),
                    x.dtype,
                    topk_weights.dtype if topk_weights is not None else None,
                    out.dtype,
                    use_interleaved,
                    x.data_ptr() % 16,
                    topk_weights.data_ptr() % 16 if topk_weights is not None else 0,
                    out.data_ptr() % 16,
                )
                _launch_fixed(x, topk_weights, out, config, use_interleaved)

    return out


silu_and_mul = silu_and_mul_routed
