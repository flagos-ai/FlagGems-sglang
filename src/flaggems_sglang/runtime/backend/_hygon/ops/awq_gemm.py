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

"""Hygon AWQ GEMM with explicit shape specializations and wave64 MMAC.

Paths include group-local GEMV, transposed FP16x2 group matmul, and prepared
FP16 activation planes. General strided inputs retain the native tuner.
AWQ shifts are [0,16,4,20,8,24,12,28]; stored zero points have no +1 offset.
Only tuning configurations are cached; output and scratch are freshly allocated.
"""
import torch
import triton
import triton.language as tl


def _ceil_div(x, y):
    return (x + y - 1) // y


def _power_of_two(x):
    return 1 << (x - 1).bit_length()


@triton.jit
def _unpack_constant(p):
    """Decode AWQ order [0,4,1,5,2,6,3,7] using immediate shifts."""
    x0 = p & 15
    x1 = (p >> 16) & 15
    x2 = (p >> 4) & 15
    x3 = (p >> 20) & 15
    x4 = (p >> 8) & 15
    x5 = (p >> 24) & 15
    x6 = (p >> 12) & 15
    x7 = (p >> 28) & 15
    e0 = tl.interleave(x0, x4)
    e1 = tl.interleave(x2, x6)
    o0 = tl.interleave(x1, x5)
    o1 = tl.interleave(x3, x7)
    return tl.interleave(tl.interleave(e0, e1), tl.interleave(o0, o1))


@triton.jit
def _awq_sum(
    P,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, RS)
    v = tl.load(
        P + s[:, None] * (M * N) + x[None, :],
        (s[:, None] < SPLIT) & (x[None, :] < M * N),
        0,
    )
    tl.store(OUT + x, tl.sum(v, 0), x < M * N)


@triton.jit
def _awq_zero(OUT, SIZE: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(OUT + x, 0.0, x < SIZE)


@triton.jit
def _awq_grouped(
    A,
    Q,
    S,
    Z,
    P,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    VECTOR: tl.constexpr,
    CONSTANT_UNPACK: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    if VECTOR:
        acc = tl.zeros((BK, BN), tl.float32)
    elif TRANSPOSE:
        acc = tl.zeros((BN, BM), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    gb = part * (K // G) // SPLIT
    ge = (part + 1) * (K // G) // SPLIT
    for group in range(gb, ge):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        if CONSTANT_UNPACK:
            zero = _unpack_constant(z).to(tl.float32)
        else:
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        if MODE == "fp16x2" and not VECTOR:
            if TRANSPOSE:
                group_acc = tl.zeros((BN, BM), tl.float32)
            else:
                group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local_k = block * BK + tl.arange(0, BK)
            rk = group * G + local_k
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local_k[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            if CONSTANT_UNPACK:
                weight = _unpack_constant(q).to(tl.float32)
            else:
                q = tl.interleave(q, q)
                q = tl.interleave(q, q)
                q = tl.interleave(q, q)
                weight = ((q >> shift[None, :]) & 15).to(tl.float32)
            delta = weight - zero[None, :]
            if MODE == "fp16x2" and not VECTOR:
                b = tl.where(local_k[:, None] < G, delta, 0.0)
            else:
                b = tl.where(local_k[:, None] < G, delta * scale[None, :], 0.0)
            if VECTOR:
                a = tl.load(
                    A + (tile // tl.cdiv(N, BN)) * AM + rk * AK, local_k < G, 0
                ).to(tl.float32)
                acc += a[:, None] * b
            else:
                a = tl.load(
                    A + rm[:, None] * AM + rk[None, :] * AK,
                    (rm[:, None] < M) & (local_k[None, :] < G),
                    0,
                ).to(tl.float32)
                if MODE == "fp16x2":
                    ah = a.to(tl.float16)
                    al = (a - ah.to(tl.float32)).to(tl.float16)
                    bh = b.to(tl.float16)
                    if TRANSPOSE:
                        group_acc = tl.dot(
                            tl.trans(bh), tl.trans(al), group_acc
                        )
                        group_acc = tl.dot(
                            tl.trans(bh), tl.trans(ah), group_acc
                        )
                    else:
                        group_acc = tl.dot(al, bh, group_acc)
                        group_acc = tl.dot(ah, bh, group_acc)
                elif MODE == "bf16x3":
                    ah = a.to(tl.bfloat16)
                    bh = b.to(tl.bfloat16)
                    al = (a - ah.to(tl.float32)).to(tl.bfloat16)
                    bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
                    if TRANSPOSE:
                        acc = tl.dot(tl.trans(bl), tl.trans(ah), acc)
                        acc = tl.dot(tl.trans(bh), tl.trans(al), acc)
                        acc = tl.dot(tl.trans(bh), tl.trans(ah), acc)
                    else:
                        acc = tl.dot(ah, bl, acc)
                        acc = tl.dot(al, bh, acc)
                        acc = tl.dot(ah, bh, acc)
                else:
                    if TRANSPOSE:
                        acc = tl.dot(
                            tl.trans(b), tl.trans(a), acc, input_precision=MODE
                        )
                    else:
                        acc = tl.dot(a, b, acc, input_precision=MODE)
        if MODE == "fp16x2" and not VECTOR:
            if TRANSPOSE:
                acc += group_acc * scale[:, None]
            else:
                acc += group_acc * scale[None, :]
    if VECTOR:
        value = tl.sum(acc, 0)
        off = (tile // tl.cdiv(N, BN)) * N + rn
        mask = rn < N
    else:
        if TRANSPOSE:
            value = tl.trans(acc)
        else:
            value = acc
        off = rm[:, None] * N + rn[None, :]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        tl.store(OUT + off, value, mask)
    else:
        tl.store(P + part * M * N + off, value, mask)


@triton.jit
def _awq_control_gemv(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    row = tile // tl.cdiv(N, BN)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
    rk = part * BK + tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BK, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(A + row * AM + rk * AK, rk < K, 0).to(tl.float32)
        packed = tl.load(
            Q + rk[:, None] * QK + rp[None, :] * QN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        group = rk // G
        zero = tl.load(
            Z + group[:, None] * ZG + rp[None, :] * ZN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        zero = tl.interleave(zero, zero)
        zero = tl.interleave(zero, zero)
        zero = tl.interleave(zero, zero)
        scale = tl.load(
            S + group[:, None] * SG + rn[None, :] * SN,
            (rk[:, None] < K) & (rn[None, :] < N),
            0,
        ).to(tl.float32)
        weight = ((packed >> shift[None, :]) & 15).to(tl.float32)
        zero = ((zero >> shift[None, :]) & 15).to(tl.float32)
        b = (weight - zero) * scale
        acc += a[:, None] * b
        rk += BK * SPLIT
    result = tl.sum(acc, axis=0)
    tl.store(P + part * M * N + row * N + rn, result, rn < N)


@triton.jit
def _awq_control_matmul(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
    rk = part * BK + tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(
            A + rm[:, None] * AM + rk[None, :] * AK,
            (rm[:, None] < M) & (rk[None, :] < K),
            0,
        ).to(tl.float32)
        packed = tl.load(
            Q + rk[:, None] * QK + rp[None, :] * QN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        if G % BK == 0:
            group = (part * BK + _ * BK * SPLIT) // G
            zero = tl.load(
                Z + group * ZG + rp * ZN, (group < K // G) & (rp < N // 8), 0
            )
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            scale = tl.load(
                S + group * SG + rn * SN, (group < K // G) & (rn < N), 0
            ).to(tl.float32)
            weight = ((packed >> shift[None, :]) & 15).to(tl.float32)
            zero = ((zero >> shift) & 15).to(tl.float32)
            b = (weight - zero[None, :]) * scale[None, :]
        else:
            group = rk // G
            zero = tl.load(
                Z + group[:, None] * ZG + rp[None, :] * ZN,
                (rk[:, None] < K) & (rp[None, :] < N // 8),
                0,
            )
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            scale = tl.load(
                S + group[:, None] * SG + rn[None, :] * SN,
                (rk[:, None] < K) & (rn[None, :] < N),
                0,
            ).to(tl.float32)
            weight = ((packed >> shift[None, :]) & 15).to(tl.float32)
            zero = ((zero >> shift[None, :]) & 15).to(tl.float32)
            b = (weight - zero) * scale
        if PRECISION == "bf16x3":
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(ah, bl, acc)
            acc = tl.dot(al, bh, acc)
            acc = tl.dot(ah, bh, acc)
        else:
            acc = tl.dot(a, b, acc, input_precision=PRECISION)
        rk += BK * SPLIT
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _awq_execute(
    A,
    Q,
    S,
    Z,
    P,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    REQUESTED: tl.constexpr,
    DEVICE: tl.constexpr,
    FAMILY: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    CONSTANT_UNPACK: tl.constexpr,
):
    if FAMILY == 0:
        _awq_grouped(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
            M,
            N,
            K,
            G,
            AM,
            AK,
            QK,
            QN,
            SG,
            SN,
            ZG,
            ZN,
            MODE,
            BM,
            BN,
            BK,
            SPLIT,
            TRANSPOSE,
            False,
            CONSTANT_UNPACK,
        )
    else:
        if SPLIT == 1:
            dest = OUT
        else:
            dest = P
        if FAMILY == 1:
            _awq_control_matmul(
                A,
                Q,
                S,
                Z,
                dest,
                M,
                N,
                K,
                G,
                AM,
                AK,
                QK,
                QN,
                SG,
                SN,
                ZG,
                ZN,
                SPLIT,
                BM,
                BN,
                BK,
                MODE,
            )
        else:
            _awq_control_gemv(
                A,
                Q,
                S,
                Z,
                dest,
                M,
                N,
                K,
                G,
                AM,
                AK,
                QK,
                QN,
                SG,
                SN,
                ZG,
                ZN,
                SPLIT,
                BN,
                BK,
            )


def _finish(args, exception):
    if exception is None and args["SPLIT"] > 1:
        split = args["SPLIT"]
        _awq_sum[(_ceil_div(args["M"] * args["N"], 256),)](
            args["P"],
            args["OUT"],
            args["M"],
            args["N"],
            split,
            _power_of_two(split),
            256,
            num_warps=4,
            num_stages=1,
        )


def _bench(fn, quantiles):
    # Explicit HIP Event callback avoids the default testing/Graph benchmark.
    # Benchmark includes the Split-K sum through the native post-hook.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
    for begin, end in zip(starts, ends):
        begin.record()
        for _ in range(5):
            fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(
        begin.elapsed_time(end) / 5 for begin, end in zip(starts, ends)
    )
    if not all(0 < value < float("inf") for value in times):
        raise ValueError("Invalid Hygon autotune Event duration")
    values = []
    for q in quantiles:
        position = q * (len(times) - 1)
        lower = int(position)
        upper = min(lower + 1, len(times) - 1)
        values.append(
            times[lower] + (times[upper] - times[lower]) * (position - lower)
        )
    return values


def _config(
    family,
    mode,
    bm,
    bn,
    bk,
    split,
    warps=4,
    transpose=False,
    unpack=False,
    stages=1,
):
    return triton.Config(
        dict(
            FAMILY=family,
            MODE=mode,
            BM=bm,
            BN=bn,
            BK=bk,
            SPLIT=split,
            TRANSPOSE=transpose,
            CONSTANT_UNPACK=unpack,
        ),
        num_warps=warps,
        num_stages=stages,
    )


def _prune(configs, named_args, **kwargs):
    # Build fresh configurations for each key: FlagTree may adjust tile sizes.
    args = dict(named_args, **kwargs)
    m, n, k, g = (args[name] for name in ("M", "N", "K", "G"))
    groups = k // g
    requested = min(16, args["REQUESTED"])
    candidates = []
    if m == 1:
        for bn, bk, nw in ((32, 64, 4), (64, 64, 2), (64, 128, 4)):
            for split in (requested, min(16, requested * 2)):
                candidates.append(_config(2, "ieee", 1, bn, bk, split, nw))
    elif m <= 16:
        splits = (requested, max(2, requested // 2)) if requested > 1 else (1,)
        for split in splits:
            candidates.append(_config(1, "ieee", 16, 64, 32, split, 2))
            candidates.append(_config(1, "ieee", 16, 64, min(64, g), split, 4))
            for mode in ("ieee", "fp16x2"):
                for transpose in (False, True):
                    candidates.append(
                        _config(
                            0,
                            mode,
                            16,
                            64,
                            64 if g % 64 == 0 else 32,
                            min(split, groups),
                            2,
                            transpose,
                        )
                    )
    else:
        # Preserve requested Split-K; compare nearby splits and a single
        # partition only when enough output tiles exist for the current DCU.
        cores = torch.cuda.get_device_properties(
            args["DEVICE"]
        ).multi_processor_count
        tiles = _ceil_div(m, 64) * _ceil_div(n, 64)
        splits = [requested]
        if requested > 1:
            splits.append(max(2, requested // 2))
            if tiles >= cores:
                splits.append(1)
        for split in splits:
            candidates.append(_config(1, "ieee", 32, 64, 32, split, 2))
            candidates.append(_config(1, "ieee", 64, 64, 32, split, 4))
            candidates.append(
                _config(0, "ieee", 64, 64, 32, min(split, groups), 4)
            )
            candidates.append(
                _config(
                    0,
                    "fp16x2",
                    64,
                    64,
                    64 if g % 64 == 0 else 32,
                    min(split, groups),
                    4,
                    False,
                    True,
                )
            )
    valid, seen = [], set()
    for config in candidates:
        c = config.kwargs
        if c["BK"] & (c["BK"] - 1):
            continue
        signature = (tuple(c.items()), config.num_warps, config.num_stages)
        if signature not in seen:
            seen.add(signature)
            valid.append(config)
    return valid


_awq_tuned = triton.autotune(
    configs=[
        _config(1, "ieee", 16, 64, 32, 1, 2),
        _config(1, "ieee", 32, 64, 32, 1, 4),
    ],
    key=[
        "M",
        "N",
        "K",
        "G",
        "AM",
        "AK",
        "QK",
        "QN",
        "SG",
        "SN",
        "ZG",
        "ZN",
        "REQUESTED",
        "DEVICE",
    ],
    prune_configs_by={"early_config_prune": _prune},
    post_hook=_finish,
    do_bench=_bench,
    use_cuda_graph=False,
)(_awq_execute)


def _awq_general(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    assert split_k_iters > 0 and qweight.shape[0] == k
    assert scales.shape[1] == n and qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return output
    if k == 0:
        _awq_zero[(_ceil_div(m * n, 256),)](
            output, m * n, num_warps=4, num_stages=1
        )
        return output
    g = k // groups
    strides = (
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
    )
    # Small complete-K GEMV tiles cover rows independently and avoid a partial
    # allocation and reduction launch. Source values are always recomputed.
    if k <= 128 and m <= 32 and n <= 128:
        bn = 32 if n <= 64 else 64
        _awq_control_gemv[(m * _ceil_div(n, bn), 1)](
            input,
            qweight,
            scales,
            qzeros,
            output,
            m,
            n,
            k,
            g,
            *strides,
            1,
            bn,
            _power_of_two(k),
            num_warps=4,
            num_stages=1,
        )
        return output
    # Validated MMAC tile for the long-K medium-M region; metadata-only routing
    # avoids the online tuner/cache lookup on repeated complete API calls.
    if m >= 128 and k >= 512 and k <= 4096 and g % 64 == 0 and n >= 512:
        split = 1 if n >= 4096 else min(int(split_k_iters), 8)
        partial = (
            output
            if split == 1
            else torch.empty(
                (split, m, n), device=input.device, dtype=torch.float32
            )
        )
        _awq_grouped[(_ceil_div(m, 64) * _ceil_div(n, 64), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            output,
            m,
            n,
            k,
            g,
            *strides,
            "fp16x2",
            64,
            64,
            64,
            split,
            False,
            False,
            True,
            num_warps=4,
            num_stages=1,
        )
        if split > 1:
            _awq_sum[(_ceil_div(m * n, 256),)](
                partial,
                output,
                m,
                n,
                split,
                _power_of_two(split),
                256,
                num_warps=4,
                num_stages=1,
            )
        return output
    if m <= 64 or k <= 128:
        # Full-API measurements favor a short direct path in this region.
        # Preserve inter-CTA Split-K for long K; no online dispatch overhead.
        split = 1 if m == 1 and n >= 8192 and k <= 512 else int(split_k_iters)
        partial = (
            output
            if split == 1
            else torch.empty(
                (split, m, n), device=input.device, dtype=torch.float32
            )
        )
        if m == 1:
            if n >= 8192 and 128 < k <= 512:
                _awq_grouped[(_ceil_div(n, 64), 1)](
                    input,
                    qweight,
                    scales,
                    qzeros,
                    partial,
                    output,
                    m,
                    n,
                    k,
                    g,
                    *strides,
                    "ieee",
                    1,
                    64,
                    32,
                    1,
                    False,
                    True,
                    False,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                _awq_control_gemv[(_ceil_div(n, 32), split)](
                    input,
                    qweight,
                    scales,
                    qzeros,
                    partial,
                    m,
                    n,
                    k,
                    g,
                    *strides,
                    split,
                    32,
                    64,
                    num_warps=4,
                    num_stages=2,
                )
        else:
            bm = (
                16
                if m <= 16
                else (
                    64
                    if m >= 128 or (m >= 64 and k > 1024 and n >= 4096)
                    else 32
                )
            )
            _awq_control_matmul[(_ceil_div(m, bm) * _ceil_div(n, 64), split)](
                input,
                qweight,
                scales,
                qzeros,
                partial,
                m,
                n,
                k,
                g,
                *strides,
                split,
                bm,
                64,
                32,
                "ieee",
                num_warps=4,
                num_stages=2 if bm <= 32 else 1,
            )
        if split > 1:
            _awq_sum[(_ceil_div(m * n, 256),)](
                partial,
                output,
                m,
                n,
                split,
                _power_of_two(split),
                256,
                num_warps=4,
                num_stages=1,
            )
        return output
    meta = (m, n, k, g, *strides, int(split_k_iters), input.device.index)
    dtype = str(scales.dtype)
    key = meta + (
        str(input.dtype),
        str(qweight.dtype),
        dtype,
        str(qzeros.dtype),
        "torch.float32",
        dtype,
    )
    config = _awq_tuned.cache.get(key)
    if config is None:
        partial = torch.empty(
            (16, m, n), device=input.device, dtype=torch.float32
        )
        _awq_tuned[
            lambda c: (
                _ceil_div(m, c["BM"]) * _ceil_div(n, c["BN"]),
                c["SPLIT"],
            )
        ](input, qweight, scales, qzeros, partial, output, *meta)
        config = _awq_tuned.best_config
    else:
        _awq_tuned.best_config = config
        c = config.kwargs
        partial = (
            output
            if c["SPLIT"] == 1
            else torch.empty(
                (c["SPLIT"], m, n), device=input.device, dtype=torch.float32
            )
        )
        grid = (_ceil_div(m, c["BM"]) * _ceil_div(n, c["BN"]), c["SPLIT"])
        if c["FAMILY"] == 0:
            _awq_grouped[grid](
                input,
                qweight,
                scales,
                qzeros,
                partial,
                output,
                m,
                n,
                k,
                g,
                *strides,
                c["MODE"],
                c["BM"],
                c["BN"],
                c["BK"],
                c["SPLIT"],
                c["TRANSPOSE"],
                False,
                c["CONSTANT_UNPACK"],
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        elif c["FAMILY"] == 1:
            _awq_control_matmul[grid](
                input,
                qweight,
                scales,
                qzeros,
                partial,
                m,
                n,
                k,
                g,
                *strides,
                c["SPLIT"],
                c["BM"],
                c["BN"],
                c["BK"],
                c["MODE"],
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        else:
            _awq_control_gemv[grid](
                input,
                qweight,
                scales,
                qzeros,
                partial,
                m,
                n,
                k,
                g,
                *strides,
                c["SPLIT"],
                c["BN"],
                c["BK"],
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
    split = config.kwargs["SPLIT"]
    if split > 1:
        _awq_sum[(_ceil_div(m * n, 256),)](
            partial,
            output,
            m,
            n,
            split,
            _power_of_two(split),
            256,
            num_warps=4,
            num_stages=1,
        )
    return output


@triton.jit
def _contiguous_gemv(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    _awq_control_gemv(
        A,
        Q,
        S,
        Z,
        P,
        M,
        N,
        K,
        G,
        K,
        1,
        N // 8,
        1,
        N,
        1,
        N // 8,
        1,
        SPLIT,
        BN,
        BK,
    )


@triton.jit
def _contiguous_matmul(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
):
    _awq_control_matmul(
        A,
        Q,
        S,
        Z,
        P,
        M,
        N,
        K,
        G,
        K,
        1,
        N // 8,
        1,
        N,
        1,
        N // 8,
        1,
        SPLIT,
        BM,
        64,
        32,
        "ieee",
    )


@triton.jit
def _contiguous_grouped(
    A,
    Q,
    S,
    Z,
    P,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
):
    _awq_grouped(
        A,
        Q,
        S,
        Z,
        P,
        OUT,
        M,
        N,
        K,
        G,
        K,
        1,
        N // 8,
        1,
        N,
        1,
        N // 8,
        1,
        "fp16x2",
        64,
        64,
        64,
        SPLIT,
        False,
        False,
        True,
    )


def _r4_dispatch(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    if not (
        input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
        and input.dtype == torch.float32
        and scales.dtype == torch.float32
    ):
        return _awq_general(input, qweight, scales, qzeros, split_k_iters)
    assert split_k_iters > 0 and qweight.shape[0] == k
    assert scales.shape[1] == n and qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    if not m or not n:
        return torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if k == 0:
        output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
        _awq_zero[(_ceil_div(m * n, 256),)](
            output, m * n, num_warps=4, num_stages=1
        )
        return output
    g = k // groups
    medium_grouped = m >= 128 and 512 <= k <= 4096 and g % 64 == 0 and n >= 512
    if not (m <= 64 or k <= 128 or medium_grouped) or (
        m == 1 and n >= 8192 and k <= 512
    ):
        return _awq_general(input, qweight, scales, qzeros, split_k_iters)
    tiny = k <= 128 and m <= 32 and n <= 128
    split = 1 if tiny else int(split_k_iters)
    if medium_grouped:
        split = 1 if n >= 4096 else min(split, 8)
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    partial = (
        output
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    if tiny or m == 1:
        bn = (32 if n <= 64 else 64) if tiny else 32
        bk = _power_of_two(k) if tiny else 64
        _contiguous_gemv[(m * _ceil_div(n, bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            g,
            split,
            bn,
            bk,
            num_warps=4,
            num_stages=1 if tiny else 2,
        )
    elif medium_grouped:
        _contiguous_grouped[(_ceil_div(m, 64) * _ceil_div(n, 64), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            output,
            m,
            n,
            k,
            g,
            split,
            num_warps=4,
            num_stages=1,
        )
    else:
        bm = (
            16
            if m <= 16
            else (
                64 if m >= 128 or (m >= 64 and k > 1024 and n >= 4096) else 32
            )
        )
        _contiguous_matmul[(_ceil_div(m, bm) * _ceil_div(n, 64), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            g,
            split,
            bm,
            num_warps=4,
            num_stages=2 if bm <= 32 else 1,
        )
    if split > 1:
        _awq_sum[(_ceil_div(m * n, 256),)](
            partial,
            output,
            m,
            n,
            split,
            _power_of_two(split),
            256,
            num_warps=4,
            num_stages=1,
        )
    return output


# Specialize contiguous FP32 metadata; other inputs retain the fallback.
# No tensor contents or output buffers are cached between calls.


# Prepare both FP16 planes of A once for the selected matrix shapes.
# Integer delta remains exactly representable in FP16; each group accumulates
# the low product before the high product, followed by FP32 scaling.
# The fully tiled helper is reached only through exact contiguous guards.
@triton.jit
def _r6_prepare(A, H, SIZE: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    a = tl.load(A + x, x < SIZE, 0).to(tl.float32)
    hi = a.to(tl.float16)
    lo = (a - hi.to(tl.float32)).to(tl.float16)
    tl.store(H + x, hi, x < SIZE)
    tl.store(H + SIZE + x, lo, x < SIZE)


@triton.jit
def _r6_prepared(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    part = 0 if SPLIT == 1 else tl.program_id(1)
    rm = tile // (N // BN) * BM + tl.arange(0, BM)
    rn = tile % (N // BN) * BN + tl.arange(0, BN)
    rp = tile % (N // BN) * (BN // 8) + tl.arange(0, BN // 8)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(
        part * (K // G) // SPLIT, (part + 1) * (K // G) // SPLIT
    ):
        z = _unpack_constant(tl.load(Z + group * (N // 8) + rp)).to(tl.float32)
        scale = tl.load(S + group * N + rn)
        ga = tl.zeros((BM, BN), tl.float32)
        for block in range(G // BK):
            rk = group * G + block * BK + tl.arange(0, BK)
            q = _unpack_constant(
                tl.load(Q + rk[:, None] * (N // 8) + rp[None, :])
            ).to(tl.float32)
            b = (q - z[None, :]).to(tl.float16)
            ah = tl.load(A + rm[:, None] * K + rk[None, :])
            al = tl.load(A + M * K + rm[:, None] * K + rk[None, :])
            ga = tl.dot(al, b, ga)
            ga = tl.dot(ah, b, ga)
        acc += ga * scale[None, :]
    tl.store(P + part * M * N + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def _r7_decode_packed(p, shift, DECODE: tl.constexpr):
    if DECODE == 0:
        p = tl.interleave(p, p)
        p = tl.interleave(p, p)
        p = tl.interleave(p, p)
        result = (p >> shift) & 15
    else:
        tl.static_assert(DECODE == 1)
        x0 = p & 15
        x1 = (p >> 16) & 15
        x2 = (p >> 4) & 15
        x3 = (p >> 20) & 15
        x4 = (p >> 8) & 15
        x5 = (p >> 24) & 15
        x6 = (p >> 12) & 15
        x7 = (p >> 28) & 15
        e0 = tl.interleave(x0, x4)
        e1 = tl.interleave(x2, x6)
        o0 = tl.interleave(x1, x5)
        o1 = tl.interleave(x3, x7)
        result = tl.interleave(tl.interleave(e0, e1), tl.interleave(o0, o1))
    return result


@triton.jit
def _r7_weight_tile(
    Q,
    S,
    Z,
    rk,
    k_base,
    rn,
    rp,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    DECODE: tl.constexpr,
    GROUP_SCALAR: tl.constexpr,
):
    tl.static_assert(DECODE == 0 or DECODE == 1 or DECODE == 2)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    if DECODE == 2:
        # Logical columns explicitly repeat each packed word eight times.
        # Whether native lowering coalesces/reuses these loads is a measurement.
        packed = tl.load(
            Q + rk[:, None] * QK + (rn[None, :] // 8) * QN,
            (rk[:, None] < K) & (rn[None, :] < N),
            0,
        )
        weight = (packed >> shift[None, :]) & 15
    else:
        packed = tl.load(
            Q + rk[:, None] * QK + rp[None, :] * QN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        weight = _r7_decode_packed(packed, shift[None, :], DECODE)
    if GROUP_SCALAR:
        # The caller proves G % BK == 0 and starts every tile at a BK multiple.
        # Recompute each iteration, including adjacent same-group tiles at split1.
        group = k_base // G
        if DECODE == 2:
            word = tl.load(
                Z + group * ZG + (rn // 8) * ZN, (group < K // G) & (rn < N), 0
            )
            zero = (word >> shift) & 15
        else:
            word = tl.load(
                Z + group * ZG + rp * ZN, (group < K // G) & (rp < N // 8), 0
            )
            zero = _r7_decode_packed(word, shift, DECODE)
        scale = tl.load(
            S + group * SG + rn * SN, (group < K // G) & (rn < N), 0
        ).to(tl.float32)
        b = (weight.to(tl.float32) - zero[None, :].to(tl.float32)) * scale[
            None, :
        ]
    else:
        # Retain original elementwise-group handling for a tile crossing groups.
        group = rk // G
        if DECODE == 2:
            word = tl.load(
                Z + group[:, None] * ZG + (rn[None, :] // 8) * ZN,
                (rk[:, None] < K) & (rn[None, :] < N),
                0,
            )
            zero = (word >> shift[None, :]) & 15
        else:
            word = tl.load(
                Z + group[:, None] * ZG + rp[None, :] * ZN,
                (rk[:, None] < K) & (rp[None, :] < N // 8),
                0,
            )
            zero = _r7_decode_packed(word, shift[None, :], DECODE)
        scale = tl.load(
            S + group[:, None] * SG + rn[None, :] * SN,
            (rk[:, None] < K) & (rn[None, :] < N),
            0,
        ).to(tl.float32)
        b = (weight.to(tl.float32) - zero.to(tl.float32)) * scale
    return b


@triton.jit
def _r7_control_gemv(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DECODE: tl.constexpr,
    GROUP_SCALAR: tl.constexpr,
):
    tl.static_assert(BN % 8 == 0)
    tl.static_assert(not GROUP_SCALAR or G % BK == 0)
    tile = tl.program_id(0)
    part = tl.program_id(1)
    row = tile // tl.cdiv(N, BN)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
    k_base = part * BK
    rk = k_base + tl.arange(0, BK)
    acc = tl.zeros((BK, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(A + row * AM + rk * AK, rk < K, 0).to(tl.float32)
        b = _r7_weight_tile(
            Q,
            S,
            Z,
            rk,
            k_base,
            rn,
            rp,
            N,
            K,
            G,
            QK,
            QN,
            SG,
            SN,
            ZG,
            ZN,
            DECODE,
            GROUP_SCALAR,
        )
        acc += a[:, None] * b
        rk += BK * SPLIT
        k_base += BK * SPLIT
    result = tl.sum(acc, axis=0)
    tl.store(P + part * M * N + row * N + rn, result, rn < N)


@triton.jit
def _r5_exact(A, Q, S, Z, P, OUT, CASE_ID: tl.constexpr):
    if CASE_ID == 0:
        _awq_control_gemv(
            A,
            Q,
            S,
            Z,
            P,
            16,
            32,
            128,
            128,
            128,
            1,
            4,
            1,
            32,
            1,
            4,
            1,
            1,
            32,
            128,
        )
    elif CASE_ID == 1:
        _awq_control_gemv(
            A,
            Q,
            S,
            Z,
            P,
            24,
            64,
            128,
            32,
            128,
            1,
            8,
            1,
            64,
            1,
            8,
            1,
            1,
            32,
            128,
        )
    elif CASE_ID == 2:
        _awq_control_gemv(
            A,
            Q,
            S,
            Z,
            P,
            32,
            128,
            128,
            128,
            128,
            1,
            16,
            1,
            128,
            1,
            16,
            1,
            1,
            64,
            128,
        )
    elif CASE_ID == 3:
        _awq_control_gemv(
            A,
            Q,
            S,
            Z,
            P,
            1,
            1024,
            4096,
            128,
            4096,
            1,
            128,
            1,
            1024,
            1,
            128,
            1,
            8,
            32,
            64,
        )
    elif CASE_ID == 4:
        _r7_control_gemv(
            A,
            Q,
            S,
            Z,
            P,
            1,
            4096,
            4096,
            128,
            4096,
            1,
            512,
            1,
            4096,
            1,
            512,
            1,
            1,
            32,
            64,
            1,
            True,
        )
    elif CASE_ID == 5:
        _awq_control_matmul(
            A,
            Q,
            S,
            Z,
            P,
            8,
            1024,
            4096,
            128,
            4096,
            1,
            128,
            1,
            1024,
            1,
            128,
            1,
            8,
            16,
            64,
            32,
            "ieee",
        )
    elif CASE_ID == 6:
        _awq_control_matmul(
            A,
            Q,
            S,
            Z,
            P,
            8,
            4096,
            4096,
            128,
            4096,
            1,
            512,
            1,
            4096,
            1,
            512,
            1,
            8,
            16,
            64,
            32,
            "ieee",
        )
    elif CASE_ID == 7:
        _awq_control_matmul(
            A,
            Q,
            S,
            Z,
            P,
            64,
            1024,
            4096,
            128,
            4096,
            1,
            128,
            1,
            1024,
            1,
            128,
            1,
            8,
            32,
            64,
            32,
            "ieee",
        )
    elif CASE_ID == 8:
        _awq_grouped(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
            64,
            4096,
            4096,
            128,
            4096,
            1,
            512,
            1,
            4096,
            1,
            512,
            1,
            "fp16x2",
            64,
            64,
            64,
            8,
            True,
            False,
            True,
        )
    elif CASE_ID == 9:
        _r6_prepared(A, Q, S, Z, P, 512, 1024, 4096, 128, 2, 64, 64, 64)
    elif CASE_ID == 10:
        _r6_prepared(A, Q, S, Z, P, 512, 4096, 4096, 128, 2, 64, 64, 64)


@triton.jit
def _r5_reduce(P, OUT, CASE_ID: tl.constexpr):
    if CASE_ID == 0:
        _awq_sum(P, OUT, 16, 32, 1, 1, 256)
    elif CASE_ID == 1:
        _awq_sum(P, OUT, 24, 64, 1, 1, 256)
    elif CASE_ID == 2:
        _awq_sum(P, OUT, 32, 128, 1, 1, 256)
    elif CASE_ID == 3:
        _awq_sum(P, OUT, 1, 1024, 8, 8, 256)
    elif CASE_ID == 4:
        _awq_sum(P, OUT, 1, 4096, 1, 1, 256)
    elif CASE_ID == 5:
        _awq_sum(P, OUT, 8, 1024, 8, 8, 256)
    elif CASE_ID == 6:
        _awq_sum(P, OUT, 8, 4096, 8, 8, 256)
    elif CASE_ID == 7:
        _awq_sum(P, OUT, 64, 1024, 8, 8, 256)
    elif CASE_ID == 8:
        _awq_sum(P, OUT, 64, 4096, 8, 8, 256)
    elif CASE_ID == 9:
        _awq_sum(P, OUT, 512, 1024, 2, 2, 256)
    elif CASE_ID == 10:
        _awq_sum(P, OUT, 512, 4096, 2, 2, 256)


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    i = -1
    if m == 16 and n == 32 and k == 128 and groups == 1 and split_k_iters == 1:
        i = 0
    elif (
        m == 24 and n == 64 and k == 128 and groups == 4 and split_k_iters == 8
    ):
        i = 1
    elif (
        m == 32
        and n == 128
        and k == 128
        and groups == 1
        and split_k_iters == 1
    ):
        i = 2
    elif (
        m == 1
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 3
    elif (
        m == 1
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 4
    elif (
        m == 8
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 5
    elif (
        m == 8
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 6
    elif (
        m == 64
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 7
    elif (
        m == 64
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 8
    elif (
        m == 512
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 9
    elif (
        m == 512
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 10
    if (
        i < 0
        or input.dtype != torch.float32
        or scales.dtype != torch.float32
        or qweight.dtype != torch.int32
        or qzeros.dtype != torch.int32
        or qweight.shape != (k, n // 8)
        or scales.shape != (groups, n)
        or qzeros.shape != (groups, n // 8)
        or not (
            input.is_contiguous()
            and qweight.is_contiguous()
            and scales.is_contiguous()
            and qzeros.is_contiguous()
        )
    ):
        return _r4_dispatch(input, qweight, scales, qzeros, split_k_iters)
    out = torch.empty((m, n), device=input.device, dtype=torch.float32)
    if i == 0:
        split, grid, warps, stages = 1, 16, 4, 1
    elif i == 1:
        split, grid, warps, stages = 1, 48, 4, 1
    elif i == 2:
        split, grid, warps, stages = 1, 64, 4, 1
    elif i == 3:
        split, grid, warps, stages = 8, 32, 4, 2
    elif i == 4:
        split, grid, warps, stages = 1, 128, 4, 2
    elif i == 5:
        split, grid, warps, stages = 8, 16, 4, 2
    elif i == 6:
        split, grid, warps, stages = 8, 64, 4, 2
    elif i == 7:
        split, grid, warps, stages = 8, 32, 4, 2
    elif i == 8:
        split, grid, warps, stages = 8, 64, 4, 1
    elif i == 9:
        split, grid, warps, stages = 2, 128, 4, 1
    elif i == 10:
        split, grid, warps, stages = 2, 512, 4, 1
    partial = (
        out
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    if i == 9 or i == 10:
        h = torch.empty((2, m, k), device=input.device, dtype=torch.float16)
        _r6_prepare[(m * k // 256,)](
            input, h, m * k, num_warps=4, num_stages=1
        )
        input = h
    _r5_exact.run(
        input,
        qweight,
        scales,
        qzeros,
        partial,
        out,
        i,
        grid=(grid, split),
        warmup=False,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _r5_reduce.run(
            partial,
            out,
            i,
            grid=((m * n + 255) // 256,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["awq_gemm"]
