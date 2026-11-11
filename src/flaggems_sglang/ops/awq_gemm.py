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

"""AWQ INT4 GEMM with NVIDIA/AMD dispatch and shape-specialized kernels.

Explicit shape routes use fixed tiles and compact JIT entries. Other shapes
and layouts retain stride-aware Triton paths. Split-K reduction is performed
by custom Triton kernels. Architecture-dependent compilation is retained.
"""
import torch
import triton
import triton.language as tl


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
def _awq_kernel(
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
    ARCH: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    VECTOR: tl.constexpr,
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
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
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


def _finish_candidate(args, exception):
    if exception is None and args["SPLIT"] > 1:
        s = args["SPLIT"]
        _awq_sum[(triton.cdiv(args["M"] * args["N"], 256),)](
            args["P"],
            args["OUT"],
            args["M"],
            args["N"],
            s,
            triton.next_power_of_2(s),
            256,
            num_warps=4,
        )


def _configs():
    # Diverse bounded seeds, not an exhaustive Cartesian search.
    tiles = (
        (16, 64, 32, 4, 1, False),
        (16, 64, 64, 4, 2, False),
        (16, 128, 64, 4, 3, False),
        (16, 64, 128, 4, 3, False),
        (16, 64, 32, 2, 2, True),
        (16, 64, 64, 4, 2, True),
        (16, 128, 64, 4, 3, True),
        (16, 64, 128, 4, 1, True),
        (32, 64, 32, 4, 1, False),
        (32, 64, 64, 4, 3, False),
        (32, 128, 64, 4, 2, False),
        (32, 64, 128, 4, 3, False),
        (64, 64, 32, 4, 1, False),
        (64, 64, 64, 4, 2, False),
        (64, 128, 64, 8, 4, False),
        (64, 64, 128, 4, 3, False),
        (32, 64, 32, 4, 1, True),
        (32, 64, 64, 4, 2, True),
        (64, 64, 32, 4, 1, True),
        (64, 64, 64, 4, 2, True),
    )
    out = []
    for mode in ("ieee", "fp16x2"):
        for bm, bn, bk, nw, ns, t in tiles:
            for split in (1, 2, 4, 8, 16):
                out.append(
                    triton.Config(
                        dict(
                            BM=bm,
                            BN=bn,
                            BK=bk,
                            SPLIT=split,
                            TRANSPOSE=t,
                            VECTOR=False,
                            MODE=mode,
                        ),
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    for bn, bk in ((32, 32), (64, 32), (32, 64), (64, 64), (32, 128)):
        for split in (1, 2, 4, 8, 16):
            out.append(
                triton.Config(
                    dict(
                        BM=1,
                        BN=bn,
                        BK=bk,
                        SPLIT=split,
                        TRANSPOSE=False,
                        VECTOR=True,
                        MODE="ieee",
                    ),
                    num_warps=4,
                    num_stages=1,
                )
            )
    return out


def _prune(configs, named_args, **kwargs):
    m, n, k, g = (named_args[x] for x in ("M", "N", "K", "G"))
    requested = named_args["REQUESTED"]
    # Up to three whole-pipeline split candidates per tile.
    # TF32 was excluded after failing normal positive-distribution validation.
    # A group-aligned split always covers all groups, even for unequal chunks.
    groups = k // g
    max_split = min(16, 1 << (groups.bit_length() - 1))
    occupancy = triton.cdiv(m, 16 if m <= 16 else 64) * triton.cdiv(n, 64)
    target = min(
        max_split, max(1, triton.next_power_of_2(triton.cdiv(128, occupancy)))
    )
    requested = min(max_split, triton.next_power_of_2(requested))
    splits = {1, target, requested}
    legal = []
    for c in configs:
        p = c.kwargs
        if p["SPLIT"] not in splits or p["SPLIT"] > groups:
            continue
        if p["TRANSPOSE"] and m > 16 and p["MODE"] != "ieee":
            continue
        if p["VECTOR"]:
            if m != 1:
                continue
        elif m == 1:
            continue
        elif (m <= 16 and p["BM"] != 16) or (
            m > 16 and p["BM"] not in (32, 64)
        ):
            continue
        if p["BK"] > g and p["BK"] != 32 and not p["VECTOR"]:
            continue
        if p["BK"] > g and p["VECTOR"] and p["BK"] != 64:
            continue
        if g % p["BK"] != 0 and p["BK"] != (64 if p["VECTOR"] else 32):
            continue
        legal.append(c)
    return legal


def _awq_bench_no_graph(fn, quantiles=(0.5, 0.2, 0.8)):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 2.0)
    samples.sort()
    if not quantiles:
        return samples[2]
    result = []
    for q in quantiles:
        position = q * 4
        lo = int(position)
        hi = min(lo + 1, 4)
        result.append(
            samples[lo] + (samples[hi] - samples[lo]) * (position - lo)
        )
    return result


_awq_tuned = triton.autotune(
    configs=_configs(),
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
        "ARCH",
    ],
    prune_configs_by={"early_config_prune": _prune},
    post_hook=_finish_candidate,
    # Short bounded measurements, with the reduction included by post_hook.
    do_bench=_awq_bench_no_graph,
)(_awq_kernel)


def _awq_tuned_entry(input, qweight, scales, qzeros, split_k_iters, m, n, k):
    scale_shape = scales.shape
    groups = scale_shape[0]
    assert split_k_iters > 0
    assert qweight.shape[0] == k and scale_shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    device = input.device
    output_dtype = scales.dtype
    output = torch.empty((m, n), device=device, dtype=output_dtype)
    if m == 0 or n == 0:
        return output
    size = m * n
    if k == 0:
        _awq_zero[((size + 255) // 256,)](output, size)
        return output
    g = k // groups
    am, ak = input.stride()
    qk, qn = qweight.stride()
    sg, sn = scales.stride()
    zg, zn = qzeros.stride()
    meta = (
        m,
        n,
        k,
        g,
        am,
        ak,
        qk,
        qn,
        sg,
        sn,
        zg,
        zn,
        split_k_iters,
        device.index,
        0,
    )  # Device index separates tuning; Triton JIT handles target architecture.
    # Preserve the native tuning key exactly; retain only measured configurations.
    scale_dtype_key = str(output_dtype)
    key = meta + (
        str(input.dtype),
        str(qweight.dtype),
        scale_dtype_key,
        str(qzeros.dtype),
        "torch.float32",
        scale_dtype_key,
    )
    config = _awq_tuned.cache.get(key)
    if config is None:
        partial = torch.empty(
            (min(16, groups), m, n), device=device, dtype=torch.float32
        )
        _awq_tuned[
            lambda p: (
                ((m + p["BM"] - 1) // p["BM"])
                * ((n + p["BN"] - 1) // p["BN"]),
                p["SPLIT"],
            )
        ](input, qweight, scales, qzeros, partial, output, *meta)
        config = _awq_tuned.best_config
    else:
        _awq_tuned.best_config = config
        p = config.kwargs
        partial = (
            output
            if p["SPLIT"] == 1
            else torch.empty(
                (p["SPLIT"], m, n), device=device, dtype=torch.float32
            )
        )
        _awq_kernel[
            (
                ((m + p["BM"] - 1) // p["BM"])
                * ((n + p["BN"] - 1) // p["BN"]),
                p["SPLIT"],
            )
        ](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            output,
            *meta,
            **config.all_kwargs(),
        )
    s = config.kwargs["SPLIT"]
    if s > 1:
        _awq_sum[((size + 255) // 256,)](
            partial,
            output,
            m,
            n,
            s,
            triton.next_power_of_2(s),
            256,
            num_warps=4,
        )
    return output


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
        ah = a.to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bh, acc)
        rk += BK * SPLIT
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _awq_control_sum(
    P,
    OUT,
    SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, RS)
    value = tl.load(
        P + part[:, None] * SIZE + index[None, :],
        (part[:, None] < SPLIT) & (index[None, :] < SIZE),
        0,
    )
    result = tl.sum(value, axis=0)
    tl.store(OUT + index, result, index < SIZE)


@triton.jit
def _awq_control_zero(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, 0.0, index < SIZE)


@triton.jit
def _awq_small_tc(
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
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tl.static_assert(M <= 16)
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    rm = tl.arange(0, 16)
    rn = tile_n * 64 + tl.arange(0, 64)
    rp = tile_n * 8 + tl.arange(0, 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((64, 16), tl.float32)
    group_begin = part * (K // G) // SPLIT
    group_end = (part + 1) * (K // G) // SPLIT
    for group in range(group_begin, group_end):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        group_acc = tl.zeros((64, 16), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local_k = block * BK + tl.arange(0, BK)
            rk = group * G + local_k
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local_k[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            b = tl.where(local_k[:, None] < G, delta, 0.0)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm[:, None] < M) & (local_k[None, :] < G),
                0,
            ).to(tl.float32)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            bh = b.to(tl.float16)
            group_acc = tl.dot(tl.trans(bh), tl.trans(al), group_acc)
            group_acc = tl.dot(tl.trans(bh), tl.trans(ah), group_acc)
        acc += group_acc * scale[:, None]
    value = tl.trans(acc)
    offset = part * M * N + rm[:, None] * N + rn[None, :]
    tl.store(P + offset, value, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _awq_small_simt(
    A,
    Q,
    S,
    Z,
    OUT,
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
):
    rn = tl.program_id(0) * 32 + tl.arange(0, 32)
    rp = tl.program_id(0) * 4 + tl.arange(0, 4)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc0 = tl.zeros((32, 32), tl.float32)
    acc1 = tl.zeros((32, 32), tl.float32)
    for group in range(K // G):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        for block in range(tl.cdiv(G, 32)):
            local_k = block * 32 + tl.arange(0, 32)
            rk = group * G + local_k
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local_k[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            b = tl.where(local_k[:, None] < G, delta, 0.0) * scale[None, :]
            a0 = tl.load(A + rk * AK, local_k < G, 0).to(tl.float32)
            a1 = tl.load(A + AM + rk * AK, local_k < G, 0).to(tl.float32)
            # Decode B once and reuse it across both rows; reduce only at the end.
            acc0 += a0[:, None] * b
            acc1 += a1[:, None] * b
    tl.store(OUT + rn, tl.sum(acc0, 0), rn < N)
    tl.store(OUT + N + rn, tl.sum(acc1, 0), rn < N)


def _awq_control_entry(input, qweight, scales, qzeros, split_k_iters, m, n, k):
    """Run the original static kernels with call-local launch metadata."""
    scale_shape = scales.shape
    groups = scale_shape[0]
    assert split_k_iters > 0
    assert qweight.shape[0] == k
    assert scale_shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    device = input.device
    output = torch.empty((m, n), device=device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return output
    size = m * n
    if k == 0:
        _awq_control_zero[((size + 255) // 256,)](output, size, 256)
        return output
    group_size = k // groups
    am, ak = input.stride()
    qk, qn = qweight.stride()
    sg, sn = scales.stride()
    zg, zn = qzeros.stride()
    if m == 2 and k <= 128:
        # Short K: both rows reuse a decoded B tile and reduce K only once.
        _awq_small_simt[((n + 31) // 32,)](
            input,
            qweight,
            scales,
            qzeros,
            output,
            n,
            k,
            group_size,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            num_warps=4,
            num_stages=1,
        )
        return output
    if 2 <= m <= 8:
        # Small M occupies columns of a [64,16] WGMMA tile.
        split = 1 if k <= 128 else min(split_k_iters, groups)
        bk = (
            128
            if group_size % 128 == 0
            else (64 if group_size % 64 == 0 else 32)
        )
        partial = (
            output
            if split == 1
            else torch.empty((split, m, n), device=device, dtype=torch.float32)
        )
        _awq_small_tc[((n + 63) // 64, split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            group_size,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            bk,
            split,
            num_warps=4,
            num_stages=1,
        )
        if split > 1:
            _awq_sum[((size + 255) // 256,)](
                partial,
                output,
                m,
                n,
                split,
                triton.next_power_of_2(split),
                256,
                num_warps=4,
            )
        return output
    split = split_k_iters
    partial = (
        output
        if split == 1
        else torch.empty((split, m, n), device=device, dtype=torch.float32)
    )
    if m == 1:
        _awq_control_gemv[((n + 31) // 32, split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            group_size,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            split,
            32,
            64,
            num_warps=4,
            num_stages=2,
        )
    else:
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        bn, bk = 64, (64 if m <= 32 else 32)
        _awq_control_matmul[
            (((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)
        ](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            group_size,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            split,
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=2 if m <= 32 else 3,
        )
    if split != 1:
        _awq_control_sum[((size + 255) // 256,)](
            partial,
            output,
            size,
            split,
            triton.next_power_of_2(split),
            256,
            num_warps=4,
        )
    return output


def _awq_legacy(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    if (m <= 64 or k <= 128) and not (m == 1 and n >= 8192 and 128 < k <= 512):
        return _awq_control_entry(
            input, qweight, scales, qzeros, split_k_iters, m, n, k
        )
    return _awq_tuned_entry(
        input, qweight, scales, qzeros, split_k_iters, m, n, k
    )


@tl.core.builtin
def _awq_compiler_target_v1(_semantic=None):
    # Read actual compile options, including for offline/asynchronous compilation.
    # This fixed accessor has no dispatch policy or persistent state.
    options = _semantic.builder.options
    return tl.constexpr(
        (getattr(options, "backend_name", ""), getattr(options, "arch", ""))
    )


@triton.jit
def _awq_group_dot(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    compile_target: tl.constexpr = _awq_compiler_target_v1()
    # Compiler target options select dot-dependent-load staging; other
    # targets keep a conservative footprint without a per-call host query.
    pipeline_hint: tl.constexpr = (
        None
        if compile_target[0] == "cuda" and compile_target[1] == "sm90"
        else 1
    )
    for group in tl.range(
        part * (K // G) // SPLIT,
        (part + 1) * (K // G) // SPLIT,
        num_stages=pipeline_hint,
    ):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            b = tl.where(local[:, None] < G, delta, 0.0).to(tl.float16)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm[:, None] < M) & (local[None, :] < G),
                0,
            ).to(tl.float32)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            group_acc = tl.dot(al, b, group_acc)
            group_acc = tl.dot(ah, b, group_acc)
        acc += group_acc * scale[None, :]
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _awq_small_prefetch(
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
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tl.static_assert(M <= 16)
    tl.static_assert(G == 128 and BK == 128)
    tl.static_assert(K % G == 0)
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    rm = tl.arange(0, 16)
    rn = tile_n * 64 + tl.arange(0, 64)
    rp = tile_n * 8 + tl.arange(0, 8)
    local_k = tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((64, 16), tl.float32)
    group_begin = part * (K // G) // SPLIT
    group_end = (part + 1) * (K // G) // SPLIT

    # Prologue: retain one complete group's raw operands in SSA values.
    first_valid = group_begin < group_end
    first_k = group_begin * G + local_k
    a_raw = tl.load(
        A + rm[:, None] * AM + first_k[None, :] * AK,
        (rm[:, None] < M) & first_valid,
        0,
    )
    q_raw = tl.load(
        Q + first_k[:, None] * QK + rp[None, :] * QN,
        (rp[None, :] < N // 8) & first_valid,
        0,
    )
    z_raw = tl.load(
        Z + group_begin * ZG + rp * ZN, (rp < N // 8) & first_valid, 0
    )
    s_raw = tl.load(S + group_begin * SG + rn * SN, (rn < N) & first_valid, 0)

    for group in range(group_begin, group_end):
        # Load the next group's raw operands before current-group arithmetic.
        # The final iteration uses masked loads; it never reads another split.
        next_group = group + 1
        next_valid = next_group < group_end
        next_k = next_group * G + local_k
        a_next = tl.load(
            A + rm[:, None] * AM + next_k[None, :] * AK,
            (rm[:, None] < M) & next_valid,
            0,
        )
        q_next = tl.load(
            Q + next_k[:, None] * QK + rp[None, :] * QN,
            (rp[None, :] < N // 8) & next_valid,
            0,
        )
        z_next = tl.load(
            Z + next_group * ZG + rp * ZN, (rp < N // 8) & next_valid, 0
        )
        s_next = tl.load(
            S + next_group * SG + rn * SN, (rn < N) & next_valid, 0
        )

        # Same dequantization and two-dot ordering as _awq_small_tc.
        z = tl.interleave(z_raw, z_raw)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = s_raw.to(tl.float32)
        q = tl.interleave(q_raw, q_raw)
        q = tl.interleave(q, q)
        q = tl.interleave(q, q)
        delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
        b = tl.where(local_k[:, None] < G, delta, 0.0)
        a = a_raw.to(tl.float32)
        ah = a.to(tl.float16)
        al = (a - ah.to(tl.float32)).to(tl.float16)
        bh = b.to(tl.float16)
        group_acc = tl.zeros((64, 16), tl.float32)
        group_acc = tl.dot(tl.trans(bh), tl.trans(al), group_acc)
        group_acc = tl.dot(tl.trans(bh), tl.trans(ah), group_acc)
        acc += group_acc * scale[:, None]

        # SSA exchange, not an in-place shared-memory overwrite.
        a_raw = a_next
        q_raw = q_next
        z_raw = z_next
        s_raw = s_next

    value = tl.trans(acc)
    offset = part * M * N + rm[:, None] * N + rn[None, :]
    tl.store(P + offset, value, (rm[:, None] < M) & (rn[None, :] < N))


def _awq_nv_previous(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    if not (input.dtype == torch.float32 and scales.dtype == torch.float32):
        return _awq_legacy(input, qweight, scales, qzeros, split_k_iters)
    tiny = k == 128 and 8 <= m <= 32 and 0 < n <= 128 and groups in (1, 4)
    regular = (
        k == 4096
        and groups == 32
        and m in (1, 8, 64, 512)
        and n in (1024, 4096)
        and split_k_iters == 8
    )
    if not (tiny or regular):
        return _awq_legacy(input, qweight, scales, qzeros, split_k_iters)
    if m == 512 and n == 4096:
        props = torch.cuda.get_device_properties(input.device)
        if getattr(props, "shared_memory_per_block_optin", 0) < 180352:
            return _awq_legacy(input, qweight, scales, qzeros, split_k_iters)
    assert split_k_iters > 0 and qweight.shape[0] == k and scales.shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    am, ak = input.stride()
    qk, qn = qweight.stride()
    sg, sn = scales.stride()
    zg, zn = qzeros.stride()
    if tiny:
        bn = 16 if n <= 64 else 32
        _awq_control_gemv[(m * ((n + bn - 1) // bn), 1)](
            input,
            qweight,
            scales,
            qzeros,
            output,
            m,
            n,
            k,
            k // groups,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            1,
            bn,
            128,
            num_warps=4,
            num_stages=1,
        )
        return output
    split = (
        16 if m == 8 and n == 1024 else (1 if m == 512 and n == 4096 else 8)
    )
    partial = (
        output
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    if m == 1:
        _awq_control_gemv[((n + 31) // 32, split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            128,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            split,
            32,
            128,
            num_warps=4,
            num_stages=1,
        )
    elif m == 8:
        _awq_small_prefetch[((n + 63) // 64, split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            128,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            128,
            split,
            num_warps=4,
            num_stages=1,
        )
    else:
        bm = 128 if m == 512 and n == 4096 else 64
        bn = 128 if (m == 64 and n == 4096) or m == 512 else 64
        nw = 8 if bm == 128 else 4
        ns = 3
        _awq_group_dot[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            128,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            bm,
            bn,
            128,
            split,
            num_warps=nw,
            num_stages=ns,
        )
    if split > 1:
        _awq_sum[((m * n + 255) // 256,)](
            partial, output, m, n, split, split, 256, num_warps=4
        )
    return output


@triton.jit
def _awq_nv_case_0(A, Q, S, Z, P):
    _awq_control_gemv(
        A, Q, S, Z, P, 16, 32, 128, 128, 128, 1, 4, 1, 32, 1, 4, 1, 1, 8, 128
    )


@triton.jit
def _awq_nv_case_1(A, Q, S, Z, P):
    _awq_control_gemv(
        A, Q, S, Z, P, 24, 64, 128, 32, 128, 1, 8, 1, 64, 1, 8, 1, 1, 8, 128
    )


@triton.jit
def _awq_nv_case_2(A, Q, S, Z, P):
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
        32,
        128,
    )


@triton.jit
def _awq_nv_case_3(A, Q, S, Z, P):
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
        128,
    )


@triton.jit
def _awq_nv_sum_3(P, OUT):
    _awq_sum(P, OUT, 1, 1024, 8, 8, 256)


@triton.jit
def _awq_nv_case_4(A, Q, S, Z, P):
    _awq_control_gemv(
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
        8,
        32,
        128,
    )


@triton.jit
def _awq_nv_sum_4(P, OUT):
    _awq_sum(P, OUT, 1, 4096, 8, 8, 256)


@triton.jit
def _awq_nv_case_5(A, Q, S, Z, P):
    _awq_small_prefetch(
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
        128,
        16,
    )


@triton.jit
def _awq_nv_sum_5(P, OUT):
    _awq_sum(P, OUT, 8, 1024, 16, 16, 256)


@triton.jit
def _awq_nv_case_6(A, Q, S, Z, P):
    _awq_small_prefetch(
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
        128,
        8,
    )


@triton.jit
def _awq_nv_sum_6(P, OUT):
    _awq_sum(P, OUT, 8, 4096, 8, 8, 256)


@triton.jit
def _awq_nv_case_7(A, Q, S, Z, P):
    _awq_group_dot(
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
        64,
        64,
        128,
        16,
    )


@triton.jit
def _awq_nv_sum_7(P, OUT):
    _awq_sum(P, OUT, 64, 1024, 16, 16, 256)


@triton.jit
def _awq_nv_case_8(A, Q, S, Z, P):
    _awq_group_dot(
        A,
        Q,
        S,
        Z,
        P,
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
        64,
        128,
        128,
        8,
    )


@triton.jit
def _awq_nv_sum_8(P, OUT):
    _awq_sum(P, OUT, 64, 4096, 8, 8, 256)


@triton.jit
def _awq_nv_case_9(A, Q, S, Z, P):
    _awq_group_dot(
        A,
        Q,
        S,
        Z,
        P,
        512,
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
        128,
        128,
        128,
        4,
    )


@triton.jit
def _awq_nv_sum_9(P, OUT):
    _awq_sum(P, OUT, 512, 1024, 4, 4, 256)


@triton.jit
def _awq_nv_case_10(A, Q, S, Z, P):
    _awq_group_dot(
        A,
        Q,
        S,
        Z,
        P,
        512,
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
        128,
        128,
        128,
        1,
    )


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    if bool(getattr(torch.version, "hip", None)):
        return awq_amd(input, qweight, scales, qzeros, split_k_iters)
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    if not (
        input.dtype == torch.float32
        and scales.dtype == torch.float32
        and input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    ):
        return _awq_nv_previous(input, qweight, scales, qzeros, split_k_iters)
    if m == 16 and n == 32 and k == 128 and groups == 1 and split_k_iters == 1:
        output = torch.empty(
            (16, 32), device=input.device, dtype=torch.float32
        )
        partial = output
        _awq_nv_case_0.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 1),
            warmup=False,
            num_warps=2,
            num_stages=1,
        )
        return output
    if m == 24 and n == 64 and k == 128 and groups == 4 and split_k_iters == 8:
        output = torch.empty(
            (24, 64), device=input.device, dtype=torch.float32
        )
        partial = output
        _awq_nv_case_1.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(192, 1),
            warmup=False,
            num_warps=2,
            num_stages=1,
        )
        return output
    if (
        m == 32
        and n == 128
        and k == 128
        and groups == 1
        and split_k_iters == 1
    ):
        output = torch.empty(
            (32, 128), device=input.device, dtype=torch.float32
        )
        partial = output
        _awq_nv_case_2.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(128, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 1
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (1, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 1, 1024), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_3.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(32, 8),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _awq_nv_sum_3.run(
            partial, output, grid=(4,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 1
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (1, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 1, 4096), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_4.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(128, 8),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _awq_nv_sum_4.run(
            partial, output, grid=(16,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 8
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (8, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (16, 8, 1024), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_5.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(16, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _awq_nv_sum_5.run(
            partial, output, grid=(32,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 8
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (8, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 8, 4096), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_6.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 8),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _awq_nv_sum_6.run(
            partial, output, grid=(128,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 64
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (64, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (16, 64, 1024), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_7.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(16, 16),
            warmup=False,
            num_warps=4,
            num_stages=3,
        )
        _awq_nv_sum_7.run(
            partial, output, grid=(256,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 64
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (64, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 64, 4096), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_8.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(32, 8),
            warmup=False,
            num_warps=4,
            num_stages=3,
        )
        _awq_nv_sum_8.run(
            partial, output, grid=(1024,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 512
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        props = getattr(
            torch.cuda,
            "_get_device_properties",
            torch.cuda.get_device_properties,
        )(input.device.index)
        if getattr(props, "shared_memory_per_block_optin", 0) < 180352:
            return _awq_nv_previous(
                input, qweight, scales, qzeros, split_k_iters
            )
        output = torch.empty(
            (512, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (4, 512, 1024), device=input.device, dtype=torch.float32
        )
        _awq_nv_case_9.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(32, 4),
            warmup=False,
            num_warps=8,
            num_stages=3,
        )
        _awq_nv_sum_9.run(
            partial, output, grid=(2048,), warmup=False, num_warps=4
        )
        return output
    if (
        m == 512
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        props = getattr(
            torch.cuda,
            "_get_device_properties",
            torch.cuda.get_device_properties,
        )(input.device.index)
        if getattr(props, "shared_memory_per_block_optin", 0) < 180352:
            return _awq_nv_previous(
                input, qweight, scales, qzeros, split_k_iters
            )
        output = torch.empty(
            (512, 4096), device=input.device, dtype=torch.float32
        )
        partial = output
        _awq_nv_case_10.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(128, 1),
            warmup=False,
            num_warps=8,
            num_stages=3,
        )
        return output
    return _awq_nv_previous(input, qweight, scales, qzeros, split_k_iters)


@triton.jit
def _amd_sum(
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
def _amd_gemv(
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
def _amd_group(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANS: tl.constexpr,
    UNPACK: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    if TRANS:
        acc = tl.zeros((BN, BM), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    for group in range(
        part * (K // G) // SPLIT, (part + 1) * (K // G) // SPLIT
    ):
        if UNPACK:
            z = tl.load(Z + group * ZG + (rn // 8) * ZN, rn < N, 0)
        else:
            z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        if TRANS:
            ga = tl.zeros((BN, BM), tl.float32)
        else:
            ga = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            if UNPACK:
                q = tl.load(
                    Q + rk[:, None] * QK + (rn[None, :] // 8) * QN,
                    (local[:, None] < G) & (rn[None, :] < N),
                    0,
                )
            else:
                q = tl.load(
                    Q + rk[:, None] * QK + rp[None, :] * QN,
                    (local[:, None] < G) & (rp[None, :] < N // 8),
                    0,
                )
                q = tl.interleave(q, q)
                q = tl.interleave(q, q)
                q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            b = tl.where(local[:, None] < G, delta, 0.0).to(tl.float16)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm[:, None] < M) & (local[None, :] < G),
                0,
            ).to(tl.float32)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            if TRANS:
                ga = tl.dot(tl.trans(b), tl.trans(al), ga)
                ga = tl.dot(tl.trans(b), tl.trans(ah), ga)
            else:
                ga = tl.dot(al, b, ga)
                ga = tl.dot(ah, b, ga)
        if TRANS:
            acc += ga * scale[:, None]
        else:
            acc += ga * scale[None, :]
    if TRANS:
        val = tl.trans(acc)
    else:
        val = acc
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        val,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _amd_prefetch(
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
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
):
    tl.static_assert(M <= 16)
    tl.static_assert(G == 128 and BK == 128)
    tl.static_assert(K % G == 0)
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    rm = tl.arange(0, 16)
    rn = tile_n * BN + tl.arange(0, BN)
    rp = tile_n * (BN // 8) + tl.arange(0, BN // 8)
    local_k = tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BN, 16), tl.float32)
    group_begin = part * (K // G) // SPLIT
    group_end = (part + 1) * (K // G) // SPLIT

    # Prologue: retain one complete group's raw operands in SSA values.
    first_valid = group_begin < group_end
    first_k = group_begin * G + local_k
    a_raw = tl.load(
        A + rm[:, None] * AM + first_k[None, :] * AK,
        (rm[:, None] < M) & first_valid,
        0,
    )
    q_raw = tl.load(
        Q + first_k[:, None] * QK + rp[None, :] * QN,
        (rp[None, :] < N // 8) & first_valid,
        0,
    )
    z_raw = tl.load(
        Z + group_begin * ZG + rp * ZN, (rp < N // 8) & first_valid, 0
    )
    s_raw = tl.load(S + group_begin * SG + rn * SN, (rn < N) & first_valid, 0)

    for group in range(group_begin, group_end):
        # Load the next group's raw operands before current-group arithmetic.
        # The final iteration uses masked loads; it never reads another split.
        next_group = group + 1
        next_valid = next_group < group_end
        next_k = next_group * G + local_k
        a_next = tl.load(
            A + rm[:, None] * AM + next_k[None, :] * AK,
            (rm[:, None] < M) & next_valid,
            0,
        )
        q_next = tl.load(
            Q + next_k[:, None] * QK + rp[None, :] * QN,
            (rp[None, :] < N // 8) & next_valid,
            0,
        )
        z_next = tl.load(
            Z + next_group * ZG + rp * ZN, (rp < N // 8) & next_valid, 0
        )
        s_next = tl.load(
            S + next_group * SG + rn * SN, (rn < N) & next_valid, 0
        )

        # Same dequantization and two-dot ordering as _awq_small_tc.
        z = tl.interleave(z_raw, z_raw)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = s_raw.to(tl.float32)
        q = tl.interleave(q_raw, q_raw)
        q = tl.interleave(q, q)
        q = tl.interleave(q, q)
        delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
        b = tl.where(local_k[:, None] < G, delta, 0.0)
        a = a_raw.to(tl.float32)
        ah = a.to(tl.float16)
        al = (a - ah.to(tl.float32)).to(tl.float16)
        bh = b.to(tl.float16)
        group_acc = tl.zeros((BN, 16), tl.float32)
        group_acc = tl.dot(tl.trans(bh), tl.trans(al), group_acc)
        group_acc = tl.dot(tl.trans(bh), tl.trans(ah), group_acc)
        acc += group_acc * scale[:, None]

        # SSA exchange, not an in-place shared-memory overwrite.
        a_raw = a_next
        q_raw = q_next
        z_raw = z_next
        s_raw = s_next

    value = tl.trans(acc)
    offset = part * M * N + rm[:, None] * N + rn[None, :]
    tl.store(P + offset, value, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _amd_zero(OUT, SIZE: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(OUT + i, 0.0, i < SIZE)


def _awq_amd_general(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return output
    if k == 0:
        _amd_zero.run(
            output,
            m * n,
            grid=(triton.cdiv(m * n, 256),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    groups = scales.shape[0]
    assert split_k_iters > 0 and groups > 0 and k % groups == 0
    assert (
        qweight.shape[0] == k
        and scales.shape[1] == n
        and qzeros.shape == (groups, n // 8)
    )
    g = k // groups
    cfg = ("simt", 1, 16, 64, min(8, split_k_iters), 4, 1, False, False)
    if input.dtype == torch.float32 and scales.dtype == torch.float32:
        if (
            m == 16
            and n == 32
            and k == 128
            and g == 128
            and split_k_iters == 1
        ):
            cfg = ("simt", 1, 8, 128, 1, 4, 1, False, False)
        elif (
            m == 24 and n == 64 and k == 128 and g == 32 and split_k_iters == 8
        ):
            cfg = ("simt", 1, 8, 128, 1, 4, 1, False, False)
        elif (
            m == 32
            and n == 128
            and k == 128
            and g == 128
            and split_k_iters == 1
        ):
            cfg = ("simt", 1, 32, 128, 1, 4, 1, False, False)
        elif (
            m == 1
            and n == 1024
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("simt", 1, 16, 128, 16, 4, 1, False, False)
        elif (
            m == 1
            and n == 4096
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("simt", 1, 64, 128, 8, 4, 1, False, False)
        elif (
            m == 8
            and n == 1024
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("prefetch", 16, 64, 128, 16, 4, 1, True, False)
        elif (
            m == 8
            and n == 4096
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("prefetch", 16, 64, 128, 16, 4, 1, True, False)
        elif (
            m == 64
            and n == 1024
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("wmma", 16, 32, 32, 8, 1, 1, False, False)
        elif (
            m == 64
            and n == 4096
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("wmma", 16, 32, 32, 8, 1, 1, False, False)
        elif (
            m == 512
            and n == 1024
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("wmma", 16, 32, 32, 8, 1, 1, False, False)
        elif (
            m == 512
            and n == 4096
            and k == 4096
            and g == 128
            and split_k_iters == 8
        ):
            cfg = ("wmma", 16, 32, 32, 8, 1, 1, False, False)
    family, bm, bn, bk, split, warps, stages, trans, unpack = cfg
    p = (
        output
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    am, ak = input.stride()
    qk, qn = qweight.stride()
    sg, sn = scales.stride()
    zg, zn = qzeros.stride()
    if family == "simt":
        _amd_gemv.run(
            input,
            qweight,
            scales,
            qzeros,
            p,
            m,
            n,
            k,
            g,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            split,
            bn,
            bk,
            grid=(m * triton.cdiv(n, bn), split),
            warmup=False,
            num_warps=warps,
            num_stages=stages,
        )
    elif family == "prefetch":
        _amd_prefetch.run(
            input,
            qweight,
            scales,
            qzeros,
            p,
            m,
            n,
            k,
            g,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            bk,
            split,
            bn,
            grid=(triton.cdiv(n, bn), split),
            warmup=False,
            num_warps=warps,
            num_stages=stages,
        )
    else:
        _amd_group.run(
            input,
            qweight,
            scales,
            qzeros,
            p,
            m,
            n,
            k,
            g,
            am,
            ak,
            qk,
            qn,
            sg,
            sn,
            zg,
            zn,
            bm,
            bn,
            bk,
            split,
            trans,
            unpack,
            grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn), split),
            warmup=False,
            num_warps=warps,
            num_stages=stages,
        )
    if split > 1:
        _amd_sum.run(
            p,
            output,
            m,
            n,
            split,
            triton.next_power_of_2(split),
            256,
            grid=(triton.cdiv(m * n, 256),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return output


@triton.jit
def _amd_case_0(A, Q, S, Z, P):
    _amd_gemv(
        A, Q, S, Z, P, 16, 32, 128, 128, 128, 1, 4, 1, 32, 1, 4, 1, 1, 8, 128
    )


@triton.jit
def _amd_case_1(A, Q, S, Z, P):
    _amd_gemv(
        A, Q, S, Z, P, 24, 64, 128, 32, 128, 1, 8, 1, 64, 1, 8, 1, 1, 8, 128
    )


@triton.jit
def _amd_case_2(A, Q, S, Z, P):
    _amd_gemv(
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
        32,
        128,
    )


@triton.jit
def _amd_case_3(A, Q, S, Z, P):
    _amd_gemv(
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
        16,
        16,
        128,
    )


@triton.jit
def _amd_reduce_case_3(P, OUT):
    _amd_sum(P, OUT, 1, 1024, 16, 16, 256)


@triton.jit
def _amd_case_4(A, Q, S, Z, P):
    _amd_gemv(
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
        8,
        64,
        128,
    )


@triton.jit
def _amd_reduce_case_4(P, OUT):
    _amd_sum(P, OUT, 1, 4096, 8, 8, 256)


@triton.jit
def _amd_case_5(A, Q, S, Z, P):
    _amd_prefetch(
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
        128,
        16,
        64,
    )


@triton.jit
def _amd_reduce_case_5(P, OUT):
    _amd_sum(P, OUT, 8, 1024, 16, 16, 256)


@triton.jit
def _amd_case_6(A, Q, S, Z, P):
    _amd_prefetch(
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
        128,
        16,
        64,
    )


@triton.jit
def _amd_reduce_case_6(P, OUT):
    _amd_sum(P, OUT, 8, 4096, 16, 16, 256)


@triton.jit
def _amd_case_7(A, Q, S, Z, P):
    _amd_group(
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
        16,
        32,
        32,
        8,
        False,
        False,
    )


@triton.jit
def _amd_reduce_case_7(P, OUT):
    _amd_sum(P, OUT, 64, 1024, 8, 8, 256)


@triton.jit
def _amd_case_8(A, Q, S, Z, P):
    _amd_group(
        A,
        Q,
        S,
        Z,
        P,
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
        16,
        32,
        32,
        8,
        False,
        False,
    )


@triton.jit
def _amd_reduce_case_8(P, OUT):
    _amd_sum(P, OUT, 64, 4096, 8, 8, 256)


@triton.jit
def _amd_case_9(A, Q, S, Z, P):
    _amd_group(
        A,
        Q,
        S,
        Z,
        P,
        512,
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
        16,
        32,
        32,
        8,
        False,
        False,
    )


@triton.jit
def _amd_reduce_case_9(P, OUT):
    _amd_sum(P, OUT, 512, 1024, 8, 8, 256)


@triton.jit
def _amd_case_10(A, Q, S, Z, P):
    _amd_group(
        A,
        Q,
        S,
        Z,
        P,
        512,
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
        16,
        32,
        32,
        8,
        False,
        False,
    )


@triton.jit
def _amd_reduce_case_10(P, OUT):
    _amd_sum(P, OUT, 512, 4096, 8, 8, 256)


def awq_amd(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    if not (
        input.dtype == torch.float32
        and scales.dtype == torch.float32
        and input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    ):
        return _awq_amd_general(input, qweight, scales, qzeros, split_k_iters)
    if m == 16 and n == 32 and k == 128 and groups == 1 and split_k_iters == 1:
        output = torch.empty(
            (16, 32), device=input.device, dtype=torch.float32
        )
        partial = output
        _amd_case_0.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if m == 24 and n == 64 and k == 128 and groups == 4 and split_k_iters == 8:
        output = torch.empty(
            (24, 64), device=input.device, dtype=torch.float32
        )
        partial = output
        _amd_case_1.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(192, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 32
        and n == 128
        and k == 128
        and groups == 1
        and split_k_iters == 1
    ):
        output = torch.empty(
            (32, 128), device=input.device, dtype=torch.float32
        )
        partial = output
        _amd_case_2.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(128, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 1
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (1, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (16, 1, 1024), device=input.device, dtype=torch.float32
        )
        _amd_case_3.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _amd_reduce_case_3.run(
            partial, output, grid=(4,), warmup=False, num_warps=4, num_stages=1
        )
        return output
    if (
        m == 1
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (1, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 1, 4096), device=input.device, dtype=torch.float32
        )
        _amd_case_4.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 8),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _amd_reduce_case_4.run(
            partial,
            output,
            grid=(16,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 8
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (8, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (16, 8, 1024), device=input.device, dtype=torch.float32
        )
        _amd_case_5.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(16, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _amd_reduce_case_5.run(
            partial,
            output,
            grid=(32,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 8
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (8, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (16, 8, 4096), device=input.device, dtype=torch.float32
        )
        _amd_case_6.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(64, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _amd_reduce_case_6.run(
            partial,
            output,
            grid=(128,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 64
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (64, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 64, 1024), device=input.device, dtype=torch.float32
        )
        _amd_case_7.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(128, 8),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
        _amd_reduce_case_7.run(
            partial,
            output,
            grid=(256,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 64
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (64, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 64, 4096), device=input.device, dtype=torch.float32
        )
        _amd_case_8.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(512, 8),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
        _amd_reduce_case_8.run(
            partial,
            output,
            grid=(1024,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 512
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (512, 1024), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 512, 1024), device=input.device, dtype=torch.float32
        )
        _amd_case_9.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(1024, 8),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
        _amd_reduce_case_9.run(
            partial,
            output,
            grid=(2048,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    if (
        m == 512
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        output = torch.empty(
            (512, 4096), device=input.device, dtype=torch.float32
        )
        partial = torch.empty(
            (8, 512, 4096), device=input.device, dtype=torch.float32
        )
        _amd_case_10.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            grid=(4096, 8),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
        _amd_reduce_case_10.run(
            partial,
            output,
            grid=(8192,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return output
    return _awq_amd_general(input, qweight, scales, qzeros, split_k_iters)


__all__ = ["awq_gemm"]
